"""Unit tests for ResearcherAgent."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import json
import tempfile
from unittest.mock import MagicMock
from types import SimpleNamespace

from agents.researcher import ResearcherAgent


@pytest.fixture(scope='module')
def test_task_dir():
    """Create a temporary task directory with dummy data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        task_root = Path(tmpdir)
        slug = "test-competition"
        iteration = 1
        run_id = 1

        # Create directory structure
        task_dir = task_root / slug
        outputs_dir = task_dir / "outputs" / str(iteration)
        outputs_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy description.md
        (task_dir / "description.md").write_text(
            "# Test Competition\n\n"
            "## Description\n"
            "This is a test competition for research planning.\n\n"
            "## Data\n"
            "The dataset contains training and test data.\n\n"
            "## Evaluation\n"
            "Submissions are evaluated using F1 score.\n"
        )

        # Create dummy starter_suggestions.json (required by build_plan)
        starter_data = {
            "task_types": ["nlp"],
            "task_summary": "Text classification task"
        }
        (outputs_dir / "starter_suggestions.json").write_text(json.dumps(starter_data))

        yield {
            'task_root': task_root,
            'slug': slug,
            'iteration': iteration,
            'run_id': run_id
        }


def test_agent_initialization(test_task_dir, monkeypatch):
    """Test ResearcherAgent initialization."""
    # Patch the module-level _TASK_ROOT constant
    monkeypatch.setattr("agents.researcher._TASK_ROOT", test_task_dir['task_root'])

    agent = ResearcherAgent(
        test_task_dir['slug'],
        test_task_dir['iteration'],
        test_task_dir['run_id']
    )

    # Check paths are set correctly
    assert agent.slug == test_task_dir['slug']
    assert agent.iteration == test_task_dir['iteration']
    assert agent.run_id == test_task_dir['run_id']
    assert agent.outputs_dir.exists()
    assert agent.media_dir.exists()
    assert agent.external_dir.exists()

    # Check description was loaded
    assert len(agent.description) > 0
    assert "test competition" in agent.description.lower()

    # Check AB test history is initialized
    assert isinstance(agent.ab_test_history, list)
    assert len(agent.ab_test_history) == 0

    print(f"✅ ResearcherAgent initialized successfully:")
    print(f"   - Slug: {agent.slug}")
    print(f"   - Iteration: {agent.iteration}")
    print(f"   - Run ID: {agent.run_id}")
    print(f"   - Outputs dir: {agent.outputs_dir}")
    print(f"   - Media dir: {agent.media_dir}")
    print(f"   - External data dir: {agent.external_dir}")
    print(f"   - Description length: {len(agent.description)} chars")


def test_build_plan_with_tool_calls(test_task_dir, monkeypatch):
    """Test build_plan method with mocked tool calls."""
    # Patch the module-level _TASK_ROOT constant
    monkeypatch.setattr("agents.researcher._TASK_ROOT", test_task_dir['task_root'])

    # Track call count to return different responses
    call_count = [0]

    def fake_call_llm_with_retry(*args, **kwargs):
        """Return different responses based on call count."""
        call_count[0] += 1
        mock_response = MagicMock()

        if call_count[0] == 1:
            # First call: Make a tool call to ask_eda
            tool_call = SimpleNamespace(
                type="function_call",
                name="ask_eda",
                call_id="call_123",
                arguments=json.dumps({"question": "What is the shape of the training data?"})
            )
            mock_response.output = [tool_call]
            mock_response.output_text = None
        elif call_count[0] == 2:
            # Second call: Return final plan (no tool calls)
            mock_response.output = []
            mock_response.output_text = (
                "# Research Plan\n\n"
                "## Data Understanding\n"
                "The training data has 10,000 samples.\n\n"
                "## Modeling Strategy\n"
                "1. Use BERT for text classification\n"
                "2. Fine-tune with cross-validation\n"
                "3. Ensemble top 3 models\n"
            )
        else:
            # Fallback
            mock_response.output = []
            mock_response.output_text = "Final plan fallback"

        return mock_response

    def fake_ask_eda(question, description, data_path, previous_ab_tests=None):
        """Mock ask_eda tool."""
        return "The training data has shape (10000, 5). Contains text and labels."

    def fake_get_tools(max_parallel_workers: int = 1):
        """Mock get_tools to return tool definitions."""
        return [
            {
                "type": "function",
                "name": "ask_eda",
                "description": "Ask domain expert about the data",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"}
                    }
                }
            }
        ]

    # Apply monkeypatches
    monkeypatch.setattr("agents.researcher.call_llm_with_retry", fake_call_llm_with_retry)
    monkeypatch.setattr("agents.researcher.ask_eda", fake_ask_eda)
    monkeypatch.setattr("agents.researcher.get_tools", fake_get_tools)

    agent = ResearcherAgent(
        test_task_dir['slug'],
        test_task_dir['iteration'],
        test_task_dir['run_id']
    )

    # Run build_plan
    plan = agent.build_plan(max_steps=5)

    # Verify plan was generated
    assert plan is not None
    assert len(plan) > 0
    assert "Research Plan" in plan or "research" in plan.lower()

    # Verify LLM was called multiple times (once for tool call, once for final plan)
    assert call_count[0] == 2

    print("✅ build_plan with tool calls works correctly:")
    print(f"   - LLM called {call_count[0]} times")
    print(f"   - Plan length: {len(plan)} chars")
    print(f"   - Plan preview: {plan[:100]}...")


def test_build_plan_direct_output(test_task_dir, monkeypatch):
    """Test build_plan when LLM returns plan immediately without tool calls."""
    # Patch the module-level _TASK_ROOT constant
    monkeypatch.setattr("agents.researcher._TASK_ROOT", test_task_dir['task_root'])

    def fake_call_llm_with_retry(*args, **kwargs):
        """Return final plan immediately."""
        mock_response = MagicMock()
        mock_response.output = []  # No tool calls
        mock_response.output_text = (
            "# Quick Research Plan\n\n"
            "Based on the description, this is a simple classification task.\n"
            "Use XGBoost with default parameters."
        )
        return mock_response

    def fake_get_tools(max_parallel_workers: int = 1):
        return []

    monkeypatch.setattr("agents.researcher.call_llm_with_retry", fake_call_llm_with_retry)
    monkeypatch.setattr("agents.researcher.get_tools", fake_get_tools)

    agent = ResearcherAgent(
        test_task_dir['slug'],
        test_task_dir['iteration'],
        test_task_dir['run_id']
    )

    # Run build_plan
    plan = agent.build_plan(max_steps=5)

    # Verify plan
    assert plan is not None
    assert len(plan) > 0
    assert "Research Plan" in plan or "XGBoost" in plan

    print("✅ build_plan with direct output works correctly:")
    print(f"   - Plan length: {len(plan)} chars")
    print(f"   - Plan preview: {plan[:80]}...")


def test_read_starter_suggestions(test_task_dir, monkeypatch):
    """Test that starter suggestions are read correctly."""
    # Patch the module-level _TASK_ROOT constant
    monkeypatch.setattr("agents.researcher._TASK_ROOT", test_task_dir['task_root'])

    agent = ResearcherAgent(
        test_task_dir['slug'],
        test_task_dir['iteration'],
        test_task_dir['run_id']
    )

    # Call internal method
    starter_text = agent._read_starter_suggestions()

    # Verify format
    assert "<task_types>" in starter_text
    assert "nlp" in starter_text
    assert "<task_summary>" in starter_text
    assert "Text classification" in starter_text

    print("✅ Starter suggestions read correctly:")
    print(f"   - Starter text length: {len(starter_text)} chars")
    print(f"   - Contains task_types: {'task_types' in starter_text}")
    print(f"   - Contains task_summary: {'task_summary' in starter_text}")


# NOTE: Complex build_plan() integration tests with tool calls are too difficult to mock.
# These will be tested after refactoring when methods are smaller.


def test_researcher_build_plan_invalid_tool_arguments(test_task_dir, monkeypatch):
    """Integration test: build_plan() handles invalid tool arguments gracefully."""
    monkeypatch.setattr("agents.researcher._TASK_ROOT", test_task_dir['task_root'])

    call_count = [0]

    def fake_call_llm(*args, **kwargs):
        call_count[0] += 1
        mock_response = MagicMock()

        if call_count[0] == 1:
            # First call: Return tool call with invalid JSON arguments
            tool_call = SimpleNamespace(
                type="function_call",
                name="ask_eda",
                call_id="call_1",
                arguments="{ invalid json }"  # Malformed JSON
            )
            mock_response.output = [tool_call]
            mock_response.output_text = None
        elif call_count[0] == 2:
            # Second call: Return tool call with missing required field
            tool_call = SimpleNamespace(
                type="function_call",
                name="ask_eda",
                call_id="call_2",
                arguments=json.dumps({"wrong_field": "value"})  # Missing 'question' field
            )
            mock_response.output = [tool_call]
            mock_response.output_text = None
        else:
            # Final call: Return plan after errors
            mock_response.output = []
            mock_response.output_text = (
                "# Research Plan\n\n"
                "Despite tool failures, here's a plan based on the description.\n"
            )

        return mock_response

    def fake_ask_eda(*args, **kwargs):
        # This should not be called due to invalid arguments
        raise Exception("Should not reach here - tool should fail before execution")

    def fake_get_tools(max_parallel_workers: int = 1):
        return [{"type": "function", "name": "ask_eda"}]

    monkeypatch.setattr("agents.researcher.call_llm_with_retry", fake_call_llm)
    monkeypatch.setattr("agents.researcher.ask_eda", fake_ask_eda)
    monkeypatch.setattr("agents.researcher.get_tools", fake_get_tools)

    agent = ResearcherAgent(
        test_task_dir['slug'],
        test_task_dir['iteration'],
        test_task_dir['run_id']
    )

    # Should handle invalid arguments gracefully
    plan = agent.build_plan(max_steps=10)

    # Should still return a plan despite tool argument errors
    assert plan is not None
    assert len(plan) > 0
    assert "plan" in plan.lower() or "research" in plan.lower()

    print("✅ ResearcherAgent.build_plan() handles invalid tool arguments:")
    print(f"   - Recovered from malformed JSON")
    print(f"   - Returned plan: {len(plan)} chars")


def test_researcher_build_plan_empty_tool_results(test_task_dir, monkeypatch):
    """Integration test: build_plan() handles empty/None tool results."""
    monkeypatch.setattr("agents.researcher._TASK_ROOT", test_task_dir['task_root'])

    call_count = [0]

    def fake_call_llm(*args, **kwargs):
        call_count[0] += 1
        mock_response = MagicMock()

        if call_count[0] == 1:
            # First call: Request tool that returns empty
            tool_call = SimpleNamespace(
                type="function_call",
                name="ask_eda",
                call_id="call_1",
                arguments=json.dumps({"question": "What's the data?"})
            )
            mock_response.output = [tool_call]
            mock_response.output_text = None
        else:
            # After empty result, return plan
            mock_response.output = []
            mock_response.output_text = (
                "# Research Plan\n\n"
                "Based on limited information, proceed with baseline approach.\n"
            )

        return mock_response

    def fake_ask_eda(*args, **kwargs):
        # Return empty string
        return ""

    def fake_get_tools(max_parallel_workers: int = 1):
        return [{"type": "function", "name": "ask_eda"}]

    monkeypatch.setattr("agents.researcher.call_llm_with_retry", fake_call_llm)
    monkeypatch.setattr("agents.researcher.ask_eda", fake_ask_eda)
    monkeypatch.setattr("agents.researcher.get_tools", fake_get_tools)

    agent = ResearcherAgent(
        test_task_dir['slug'],
        test_task_dir['iteration'],
        test_task_dir['run_id']
    )

    # Should handle empty tool results gracefully
    plan = agent.build_plan(max_steps=5)

    # Should still return a plan
    assert plan is not None
    assert len(plan) > 0

    print("✅ ResearcherAgent.build_plan() handles empty tool results:")
    print(f"   - Continued after empty result")
    print(f"   - Plan length: {len(plan)} chars")


# Tests for extracted methods after refactoring

def test_execute_tool_call_ask_eda(test_task_dir, monkeypatch):
    """Test _execute_tool_call with ask_eda tool."""
    monkeypatch.setattr("agents.researcher._TASK_ROOT", test_task_dir['task_root'])

    # Mock ask_eda
    def fake_ask_eda(question, description, data_path, previous_ab_tests=None):
        return f"Analysis for: {question}\nData shape: (1000, 10)"

    monkeypatch.setattr("agents.researcher.ask_eda", fake_ask_eda)

    agent = ResearcherAgent(
        test_task_dir['slug'],
        test_task_dir['iteration'],
        test_task_dir['run_id']
    )

    # Create mock tool call item
    tool_call = SimpleNamespace(
        type="function_call",
        name="ask_eda",
        call_id="call_123",
        arguments=json.dumps({"question": "What is the data shape?"})
    )

    input_list = []
    agent._execute_tool_call(tool_call, input_list)

    # Verify tool result was appended
    assert len(input_list) == 1
    assert input_list[0]["type"] == "function_call_output"
    assert input_list[0]["call_id"] == "call_123"
    output = json.loads(input_list[0]["output"])
    assert "insights" in output
    assert "Data shape" in output["insights"]

    print("✅ _execute_tool_call() with ask_eda works correctly:")
    print(f"   - Tool result appended to input_list")
    print(f"   - Insights length: {len(output['insights'])} chars")


def test_execute_tool_call_run_ab_test(test_task_dir, monkeypatch):
    """Test _execute_tool_call with run_ab_test tool stores AB test history."""
    monkeypatch.setattr("agents.researcher._TASK_ROOT", test_task_dir['task_root'])

    # Mock ask_eda
    def fake_ask_eda(question, description, data_path, previous_ab_tests=None, **kwargs):
        return "AB test result: Model A outperforms Model B"

    monkeypatch.setattr("agents.researcher.ask_eda", fake_ask_eda)

    agent = ResearcherAgent(
        test_task_dir['slug'],
        test_task_dir['iteration'],
        test_task_dir['run_id']
    )

    # Create eda_temp_1.py file to simulate executed code (index 0, suffix _1)
    eda_temp_path = agent.base_dir / "eda_temp_1.py"
    eda_temp_path.write_text("import pandas as pd\nprint('AB test code')")

    # Create mock tool call item
    tool_call = SimpleNamespace(
        type="function_call",
        name="run_ab_test",
        call_id="call_456",
        arguments=json.dumps({"questions": ["Compare Model A vs Model B"]})
    )

    input_list = []
    initial_ab_test_count = len(agent.ab_test_history)

    agent._execute_tool_call(tool_call, input_list)

    # Verify AB test was stored in history
    assert len(agent.ab_test_history) == initial_ab_test_count + 1
    assert agent.ab_test_history[-1]['question'] == "Compare Model A vs Model B"
    assert "import pandas" in agent.ab_test_history[-1]['code']

    print("✅ _execute_tool_call() with run_ab_test works correctly:")
    print(f"   - AB test stored in history")
    print(f"   - History size: {len(agent.ab_test_history)}")


def test_execute_tool_call_download_external_datasets(test_task_dir, monkeypatch):
    """Test _execute_tool_call with download_external_datasets tool."""
    monkeypatch.setattr("agents.researcher._TASK_ROOT", test_task_dir['task_root'])

    # Mock download_external_datasets
    def fake_download_external_datasets(q1, q2, q3, slug):
        return "Downloaded 3 external datasets:\n- Dataset A\n- Dataset B\n- Dataset C"

    monkeypatch.setattr("agents.researcher.download_external_datasets", fake_download_external_datasets)

    agent = ResearcherAgent(
        test_task_dir['slug'],
        test_task_dir['iteration'],
        test_task_dir['run_id']
    )

    # Create mock tool call item
    tool_call = SimpleNamespace(
        type="function_call",
        name="download_external_datasets",
        call_id="call_789",
        arguments=json.dumps({
            "question_1": "external data about topic",
            "question_2": "datasets related to topic",
            "question_3": "topic external sources"
        })
    )

    input_list = []
    agent._execute_tool_call(tool_call, input_list)

    # Verify tool result was appended
    assert len(input_list) == 1
    assert input_list[0]["type"] == "function_call_output"
    assert input_list[0]["call_id"] == "call_789"
    output = json.loads(input_list[0]["output"])
    assert "results" in output
    assert "Downloaded 3 external datasets" in output["results"]

    print("✅ _execute_tool_call() with download_external_datasets works correctly:")
    print(f"   - Tool result appended to input_list")
    print(f"   - Results: {output['results'][:50]}...")


def test_execute_tool_call_invalid_arguments(test_task_dir, monkeypatch):
    """Test _execute_tool_call handles invalid JSON arguments gracefully."""
    monkeypatch.setattr("agents.researcher._TASK_ROOT", test_task_dir['task_root'])

    agent = ResearcherAgent(
        test_task_dir['slug'],
        test_task_dir['iteration'],
        test_task_dir['run_id']
    )

    # Create mock tool call with invalid JSON
    tool_call = SimpleNamespace(
        type="function_call",
        name="ask_eda",
        call_id="call_error",
        arguments="{ invalid json }"
    )

    input_list = []
    agent._execute_tool_call(tool_call, input_list)

    # Should append error message
    assert len(input_list) == 1
    output = json.loads(input_list[0]["output"])
    assert "error occurred" in output["insights"].lower()

    print("✅ _execute_tool_call() with invalid arguments works correctly:")
    print(f"   - Error handled gracefully")
    print(f"   - Error message: {output['insights']}")


def test_execute_tool_call_missing_required_fields(test_task_dir, monkeypatch):
    """Test _execute_tool_call handles missing required fields for download_external_datasets."""
    monkeypatch.setattr("agents.researcher._TASK_ROOT", test_task_dir['task_root'])

    agent = ResearcherAgent(
        test_task_dir['slug'],
        test_task_dir['iteration'],
        test_task_dir['run_id']
    )

    # Create mock tool call with missing fields
    tool_call = SimpleNamespace(
        type="function_call",
        name="download_external_datasets",
        call_id="call_missing",
        arguments=json.dumps({
            "question_1": "only one question"
            # question_2 and question_3 missing
        })
    )

    input_list = []
    agent._execute_tool_call(tool_call, input_list)

    # Should return error message
    assert len(input_list) == 1
    output = json.loads(input_list[0]["output"])
    assert "results" in output
    assert "All 3 question phrasings are required" in output["results"]

    print("✅ _execute_tool_call() with missing fields works correctly:")
    print(f"   - Missing field error handled")
    print(f"   - Error message: {output['results'][:80]}...")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
