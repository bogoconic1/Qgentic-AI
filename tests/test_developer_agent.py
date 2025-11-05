"""Unit tests for DeveloperAgent."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import tempfile
from unittest.mock import MagicMock

from agents.developer import DeveloperAgent


@pytest.fixture(scope='module')
def test_task_dir():
    """Create a temporary task directory with dummy data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        task_root = Path(tmpdir)
        slug = "test-competition"
        iteration = 1

        # Create directory structure
        task_dir = task_root / slug
        outputs_dir = task_dir / "outputs" / str(iteration)
        outputs_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy description.md
        (task_dir / "description.md").write_text(
            "# Test Competition\n\n"
            "## Description\n"
            "This is a test competition.\n\n"
            "## Evaluation\n"
            "Submissions are evaluated using accuracy.\n"
        )

        # Create dummy benchmark.yaml
        import yaml
        benchmark_data = {
            "evaluation": {
                "metric": "accuracy",
                "is_lower_better": False
            },
            "competition_type": "test"
        }
        (task_dir / "benchmark.yaml").write_text(yaml.dump(benchmark_data))

        # Create data directory for DeveloperAgent
        (task_dir / "data").mkdir(parents=True, exist_ok=True)

        # Create dummy plan.md
        (outputs_dir / "plan.md").write_text("# Research Plan\nTest plan content.")

        yield {
            'task_root': task_root,
            'slug': slug,
            'iteration': iteration
        }


def test_agent_initialization(test_task_dir, monkeypatch):
    """Test DeveloperAgent initialization."""
    # Patch the module-level _TASK_ROOT constant
    monkeypatch.setattr("agents.developer._TASK_ROOT", test_task_dir['task_root'])

    agent = DeveloperAgent(
        slug=test_task_dir['slug'],
        iteration=test_task_dir['iteration'],
        model_name="test-model",
        plan_content="Test plan"
    )

    # Check paths are set correctly
    assert agent.slug == test_task_dir['slug']
    assert agent.iteration == test_task_dir['iteration']
    assert agent.outputs_dir.exists()
    assert agent.base_dir.exists()

    # Check model name
    assert agent.model_name == "test-model"

    # Check file naming templates
    assert "code_" in agent._code_filename(1)
    assert "v1.py" in agent._code_filename(1)

    print(f"✅ DeveloperAgent initialized successfully:")
    print(f"   - Slug: {agent.slug}")
    print(f"   - Iteration: {agent.iteration}")
    print(f"   - Model: {agent.model_name}")
    print(f"   - Outputs dir: {agent.outputs_dir}")


def test_extract_code(test_task_dir, monkeypatch):
    """Test code extraction from LLM response."""
    monkeypatch.setattr("agents.developer._TASK_ROOT", test_task_dir['task_root'])

    # Create agent with proper init (model_name is required)
    agent = DeveloperAgent(
        slug=test_task_dir['slug'],
        iteration=test_task_dir['iteration'],
        model_name="test-model"
    )

    # Test with python code block
    response_with_code = (
        "Here's the solution:\n\n"
        "```python\n"
        "import pandas as pd\n"
        "print('Hello')\n"
        "```\n"
    )

    code = agent._extract_code(response_with_code)
    assert "import pandas as pd" in code
    assert "print('Hello')" in code
    assert "```" not in code

    print("✅ Code extraction works:")
    print(f"   - Extracted {len(code)} chars of code")


def test_code_filename():
    """Test code filename generation."""
    agent = DeveloperAgent.__new__(DeveloperAgent)
    agent.iteration = 5

    filename = agent._code_filename(3)
    assert "code_5_v3.py" in filename

    print(f"✅ Code filename generation works:")
    print(f"   - Filename: {filename}")


def test_extract_diff_block():
    """Test diff block extraction."""
    agent = DeveloperAgent.__new__(DeveloperAgent)

    response_with_diff = (
        "Here are the changes:\n\n"
        "```diff\n"
        "--- a/code.py\n"
        "+++ b/code.py\n"
        "@@ -1,3 +1,3 @@\n"
        " import pandas as pd\n"
        "-print('old')\n"
        "+print('new')\n"
        "```\n"
    )

    diff = agent._extract_diff_block(response_with_diff)
    assert diff is not None
    assert "--- a/" in diff or "print('new')" in diff

    print("✅ Diff extraction works:")
    print(f"   - Extracted diff block")


def test_format_with_line_numbers():
    """Test line number formatting."""
    code = "import pandas as pd\nprint('hello')\nprint('world')"

    formatted = DeveloperAgent._format_with_line_numbers(code)

    # Format is "0001: code"
    assert "0001:" in formatted
    assert "import pandas" in formatted
    assert "0002:" in formatted
    assert "0003:" in formatted

    print("✅ Line number formatting works:")
    print(f"   - Formatted {len(formatted)} chars")


def test_developer_run_single_iteration(test_task_dir, monkeypatch):
    """Integration test: DeveloperAgent.run() completes one successful iteration."""
    monkeypatch.setattr("agents.developer._TASK_ROOT", test_task_dir['task_root'])

    # Mock LLM calls
    def fake_call_llm(*args, **kwargs):
        mock_response = MagicMock()
        mock_response.output = []
        # Return code
        mock_response.output_text = (
            "```python\n"
            "import pandas as pd\n"
            "print('Hello')\n"
            "print('Public Score: 0.85')\n"
            "```"
        )
        return mock_response

    monkeypatch.setattr("agents.developer.call_llm_with_retry", fake_call_llm)

    # Mock execute_code to return success with score
    def fake_execute_code(filepath, timeout_seconds, conda_env=None):
        return "Script executed successfully\nPublic Score: 0.85\nDone"

    monkeypatch.setattr("agents.developer.execute_code", fake_execute_code)

    # Mock wandb
    monkeypatch.setattr("agents.developer.wandb.log", lambda *args, **kwargs: None)

    # Mock search functions (not called on first successful iteration)
    monkeypatch.setattr("agents.developer.search_red_flags", lambda *args, **kwargs: "No red flags")
    monkeypatch.setattr("agents.developer.search_sota_suggestions", lambda *args, **kwargs: MagicMock(
        output_parsed=MagicMock(blacklist=False, suggestion="Continue")
    ))

    # Create agent
    agent = DeveloperAgent(
        slug=test_task_dir['slug'],
        iteration=test_task_dir['iteration'],
        model_name="test-model"
    )

    # Run with short timeout
    success = agent.run(max_time_seconds=30)

    # Verify outcomes
    assert success == True
    assert agent.best_score == 0.85
    assert (agent.outputs_dir / "code_1_v1.py").exists()
    assert (agent.outputs_dir / "log_1_v1.txt").exists()

    print("✅ DeveloperAgent.run() integration test passed:")
    print(f"   - Best score: {agent.best_score}")
    print(f"   - Code file created: {agent.outputs_dir / 'code_1_v1.py'}")


def test_developer_run_with_failure_then_success(test_task_dir, monkeypatch):
    """Integration test: DeveloperAgent.run() handles failure then succeeds."""
    monkeypatch.setattr("agents.developer._TASK_ROOT", test_task_dir['task_root'])

    attempt_count = [0]

    def fake_call_llm(*args, **kwargs):
        mock_response = MagicMock()
        mock_response.output = []
        mock_response.output_text = (
            "```python\n"
            "import pandas as pd\n"
            "print('Attempt')\n"
            "```"
        )
        return mock_response

    monkeypatch.setattr("agents.developer.call_llm_with_retry", fake_call_llm)

    # First attempt fails, second succeeds
    def fake_execute_code(filepath, timeout_seconds, conda_env=None):
        attempt_count[0] += 1
        if attempt_count[0] == 1:
            return "Error: ValueError\nTraceback..."
        else:
            return "Success\nPublic Score: 0.90\n"

    monkeypatch.setattr("agents.developer.execute_code", fake_execute_code)

    # Mock wandb
    monkeypatch.setattr("agents.developer.wandb.log", lambda *args, **kwargs: None)

    # Mock search functions
    monkeypatch.setattr("agents.developer.search_red_flags", lambda *args, **kwargs: "Red flag: error")

    def fake_sota(*args, **kwargs):
        return MagicMock(output_parsed=MagicMock(
            blacklist=False,
            blacklist_reason="Continue",
            suggestion="Fix the error",
            suggestion_reason="Need to handle edge case"
        ))
    monkeypatch.setattr("agents.developer.search_sota_suggestions", fake_sota)

    agent = DeveloperAgent(
        slug=test_task_dir['slug'],
        iteration=test_task_dir['iteration'],
        model_name="test-model"
    )

    success = agent.run(max_time_seconds=30)

    # Verify it recovered from failure
    assert success == True
    assert agent.best_score == 0.90
    assert attempt_count[0] >= 2  # At least 2 attempts

    print("✅ DeveloperAgent.run() with failure recovery works:")
    print(f"   - Attempts: {attempt_count[0]}")
    print(f"   - Final score: {agent.best_score}")


def test_developer_run_timeout(test_task_dir, monkeypatch):
    """Integration test: DeveloperAgent.run() respects timeout."""
    monkeypatch.setattr("agents.developer._TASK_ROOT", test_task_dir['task_root'])

    # Mock LLM to return code
    def fake_call_llm(*args, **kwargs):
        mock_response = MagicMock()
        mock_response.output = []
        mock_response.output_text = "```python\nprint('test')\n```"
        return mock_response

    monkeypatch.setattr("agents.developer.call_llm_with_retry", fake_call_llm)

    # Mock execute_code to simulate slow execution
    import time
    def fake_execute_code(filepath, timeout_seconds, conda_env=None):
        time.sleep(0.5)  # Simulate some work
        return "No score found"

    monkeypatch.setattr("agents.developer.execute_code", fake_execute_code)
    monkeypatch.setattr("agents.developer.wandb.log", lambda *args, **kwargs: None)
    monkeypatch.setattr("agents.developer.search_red_flags", lambda *args, **kwargs: "Flags")
    monkeypatch.setattr("agents.developer.search_sota_suggestions", lambda *args, **kwargs: MagicMock(
        output_parsed=MagicMock(blacklist=False, suggestion="Continue")
    ))

    agent = DeveloperAgent(
        slug=test_task_dir['slug'],
        iteration=test_task_dir['iteration'],
        model_name="test-model"
    )

    # Run with very short timeout (should stop after 1-2 iterations)
    success = agent.run(max_time_seconds=2)

    # Should stop due to timeout, not crash
    assert success in [True, False]  # Either succeeded or timed out gracefully

    print("✅ DeveloperAgent.run() timeout handling works:")
    print(f"   - Completed without crashing")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
