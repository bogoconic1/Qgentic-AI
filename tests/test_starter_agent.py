"""Unit tests for StarterAgent."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import json
import tempfile
from unittest.mock import MagicMock
from pydantic import BaseModel

from agents.starter import StarterAgent


@pytest.fixture(autouse=True)
def patch_llm_calls(monkeypatch):
    """Mock all LLM API calls to avoid actual network requests."""

    def fake_call_llm_with_retry(*args, **kwargs):
        """Return a mock response based on the text_format (Pydantic schema)."""
        from schemas.starter import StarterSuggestions

        mock_response = MagicMock()

        # Get the expected schema from text_format parameter
        text_format = kwargs.get('text_format')

        if text_format and issubclass(text_format, BaseModel):
            # Create a mock instance of the Pydantic model with dummy data
            if text_format.__name__ == 'StarterSuggestions':
                mock_parsed = text_format(
                    task_types=["nlp"],
                    task_summary="Text classification task for sentiment analysis"
                )
            else:
                # Generic fallback
                mock_parsed = text_format()

            mock_response.output_parsed = mock_parsed
            mock_response.output_text = "Mock LLM response text"
        else:
            mock_response.output_parsed = None
            mock_response.output_text = "Mock LLM response"

        return mock_response

    monkeypatch.setattr("agents.starter.call_llm_with_retry", fake_call_llm_with_retry)


@pytest.fixture(scope='module')
def test_task_dir():
    """Create a temporary task directory with dummy data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        task_root = Path(tmpdir)
        slug = "test-competition"
        iteration = 1

        # Create directory structure
        task_dir = task_root / slug
        task_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy description.md
        (task_dir / "description.md").write_text(
            "# Test Competition\n\n"
            "## Description\n"
            "This is a test competition for text classification.\n\n"
            "## Evaluation\n"
            "Submissions are evaluated using accuracy.\n"
        )

        yield {
            'task_root': task_root,
            'slug': slug,
            'iteration': iteration
        }


def test_agent_initialization(test_task_dir, monkeypatch):
    """Test StarterAgent initialization."""
    # Patch the module-level _TASK_ROOT constant
    monkeypatch.setattr("agents.starter._TASK_ROOT", test_task_dir['task_root'])

    agent = StarterAgent(test_task_dir['slug'], test_task_dir['iteration'])

    # Check paths are set correctly
    assert agent.slug == test_task_dir['slug']
    assert agent.iteration == test_task_dir['iteration']
    assert agent.outputs_dir.exists()
    assert agent.json_path.name == "starter_suggestions.json"
    assert agent.text_path.name == "starter_suggestions.txt"

    print(f"✅ StarterAgent initialized successfully:")
    print(f"   - Slug: {agent.slug}")
    print(f"   - Iteration: {agent.iteration}")
    print(f"   - Outputs dir: {agent.outputs_dir}")


def test_run_starter_agent(test_task_dir, monkeypatch):
    """Test running StarterAgent with mocked LLM call."""
    # Patch the module-level _TASK_ROOT constant
    monkeypatch.setattr("agents.starter._TASK_ROOT", test_task_dir['task_root'])

    agent = StarterAgent(test_task_dir['slug'], test_task_dir['iteration'])

    # Run the agent (should use mocked LLM call)
    agent.run()

    # Verify JSON output was created
    assert agent.json_path.exists()

    # Load and verify the output
    with open(agent.json_path, 'r') as f:
        suggestions = json.load(f)

    # Verify structure
    assert "task_types" in suggestions
    assert "task_summary" in suggestions

    # Verify content
    assert isinstance(suggestions["task_types"], list)
    assert len(suggestions["task_types"]) > 0
    assert "nlp" in suggestions["task_types"]
    assert len(suggestions["task_summary"]) > 0

    # Verify text output was created
    assert agent.text_path.exists()

    print("✅ StarterAgent run successfully:")
    print(f"   - Task types: {suggestions['task_types']}")
    print(f"   - Task summary: {suggestions['task_summary'][:60]}...")
    print(f"   - JSON output: {agent.json_path}")
    print(f"   - Text output: {agent.text_path}")


def test_task_type_normalization(test_task_dir, monkeypatch):
    """Test that task types are properly normalized."""
    # Patch the module-level _TASK_ROOT constant
    monkeypatch.setattr("agents.starter._TASK_ROOT", test_task_dir['task_root'])

    # Create a custom mock that returns uppercase NLP
    def fake_call_with_uppercase(*args, **kwargs):
        from schemas.starter import StarterSuggestions
        mock_response = MagicMock()
        text_format = kwargs.get('text_format')

        if text_format and text_format.__name__ == 'StarterSuggestions':
            mock_parsed = text_format(
                task_types=["NLP"],  # Uppercase - should be normalized
                task_summary="Natural language processing task"
            )
            mock_response.output_parsed = mock_parsed
            mock_response.output_text = "Mock response"
        return mock_response

    monkeypatch.setattr("agents.starter.call_llm_with_retry", fake_call_with_uppercase)

    agent = StarterAgent(test_task_dir['slug'], test_task_dir['iteration'])
    agent.run()

    # Load and verify the output
    with open(agent.json_path, 'r') as f:
        suggestions = json.load(f)

    # Task type should be normalized to lowercase
    assert "nlp" in suggestions["task_types"]
    assert "NLP" not in suggestions["task_types"]

    print("✅ Task type normalization works correctly:")
    print(f"   - Normalized task types: {suggestions['task_types']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
