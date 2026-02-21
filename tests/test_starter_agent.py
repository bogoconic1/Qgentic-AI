"""Unit tests for StarterAgent."""

import pytest
import json
import tempfile
from pathlib import Path

from agents.starter import StarterAgent


@pytest.fixture(autouse=True)
def patch_llm_calls(monkeypatch):
    """Mock all LLM API calls to avoid actual network requests."""

    def fake_call_llm(*args, **kwargs):
        from schemas.starter import StarterSuggestions

        return StarterSuggestions(
            task_types=["nlp"],
            task_summary="Text classification task for sentiment analysis",
        )

    monkeypatch.setattr("agents.starter.call_llm", fake_call_llm)


@pytest.fixture(scope="function")
def test_task_dir():
    """Create a temporary task directory with dummy data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        task_root = Path(tmpdir)
        slug = "test-competition"
        iteration = 1

        task_dir = task_root / slug
        outputs_dir = task_dir / "outputs" / str(iteration)
        outputs_dir.mkdir(parents=True, exist_ok=True)

        (task_dir / "description.md").write_text(
            "# Test Competition\n\n"
            "## Description\n"
            "This is a test competition for text classification.\n\n"
            "## Evaluation\n"
            "Submissions are evaluated using accuracy.\n"
        )

        yield {
            "task_root": task_root,
            "slug": slug,
            "iteration": iteration,
        }


def test_agent_initialization(test_task_dir, monkeypatch):
    """Test StarterAgent initialization."""
    monkeypatch.setattr("agents.starter._TASK_ROOT", test_task_dir["task_root"])

    agent = StarterAgent(test_task_dir["slug"], test_task_dir["iteration"])

    assert agent.slug == test_task_dir["slug"]
    assert agent.iteration == test_task_dir["iteration"]
    assert agent.outputs_dir.exists()
    assert agent.json_path.name == "starter_suggestions.json"
    assert agent.text_path.name == "starter_suggestions.txt"


def test_run_starter_agent(test_task_dir, monkeypatch):
    """Test running StarterAgent with mocked LLM call."""
    monkeypatch.setattr("agents.starter._TASK_ROOT", test_task_dir["task_root"])

    agent = StarterAgent(test_task_dir["slug"], test_task_dir["iteration"])
    agent.run()

    assert agent.json_path.exists()

    with open(agent.json_path, "r") as f:
        suggestions = json.load(f)

    assert "task_types" in suggestions
    assert "task_summary" in suggestions
    assert isinstance(suggestions["task_types"], list)
    assert len(suggestions["task_types"]) > 0
    assert "nlp" in suggestions["task_types"]
    assert len(suggestions["task_summary"]) > 0


def test_task_type_literal_constraint():
    """Test that Pydantic Literal constraint rejects invalid task types."""
    from pydantic import ValidationError
    from schemas.starter import StarterSuggestions

    # Valid task types should pass
    valid = StarterSuggestions(
        task_types=["nlp", "tabular"],
        task_summary="Multi-modal task",
    )
    assert valid.task_types == ["nlp", "tabular"]

    # Invalid task type should be rejected
    with pytest.raises(ValidationError):
        StarterSuggestions(
            task_types=["NLP"],
            task_summary="Should fail",
        )

    with pytest.raises(ValidationError):
        StarterSuggestions(
            task_types=["invalid_type"],
            task_summary="Should fail",
        )
