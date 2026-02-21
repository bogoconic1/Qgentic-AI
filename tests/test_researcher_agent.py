"""Unit tests for ResearcherAgent."""

import pytest
import json
import tempfile
from pathlib import Path

from agents.researcher import ResearcherAgent


@pytest.fixture(scope="module")
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
            "This is a test competition for research planning.\n\n"
            "## Data\n"
            "The dataset contains training and test data.\n\n"
            "## Evaluation\n"
            "Submissions are evaluated using F1 score.\n"
        )

        starter_data = {
            "task_types": ["nlp"],
            "task_summary": "Text classification task",
        }
        (outputs_dir / "starter_suggestions.json").write_text(
            json.dumps(starter_data)
        )

        yield {
            "task_root": task_root,
            "slug": slug,
            "iteration": iteration,
        }


def test_agent_initialization(test_task_dir, monkeypatch):
    """Test ResearcherAgent initialization."""
    monkeypatch.setattr("agents.researcher._TASK_ROOT", test_task_dir["task_root"])

    agent = ResearcherAgent(
        test_task_dir["slug"],
        test_task_dir["iteration"],
    )

    assert agent.slug == test_task_dir["slug"]
    assert agent.iteration == test_task_dir["iteration"]
    assert agent.outputs_dir.exists()
    assert agent.media_dir.exists()
    assert agent.external_dir.exists()

    assert len(agent.description) > 0
    assert "test competition" in agent.description.lower()


def test_read_starter_suggestions(test_task_dir, monkeypatch):
    """Test that starter suggestions are read correctly."""
    monkeypatch.setattr("agents.researcher._TASK_ROOT", test_task_dir["task_root"])

    agent = ResearcherAgent(
        test_task_dir["slug"],
        test_task_dir["iteration"],
    )

    starter_text = agent._read_starter_suggestions()

    assert "<task_types>" in starter_text
    assert "nlp" in starter_text
    assert "<task_summary>" in starter_text
    assert "Text classification" in starter_text
