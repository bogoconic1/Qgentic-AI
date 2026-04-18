"""Unit tests for ResearcherAgent."""

import pytest
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
