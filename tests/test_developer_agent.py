"""Unit tests for DeveloperAgent."""

import pytest
import json
import tempfile
from pathlib import Path

from agents.developer import DeveloperAgent


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
            "This is a test competition.\n\n"
            "## Evaluation\n"
            "Submissions are evaluated using accuracy.\n"
        )

        import yaml

        benchmark_data = {
            "evaluation": {"metric": "accuracy", "is_lower_better": False},
            "competition_type": "test",
        }
        (task_dir / "benchmark.yaml").write_text(yaml.dump(benchmark_data))

        (task_dir / "data").mkdir(parents=True, exist_ok=True)

        (outputs_dir / "plan.md").write_text("# Research Plan\nTest plan content.")

        cv_splits = {"fold_0": {"train": [0, 1, 2], "val": [3, 4]}}
        (task_dir / "cv_splits.json").write_text(json.dumps(cv_splits))
        (task_dir / "metric.py").write_text("def score(y_true, y_pred): return 0.5")

        yield {
            "task_root": task_root,
            "slug": slug,
            "iteration": iteration,
        }


def test_agent_initialization(test_task_dir, monkeypatch):
    """Test DeveloperAgent initialization."""
    monkeypatch.setattr("agents.developer._TASK_ROOT", test_task_dir["task_root"])

    agent = DeveloperAgent(
        slug=test_task_dir["slug"],
        iteration=test_task_dir["iteration"],
        model_name="test-model",
        plan_content="Test plan",
    )

    assert agent.slug == test_task_dir["slug"]
    assert agent.iteration == test_task_dir["iteration"]
    assert agent.outputs_dir.exists()
    assert agent.base_dir.exists()
    assert agent.model_name == "test-model"


def test_extract_code(test_task_dir, monkeypatch):
    """Test code extraction from LLM response using extract_python_code utility."""
    from utils.code_utils import extract_python_code

    response_with_code = (
        "Here's the solution:\n\n"
        "```python\n"
        "import pandas as pd\n"
        "print('Hello')\n"
        "```\n"
    )

    code = extract_python_code(response_with_code)
    assert "import pandas as pd" in code
    assert "print('Hello')" in code
    assert "```" not in code
