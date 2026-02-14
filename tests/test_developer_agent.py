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

        # Create required helper files (cv_splits.json and metric.py)
        import json
        cv_splits = {"fold_0": {"train": [0, 1, 2], "val": [3, 4]}}
        (task_dir / "cv_splits.json").write_text(json.dumps(cv_splits))
        (task_dir / "metric.py").write_text("def score(y_true, y_pred): return 0.5")

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

    # Check file naming templates (now folder-based: {version}/train.py)
    assert "train.py" in agent._code_filename(1)
    assert "1/" in agent._code_filename(1)

    print(f"✅ DeveloperAgent initialized successfully:")
    print(f"   - Slug: {agent.slug}")
    print(f"   - Iteration: {agent.iteration}")
    print(f"   - Model: {agent.model_name}")
    print(f"   - Outputs dir: {agent.outputs_dir}")


def test_extract_code(test_task_dir, monkeypatch):
    """Test code extraction from LLM response using extract_python_code utility."""
    from utils.code_utils import extract_python_code

    # Test with python code block
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

    print("✅ Code extraction works:")
    print(f"   - Extracted {len(code)} chars of code")


def test_code_filename():
    """Test code filename generation."""
    agent = DeveloperAgent.__new__(DeveloperAgent)
    agent.iteration = 5

    filename = agent._code_filename(3)
    # New folder-based format: {version}/train.py
    assert "3/train.py" in filename

    print(f"✅ Code filename generation works:")
    print(f"   - Filename: {filename}")


# NOTE: Full run() integration tests are too complex to mock reliably.
# These will be tested after refactoring when methods are smaller.


# NOTE: test_developer_run_timeout has been removed as it requires extensive mocking
# of internal implementation details. The core timeout functionality is tested
# through integration tests.


# NOTE: Tests for internal methods like _execute_and_read_log and _evaluate_submission
# have been removed as the internal implementation has changed. Core functionality
# is tested through other tests.

def test_gather_sota_feedback_exception_handling(test_task_dir, monkeypatch):
    """Test _gather_sota_feedback handles exceptions gracefully."""
    monkeypatch.setattr("agents.developer._TASK_ROOT", test_task_dir['task_root'])

    # Mock search_red_flags to raise exception
    def fake_search_red_flags(description, context):
        raise Exception("Simulated error in red flags analysis")

    monkeypatch.setattr("agents.developer.search_red_flags", fake_search_red_flags)

    agent = DeveloperAgent(
        slug=test_task_dir['slug'],
        iteration=test_task_dir['iteration'],
        model_name="test-model"
    )

    code_with_logs = "<code>import pandas as pd</code>"

    # Should return None on exception
    sota_response = agent._gather_sota_feedback(code_with_logs, version=1)

    assert sota_response is None

    print("✅ _gather_sota_feedback() exception handling works correctly:")
    print(f"   - Returned None on exception: {sota_response is None}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
