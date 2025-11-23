"""Unit tests for EnsemblerAgent."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import tempfile

from agents.ensembler import EnsemblerAgent


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
            }
        }
        (task_dir / "benchmark.yaml").write_text(yaml.dump(benchmark_data))

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
    """Test EnsemblerAgent initialization."""
    # Patch the module-level _TASK_ROOT constant (in developer, not ensembler)
    monkeypatch.setattr("agents.developer._TASK_ROOT", test_task_dir['task_root'])

    # Create ensemble strategy
    strategy = {
        "strategy": "Weighted average ensemble of top 3 models",
        "models_needed": ["model1", "model2", "model3"]
    }

    baseline_metadata = {
        "baselines": [
            {"model": "model1", "score": 0.85},
            {"model": "model2", "score": 0.83},
            {"model": "model3", "score": 0.82}
        ]
    }

    agent = EnsemblerAgent(
        slug=test_task_dir['slug'],
        iteration=test_task_dir['iteration'],
        strategy_index=1,
        strategy=strategy,
        baseline_metadata=baseline_metadata,
        plan_content="Test ensemble plan"
    )

    # Check paths are set correctly
    assert agent.slug == test_task_dir['slug']
    assert agent.original_iteration == test_task_dir['iteration']
    assert agent.strategy_index == 1
    assert agent.outputs_dir.exists()

    # Check strategy
    assert agent.strategy == strategy
    assert "Weighted average" in agent.strategy["strategy"]

    # Check iteration naming (should be "1_1_ens")
    assert "_ens" in str(agent.iteration)

    # Check file naming includes strategy index (folder-based: {version}/train.py)
    code_file = agent._code_filename(1)
    assert "train.py" in code_file
    assert "1/" in code_file

    print(f"✅ EnsemblerAgent initialized successfully:")
    print(f"   - Slug: {agent.slug}")
    print(f"   - Original iteration: {agent.original_iteration}")
    print(f"   - Iteration (with strategy): {agent.iteration}")
    print(f"   - Strategy index: {agent.strategy_index}")
    print(f"   - Strategy: {agent.strategy['strategy'][:50]}...")
    print(f"   - Code filename: {code_file}")


def test_system_prompt_is_ensemble_specific(test_task_dir, monkeypatch):
    """Test that EnsemblerAgent uses ensemble-specific system prompt."""
    # Patch the module-level _TASK_ROOT constant (in developer, not ensembler)
    monkeypatch.setattr("agents.developer._TASK_ROOT", test_task_dir['task_root'])

    strategy = {
        "strategy": "Stacking ensemble with meta-learner",
        "models_needed": ["model1", "model2"]
    }

    baseline_metadata = {
        "baselines": [
            {"model": "model1", "score": 0.85},
            {"model": "model2", "score": 0.83}
        ]
    }

    agent = EnsemblerAgent(
        slug=test_task_dir['slug'],
        iteration=test_task_dir['iteration'],
        strategy_index=2,
        strategy=strategy,
        baseline_metadata=baseline_metadata,
    )

    # Get system prompt
    system_prompt = agent._compose_system()

    # Should be ensemble-specific (not the regular developer prompt)
    # The ensemble prompt should mention "ensemble" or "combine"
    assert "ensemble" in system_prompt.lower() or "combine" in system_prompt.lower()

    print("✅ EnsemblerAgent uses ensemble-specific system prompt:")
    print(f"   - Prompt length: {len(system_prompt)} chars")
    print(f"   - Contains 'ensemble': {'ensemble' in system_prompt.lower()}")


def test_baseline_metadata_format(test_task_dir, monkeypatch):
    """Test that baseline metadata is correctly formatted."""
    # Patch the module-level _TASK_ROOT constant (in developer, not ensembler)
    monkeypatch.setattr("agents.developer._TASK_ROOT", test_task_dir['task_root'])

    strategy = {
        "strategy": "Simple averaging",
        "models_needed": ["model1", "model2", "model3"]
    }

    baseline_metadata = {
        "baselines": [
            {"model": "model1", "score": 0.91, "path": "/path/to/model1"},
            {"model": "model2", "score": 0.89, "path": "/path/to/model2"},
            {"model": "model3", "score": 0.87, "path": "/path/to/model3"}
        ],
        "best_score": 0.91
    }

    agent = EnsemblerAgent(
        slug=test_task_dir['slug'],
        iteration=test_task_dir['iteration'],
        strategy_index=3,
        strategy=strategy,
        baseline_metadata=baseline_metadata,
    )

    # Check baseline metadata is stored
    assert agent.baseline_metadata == baseline_metadata
    assert len(agent.baseline_metadata["baselines"]) == 3
    assert agent.baseline_metadata["best_score"] == 0.91

    print("✅ Baseline metadata correctly stored:")
    print(f"   - Number of baselines: {len(agent.baseline_metadata['baselines'])}")
    print(f"   - Best score: {agent.baseline_metadata['best_score']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
