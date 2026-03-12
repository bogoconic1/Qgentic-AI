"""Tests for the legal retrieval task scaffold."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

from setup_legal_retrieval_task import scaffold_task


def _load_metric_module(metric_path: Path):
    spec = importlib.util.spec_from_file_location("legal_metric", metric_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_scaffold_task_writes_helper_files(tmp_path):
    task_dir = tmp_path / "llm-agentic-legal-information-retrieval"
    task_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "query_id": ["val_0", "val_1"],
            "query": ["q1", "q2"],
            "gold_citations": ["Art. 1 OR", "BGE 145 II 32 E. 3.1"],
        }
    ).to_csv(task_dir / "val.csv", index=False)

    scaffold_task(task_dir, overwrite=True)

    assert (task_dir / "description.md").exists()
    assert (task_dir / "metric.py").exists()
    assert (task_dir / "INSTRUCTIONS.md").exists()
    assert (task_dir / "cv_splits.json").exists()

    cv_splits = json.loads((task_dir / "cv_splits.json").read_text(encoding="utf-8"))
    assert cv_splits == {"fold_0": {"train": [], "val": [0, 1]}}

    metric_module = _load_metric_module(task_dir / "metric.py")
    score = metric_module.score(
        ["Art. 1 OR", "BGE 145 II 32 E. 3.1"],
        ["Art. 1 OR", "BGE 999"],
    )
    assert 0.0 <= score <= 1.0

    instructions = (task_dir / "INSTRUCTIONS.md").read_text(encoding="utf-8")
    assert "mistral-7b-instruct-v0.2" in instructions
