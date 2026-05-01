"""Unit tests for DeveloperAgent (codegen disabled in this PR).

PR 1 disables the fenced-``python`` codegen contract. Until PR 2 reintroduces
a structured ``write_file``/``edit_file`` codegen loop, ``run()`` is expected
to raise ``NotImplementedError``. The helpers ``_find_previous_code`` and
``_load_custom_instructions`` continue to be exercised here because PR 2
will reuse them.
"""

from __future__ import annotations

import json

import pytest

from agents import developer
from agents.developer import DeveloperAgent


def test_run_raises_not_implemented(monkeypatch, tmp_path):
    monkeypatch.setattr(developer, "_TASK_ROOT", tmp_path / "task")

    dev = DeveloperAgent(slug="test-slug", run_id="r1", dev_iter=1)
    with pytest.raises(NotImplementedError, match="codegen is disabled"):
        dev.run(idea="Produce a finite score.")


def test_find_previous_code_returns_none_for_first_iter(monkeypatch, tmp_path):
    monkeypatch.setattr(developer, "_TASK_ROOT", tmp_path / "task")

    dev = DeveloperAgent(slug="test-slug", run_id="r1", dev_iter=1)
    assert dev._find_previous_code() is None


def test_find_previous_code_walks_back_to_last_successful(monkeypatch, tmp_path):
    monkeypatch.setattr(developer, "_TASK_ROOT", tmp_path / "task")

    prev_attempt = (
        tmp_path / "task" / "test-slug" / "r1" / "developer_1" / "1"
    )
    prev_attempt.mkdir(parents=True)
    (prev_attempt / "train.py").write_text("# marker-prior-code\nprint('prev')\n")
    (prev_attempt / "train_stats.json").write_text(json.dumps({"score": 0.3}))

    dev = DeveloperAgent(slug="test-slug", run_id="r1", dev_iter=2)
    code = dev._find_previous_code()
    assert code is not None
    assert "marker-prior-code" in code


def test_load_custom_instructions_inlines_when_present(monkeypatch, tmp_path):
    monkeypatch.setattr(developer, "_TASK_ROOT", tmp_path / "task")

    task_dir = tmp_path / "task" / "test-slug"
    task_dir.mkdir(parents=True, exist_ok=True)
    body = "Always emit the score as a percent multiplied by 100."
    (task_dir / "DEVELOPER_INSTRUCTIONS.md").write_text(body)

    dev = DeveloperAgent(slug="test-slug", run_id="r1", dev_iter=1)
    assert dev._load_custom_instructions() == body


def test_load_custom_instructions_returns_none_when_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(developer, "_TASK_ROOT", tmp_path / "task")
    dev = DeveloperAgent(slug="test-slug", run_id="r1", dev_iter=1)
    assert dev._load_custom_instructions() is None
