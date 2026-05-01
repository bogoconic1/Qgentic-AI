"""Unit tests for the DeveloperAgent stub."""

from __future__ import annotations

import pytest

from agents import developer
from agents.developer import DeveloperAgent


def test_run_raises_not_implemented(monkeypatch, tmp_path):
    monkeypatch.setattr(developer, "_TASK_ROOT", tmp_path / "task")

    dev = DeveloperAgent(slug="test-slug", run_id="r1", dev_iter=1)
    with pytest.raises(NotImplementedError, match="codegen is disabled"):
        dev.run(idea="Produce a finite score.")
