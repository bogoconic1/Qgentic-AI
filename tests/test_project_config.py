"""Tests for project-level config helpers."""

from __future__ import annotations

from pathlib import Path

import project_config


def test_get_instructions_prefers_task_specific_file(tmp_path, monkeypatch):
    root_instructions = tmp_path / "INSTRUCTIONS.md"
    root_instructions.write_text("# Models\nroot-model\n", encoding="utf-8")

    task_dir = tmp_path / "task" / "demo-slug"
    task_dir.mkdir(parents=True)
    (task_dir / "INSTRUCTIONS.md").write_text(
        "# Models\ntask-model\n", encoding="utf-8"
    )

    monkeypatch.setattr(project_config, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(
        project_config,
        "get_config",
        lambda: {"paths": {"task_root": "task"}},
    )
    project_config.get_instructions.cache_clear()

    instructions = project_config.get_instructions("demo-slug")

    assert instructions["# Models"] == ["task-model"]


def test_get_instructions_falls_back_to_repo_root(tmp_path, monkeypatch):
    root_instructions = tmp_path / "INSTRUCTIONS.md"
    root_instructions.write_text(
        "# Researcher Instructions\nroot researcher\n", encoding="utf-8"
    )

    monkeypatch.setattr(project_config, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(
        project_config,
        "get_config",
        lambda: {"paths": {"task_root": "task"}},
    )
    project_config.get_instructions.cache_clear()

    instructions = project_config.get_instructions("missing-slug")

    assert instructions["# Researcher Instructions"] == ["root researcher"]
