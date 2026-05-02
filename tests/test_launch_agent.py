"""Tests for the source-of-truth metadata sync at launch."""

from __future__ import annotations

from pathlib import Path

import pytest

import launch_agent


@pytest.fixture
def repo_with_metadata(monkeypatch, tmp_path):
    """Stand up a fake repo root with all three required metadata files."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "GOAL.md").write_text("# goal\n", encoding="utf-8")
    (repo_root / "DEVELOPER_INSTRUCTIONS.md").write_text(
        "# dev\nuse mixed precision\n", encoding="utf-8"
    )
    (repo_root / "RESEARCHER_INSTRUCTIONS.md").write_text(
        "# research\ncite primary sources\n", encoding="utf-8"
    )
    monkeypatch.setattr(launch_agent, "_REPO_ROOT", repo_root)
    return repo_root


def test_sync_copies_all_three_files(repo_with_metadata, tmp_path):
    """Happy path: all three files at root → all three land in base_dir."""
    base_dir = tmp_path / "task" / "demo-slug"
    base_dir.mkdir(parents=True)

    launch_agent._sync_task_metadata(base_dir)

    assert (base_dir / "GOAL.md").read_text(encoding="utf-8") == "# goal\n"
    assert "mixed precision" in (
        base_dir / "DEVELOPER_INSTRUCTIONS.md"
    ).read_text(encoding="utf-8")
    assert "primary sources" in (
        base_dir / "RESEARCHER_INSTRUCTIONS.md"
    ).read_text(encoding="utf-8")


def test_sync_overwrites_stale_copies_in_task_dir(repo_with_metadata, tmp_path):
    """Re-running the launcher always overwrites — root is source of truth."""
    base_dir = tmp_path / "task" / "demo-slug"
    base_dir.mkdir(parents=True)
    (base_dir / "GOAL.md").write_text("# stale\n", encoding="utf-8")
    (base_dir / "DEVELOPER_INSTRUCTIONS.md").write_text("# stale\n", encoding="utf-8")

    launch_agent._sync_task_metadata(base_dir)

    assert (base_dir / "GOAL.md").read_text(encoding="utf-8") == "# goal\n"
    assert "mixed precision" in (
        base_dir / "DEVELOPER_INSTRUCTIONS.md"
    ).read_text(encoding="utf-8")


def test_sync_throws_when_any_root_file_missing(monkeypatch, tmp_path):
    """Missing root file → FileNotFoundError naming all missing files; no
    partial writes."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "GOAL.md").write_text("# goal\n", encoding="utf-8")
    # DEVELOPER_INSTRUCTIONS.md and RESEARCHER_INSTRUCTIONS.md intentionally absent.
    monkeypatch.setattr(launch_agent, "_REPO_ROOT", repo_root)

    base_dir = tmp_path / "task" / "demo-slug"
    base_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError) as excinfo:
        launch_agent._sync_task_metadata(base_dir)

    msg = str(excinfo.value)
    assert "DEVELOPER_INSTRUCTIONS.md" in msg
    assert "RESEARCHER_INSTRUCTIONS.md" in msg
    # Pre-flight check fails before any copy happens — no GOAL.md ends up
    # in the task dir even though it was present at the root.
    assert not (base_dir / "GOAL.md").exists()
