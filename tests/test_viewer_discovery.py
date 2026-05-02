"""Tests for scripts.viewer.discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.viewer import discovery


def _make_run(root: Path, slug: str, run_id: str, *,
              with_main_log: bool = True,
              research_indices: tuple[int, ...] = ()) -> Path:
    run = root / slug / run_id
    run.mkdir(parents=True, exist_ok=True)
    if with_main_log:
        (run / "main_agent_chat.jsonl").write_text("")
    for n in research_indices:
        (run / f"research_{n}").mkdir(exist_ok=True)
    return run


def test_run_id_regex_filters_out_non_runs():
    assert discovery.RUN_ID_RE.match("20260502_082208")
    assert not discovery.RUN_ID_RE.match("claude-onnx-9")
    assert not discovery.RUN_ID_RE.match("task001.json")
    assert not discovery.RUN_ID_RE.match(".ipynb_checkpoints")
    assert not discovery.RUN_ID_RE.match("20260502_82208")  # 5 not 6 digits
    assert not discovery.RUN_ID_RE.match("2026050_082208")  # 7 not 8 digits


def test_list_slugs_handles_missing_root(tmp_path: Path):
    assert discovery.list_slugs(tmp_path / "does_not_exist") == []


def test_list_slugs_reports_latest_and_count(tmp_path: Path):
    _make_run(tmp_path, "alpha", "20260101_000000")
    _make_run(tmp_path, "alpha", "20260201_000000")
    _make_run(tmp_path, "beta", "20260301_000000")
    # noise that should be ignored at the slug level
    (tmp_path / "alpha" / "GOAL.md").write_text("hi")
    (tmp_path / "alpha" / "claude-onnx-9").mkdir()
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "alpha" / "task001.json").write_text("{}")

    slugs = {s.name: s for s in discovery.list_slugs(tmp_path)}
    assert set(slugs) == {"alpha", "beta"}
    assert slugs["alpha"].latest_run == "20260201_000000"
    assert slugs["alpha"].run_count == 2
    assert slugs["beta"].latest_run == "20260301_000000"
    assert slugs["beta"].run_count == 1


def test_list_runs_newest_first_with_subagents(tmp_path: Path):
    _make_run(tmp_path, "alpha", "20260101_000000", research_indices=(1,))
    _make_run(tmp_path, "alpha", "20260201_000000")
    # non-run dirs that must be filtered out
    (tmp_path / "alpha" / "claude-onnx-9").mkdir()
    (tmp_path / "alpha" / "data").mkdir()

    runs = discovery.list_runs(tmp_path, "alpha")
    assert [r.run_id for r in runs] == ["20260201_000000", "20260101_000000"]
    older = runs[1]
    assert older.research_indices == (1,)
    assert older.has_main_log is True


def test_list_runs_missing_slug(tmp_path: Path):
    assert discovery.list_runs(tmp_path, "ghost") == []


def test_get_run_raises_on_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        discovery.get_run(tmp_path, "ghost", "20260101_000000")


def test_get_run_rejects_bad_run_id(tmp_path: Path):
    _make_run(tmp_path, "alpha", "20260101_000000")
    with pytest.raises(FileNotFoundError):
        discovery.get_run(tmp_path, "alpha", "../escape")


def test_get_run_no_main_log(tmp_path: Path):
    _make_run(tmp_path, "alpha", "20260101_000000", with_main_log=False)
    info = discovery.get_run(tmp_path, "alpha", "20260101_000000")
    assert info.has_main_log is False


def test_is_safe_path_accepts_contained(tmp_path: Path):
    target = tmp_path / "alpha" / "20260101_000000" / "MAIN.md"
    target.parent.mkdir(parents=True)
    target.write_text("ok")
    resolved = discovery.is_safe_path(tmp_path, "alpha/20260101_000000/MAIN.md")
    assert resolved is not None
    assert resolved.read_text() == "ok"


def test_is_safe_path_rejects_traversal(tmp_path: Path):
    assert discovery.is_safe_path(tmp_path, "../etc/passwd") is None
    assert discovery.is_safe_path(tmp_path, "alpha/../../etc/passwd") is None
    assert discovery.is_safe_path(tmp_path, "/etc/passwd") is None
    assert discovery.is_safe_path(tmp_path, "") is None
