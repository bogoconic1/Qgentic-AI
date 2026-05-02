"""Filesystem walk + filtering for agent run transcripts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import project_config

RUN_ID_RE = re.compile(r"^[0-9]{8}_[0-9]{6}$")
_RESEARCH_RE = re.compile(r"^research_([0-9]+)$")
_MAIN_LOG_NAME = "main_agent_chat.jsonl"


@dataclass(frozen=True)
class SlugInfo:
    name: str
    latest_run: str | None
    run_count: int


@dataclass(frozen=True)
class RunInfo:
    slug: str
    run_id: str
    has_main_log: bool
    research_indices: tuple[int, ...]
    mtime: float


def task_root() -> Path:
    """Resolve the project's task root directory.

    Mirrors how the agents resolve `paths.task_root` (project_config.py:20),
    but anchors a relative path against the directory holding `project_config.py`
    so the viewer works regardless of the current working directory.
    """
    raw = project_config.get_config_value("paths", "task_root", default="task")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = Path(project_config.__file__).resolve().parent / candidate
    return candidate


def list_slugs(root: Path) -> list[SlugInfo]:
    if not root.is_dir():
        return []
    out: list[SlugInfo] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        run_ids = sorted(
            (p.name for p in child.iterdir() if p.is_dir() and RUN_ID_RE.match(p.name)),
            reverse=True,
        )
        out.append(
            SlugInfo(
                name=child.name,
                latest_run=run_ids[0] if run_ids else None,
                run_count=len(run_ids),
            )
        )
    return out


def list_runs(root: Path, slug: str) -> list[RunInfo]:
    slug_dir = root / slug
    if not slug_dir.is_dir():
        return []
    runs: list[RunInfo] = []
    for child in slug_dir.iterdir():
        if not child.is_dir() or not RUN_ID_RE.match(child.name):
            continue
        runs.append(_run_info(slug, child))
    runs.sort(key=lambda r: r.run_id, reverse=True)
    return runs


def get_run(root: Path, slug: str, run_id: str) -> RunInfo:
    if not RUN_ID_RE.match(run_id):
        raise FileNotFoundError(f"invalid run_id: {run_id!r}")
    run_dir = root / slug / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(str(run_dir))
    return _run_info(slug, run_dir)


def _run_info(slug: str, run_dir: Path) -> RunInfo:
    researches: list[int] = []
    for entry in run_dir.iterdir():
        if not entry.is_dir():
            continue
        m = _RESEARCH_RE.match(entry.name)
        if m:
            researches.append(int(m.group(1)))
    researches.sort()
    return RunInfo(
        slug=slug,
        run_id=run_dir.name,
        has_main_log=(run_dir / _MAIN_LOG_NAME).is_file(),
        research_indices=tuple(researches),
        mtime=run_dir.stat().st_mtime,
    )


def is_safe_path(root: Path, rel: str) -> Path | None:
    """Resolve `rel` under `root`, returning the resolved Path iff contained.

    Returns None for any traversal attempt (`../`), absolute paths, or values
    that resolve outside `root`.
    """
    if not rel:
        return None
    if Path(rel).is_absolute():
        return None
    root_resolved = root.resolve()
    candidate = (root_resolved / rel).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError:
        return None
    return candidate
