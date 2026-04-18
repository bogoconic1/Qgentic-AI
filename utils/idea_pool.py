"""Idea-pool storage for the Main Agent.

Flat-file layout under ``task/<slug>/<run_id>/ideas/``:

    ideas/
    ├── INDEX.md                       # regenerated on every mutation
    ├── 001_try_random_forest.md
    ├── 002_add_external_dataset.md
    └── ...

Each idea file is pure markdown with the title as the first ``# H1``. The
integer id lives as a zero-padded prefix in the filename so lookups and
orderings are cheap. INDEX.md is a one-line-per-idea summary that MainAgent
always has in its system prompt; ``load_index`` applies line + byte caps
(mirroring ``claude-code/memdir/memdir.ts:truncateEntrypointContent``) with a
trailing warning when the pool outgrows them.
"""

from __future__ import annotations

import re
from pathlib import Path

_INDEX_FILENAME = "INDEX.md"
_INDEX_HEADER = "# Idea pool"
_H1 = re.compile(r"^#\s+(.+)$", re.MULTILINE)

MAX_INDEX_LINES = 200
MAX_INDEX_BYTES = 25_000


def _truncate_with_warning(content: str) -> str:
    trimmed = content.rstrip("\n")
    lines = trimmed.split("\n")
    if len(lines) <= MAX_INDEX_LINES and len(trimmed) <= MAX_INDEX_BYTES:
        return trimmed + "\n"
    truncated = "\n".join(lines[:MAX_INDEX_LINES])
    if len(truncated) > MAX_INDEX_BYTES:
        cut = truncated.rfind("\n", 0, MAX_INDEX_BYTES)
        truncated = truncated[:cut]
    warning = (
        f"\n\n> WARNING: {_INDEX_FILENAME} truncated at {MAX_INDEX_LINES} "
        f"lines / {MAX_INDEX_BYTES} bytes. Prune the pool."
    )
    return truncated + warning + "\n"


def render_index(ideas_dir: Path) -> str:
    entries = sorted(
        (int(p.stem.split("_", 1)[0]), p)
        for p in ideas_dir.glob("[0-9][0-9][0-9]_*.md")
    )
    lines = [_INDEX_HEADER, ""]
    for idea_id, path in entries:
        title = _H1.search(path.read_text()).group(1).strip()
        lines.append(f"- [{idea_id:03d}] {title}")
    content = "\n".join(lines) + "\n"
    (ideas_dir / _INDEX_FILENAME).write_text(content)
    return content


def load_index(ideas_dir: Path) -> str:
    return _truncate_with_warning((ideas_dir / _INDEX_FILENAME).read_text())


def add_idea(ideas_dir: Path, title: str, description: str) -> int:
    existing = [
        int(p.stem.split("_", 1)[0])
        for p in ideas_dir.glob("[0-9][0-9][0-9]_*.md")
    ]
    idea_id = (max(existing) + 1) if existing else 1
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    (ideas_dir / f"{idea_id:03d}_{slug}.md").write_text(
        f"# {title}\n\n{description}\n"
    )
    render_index(ideas_dir)
    return idea_id


def remove_idea(ideas_dir: Path, idea_id: int) -> None:
    (path,) = ideas_dir.glob(f"{idea_id:03d}_*.md")
    path.unlink()
    render_index(ideas_dir)


def update_idea(ideas_dir: Path, idea_id: int, description: str) -> None:
    (path,) = ideas_dir.glob(f"{idea_id:03d}_*.md")
    title = _H1.search(path.read_text()).group(1).strip()
    path.write_text(f"# {title}\n\n{description}\n")
    render_index(ideas_dir)
