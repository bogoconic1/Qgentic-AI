"""Developer subagent — codegen disabled in this PR.

PR 1 of the issue-#221 stack strips the legacy fenced-``python`` codegen
contract (the model emitting ``train.py`` as a markdown code block that
``extract_python_code`` writes to disk wholesale). PR 2 will reintroduce a
structured contract with ``write_file`` / ``edit_file`` tools modeled on
``claude-code/tools/FileWriteTool`` and ``claude-code/tools/FileEditTool``.

Until PR 2 lands, ``DeveloperAgent.run()`` raises ``NotImplementedError`` so
a ``develop`` call from MainAgent surfaces a clean error instead of silently
producing nothing. ``_find_previous_code`` and ``_load_custom_instructions``
are kept because PR 2's reborn codegen loop will need them.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import weave

from project_config import get_config


logger = logging.getLogger(__name__)


_CONFIG = get_config()
_PATH_CFG = _CONFIG["paths"]
_TASK_ROOT = Path(_PATH_CFG["task_root"])


class DeveloperAgent:
    """One-training-iteration subagent (codegen disabled until PR 2)."""

    def __init__(
        self,
        slug: str,
        run_id: str,
        dev_iter: int,
        conda_env: str | None = None,
    ):
        self.slug = slug
        self.run_id = run_id
        self.dev_iter = dev_iter
        self.conda_env = conda_env
        self.base_dir = _TASK_ROOT / slug / run_id / f"developer_{dev_iter}"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _find_previous_code(self) -> str | None:
        """Return the text of the last successful ``train.py`` from prior dev_iter, if any."""
        if self.dev_iter <= 1:
            return None
        for prev_iter in range(self.dev_iter - 1, 0, -1):
            prev_base = _TASK_ROOT / self.slug / self.run_id / f"developer_{prev_iter}"
            if not prev_base.exists():
                continue
            successful = sorted(
                (
                    d
                    for d in prev_base.iterdir()
                    if d.is_dir()
                    and d.name.isdigit()
                    and (d / "train_stats.json").exists()
                ),
                key=lambda p: int(p.name),
            )
            if successful:
                train_py = successful[-1] / "train.py"
                if train_py.exists():
                    return train_py.read_text()
        return None

    def _load_custom_instructions(self) -> str | None:
        """Read `task/<slug>/DEVELOPER_INSTRUCTIONS.md` if it exists."""
        path = _TASK_ROOT / self.slug / "DEVELOPER_INSTRUCTIONS.md"
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    @weave.op()
    def run(self, idea: str) -> dict[str, Any]:
        raise NotImplementedError(
            "DeveloperAgent codegen is disabled in this PR. PR 2 will introduce "
            "write_file / edit_file tools (modeled on claude-code's FileWriteTool "
            "and FileEditTool) and re-enable a structured codegen loop."
        )
