"""DeveloperAgent stub.

#270 stripped the legacy fenced-``python`` codegen contract. #271 gutted
the retry loop and orphaned helpers. This rename PR shifts the canonical
artifacts from ``train.py`` / ``train_stats.json`` / ``developer_<iter>/``
to ``SOLUTION.py`` / ``SOLUTION.json`` / ``developer_v{N}/``. A follow-up
PR will replace this stub with a fresh flat tool loop modeled on
``ResearcherAgent.run()`` plus the ``run_solution`` / filesystem /
``web_search_stack_trace`` palette.

Until then, ``run()`` raises ``NotImplementedError`` so a ``develop`` call
from MainAgent surfaces a clean error.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import weave

from project_config import get_config


logger = logging.getLogger(__name__)


_CONFIG = get_config()
_TASK_ROOT = Path(_CONFIG["paths"]["task_root"])


class DeveloperAgent:
    """Stub. Reborn in a follow-up PR."""

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
        self.base_dir = _TASK_ROOT / slug / run_id / f"developer_v{dev_iter}"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @weave.op()
    def run(self, idea: str) -> dict[str, Any]:
        raise NotImplementedError(
            "DeveloperAgent codegen is disabled. A follow-up PR will introduce "
            "the new flat-loop codegen design."
        )
