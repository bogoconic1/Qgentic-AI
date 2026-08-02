"""Deep Research sub-agent, exposed as ``ResearcherAgent``.

The Main Agent instantiates this with a run's ``(slug, run_id, research_iter)``
and calls ``run(instruction)``. The work is delegated wholesale to a
``codex exec`` subprocess: codex plans, searches the web, runs shell commands,
and writes files, with the research directory as its cwd and only writable root.
This module makes no LLM API calls of its own — that is the point, since codex
draws on the user's own subscription rather than a metered API key.

Per-invocation layout (owned by this module):

    task/<slug>/<run_id>/research_<research_iter>/
    ├── RESEARCH.md            # scaffolded here with a single H1, populated by
    │                          # codex, read back at exit as the return value
    ├── SOURCES.md             # codex-maintained record of what it read
    ├── researcher_chat.jsonl  # transcript, translated from codex's event stream
    ├── codex_events.jsonl     # raw `codex exec --json` output
    ├── .codex_prompt.md       # exact prompt sent, for reproducibility
    └── ...                    # whatever scratch files codex authored
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import weave
from dotenv import load_dotenv

from project_config import get_config
from prompts.research import build_codex_prompt


load_dotenv()

logger = logging.getLogger(__name__)

_TASK_ROOT = Path(get_config()["paths"]["task_root"])

# stderr tail returned to the parent agent when codex exits non-zero. Enough to
# carry a traceback or an auth error without flooding the caller's context.
_STDERR_TAIL_CHARS = 4_000


def translate_events(events_path: Path, chat_log_path: Path) -> None:
    """Render codex's JSONL event stream as agent chat records.

    Maps codex items onto the same two record shapes ``MainAgent`` writes, so a
    research transcript reads identically to a main-agent one. Runs once, after
    the process has exited — ``events_path`` is complete by then.

    Unrecognised item types are skipped rather than guessed at; the raw stream
    stays on disk beside this file.
    """
    records: list[dict] = []

    for line in events_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        if event.get("type") != "item.completed":
            continue

        item = event.get("item", {})
        item_type = item.get("type")

        if item_type == "agent_message":
            records.append({
                "role": "assistant",
                "content": item.get("text", ""),
                "function_calls": [],
            })
        elif item_type == "command_execution":
            records.append({
                "role": "tool",
                "name": "bash",
                "args": {"command": item.get("command", "")},
                "result": item.get("aggregated_output", ""),
            })
        elif item_type == "web_search":
            records.append({
                "role": "tool",
                "name": "web_research",
                "args": {"query": item.get("query", "")},
                "result": (item.get("action") or {}).get("type", ""),
            })
        elif item_type == "file_change":
            records.append({
                "role": "tool",
                "name": "edit_file",
                "args": {"changes": item.get("changes", [])},
                "result": item.get("status", ""),
            })

    with chat_log_path.open("a", encoding="utf-8") as f:
        for record in records:
            record["ts"] = datetime.now(timezone.utc).isoformat()
            f.write(json.dumps(record) + "\n")


@weave.op()
def record_usage(usage: dict) -> dict:
    """Surface codex's token usage to Weave.

    These tokens bill to the user's codex subscription rather than to an API
    key, but they are still the cost of a research call and belong in tracking.
    """
    return usage


def _read_usage(events_path: Path) -> dict | None:
    for line in reversed(events_path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "turn.completed":
            return event.get("usage")
    return None


class ResearcherAgent:
    """Deep Research sub-agent. See module docstring."""

    def __init__(
        self,
        slug: str,
        run_id: str,
        research_iter: int,
    ):
        self.slug = slug
        self.run_id = run_id
        self.research_iter = research_iter
        self.research_dir = _TASK_ROOT / slug / run_id / f"research_{research_iter}"
        self.research_md_path = self.research_dir / "RESEARCH.md"
        self.chat_log = self.research_dir / "researcher_chat.jsonl"
        self.events_path = self.research_dir / "codex_events.jsonl"
        self.prompt_path = self.research_dir / ".codex_prompt.md"

        codex_cfg = get_config()["subagents"]["codex"]
        self.model = codex_cfg["model"]
        self.sandbox = codex_cfg["sandbox"]

        if shutil.which("codex") is None:
            raise RuntimeError(
                "The `codex` CLI is not on PATH. Deep Research runs on codex "
                "so it draws on your subscription instead of API credit — "
                "install it and sign in before launching the agent."
            )

    def _load_custom_instructions(self) -> str | None:
        """Read `task/<slug>/RESEARCHER_INSTRUCTIONS.md` if it exists."""
        path = _TASK_ROOT / self.slug / "RESEARCHER_INSTRUCTIONS.md"
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    @weave.op()
    def run(self, instruction: str) -> str:
        """Run one research iteration under codex and return the report.

        Scaffolds ``RESEARCH.md`` with a single H1 so codex has a known target,
        runs ``codex exec`` to completion with no timeout, then reads
        ``RESEARCH.md`` back — that file IS the report.

        Args:
            instruction: Free-form research instruction — as long as needed.

        Returns:
            Contents of ``RESEARCH.md`` after the run, or a JSON error string
            if codex exited non-zero.
        """
        self.research_dir.mkdir(parents=True, exist_ok=True)
        if not self.research_md_path.exists():
            self.research_md_path.write_text(f"# {instruction}\n", encoding="utf-8")

        self.prompt_path.write_text(
            build_codex_prompt(instruction, self._load_custom_instructions()),
            encoding="utf-8",
        )

        command = [
            "codex", "exec", "--json",
            "--cd", str(self.research_dir),
            "--sandbox", self.sandbox,
            "--skip-git-repo-check",
            "-m", self.model,
            "-",
        ]
        logger.info(
            "ResearcherAgent slug=%s run_id=%s iter=%d dir=%s instruction=%r",
            self.slug,
            self.run_id,
            self.research_iter,
            self.research_dir,
            instruction,
        )

        # No timeout: research legitimately runs for tens of minutes, and a cap
        # would kill real work. A wedged codex therefore blocks this call — the
        # elapsed-time log line below is what distinguishes a hang from depth.
        started = datetime.now(timezone.utc)
        with (
            self.prompt_path.open("r", encoding="utf-8") as stdin,
            self.events_path.open("w", encoding="utf-8") as stdout,
        ):
            proc = subprocess.run(
                command,
                stdin=stdin,
                stdout=stdout,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        logger.info(
            "codex exec exited rc=%d after %.0fs (%s)",
            proc.returncode,
            elapsed,
            self.events_path,
        )

        if proc.returncode != 0:
            logger.error("codex exec failed: %s", proc.stderr[-_STDERR_TAIL_CHARS:])
            return json.dumps({
                "error": f"codex exec exited {proc.returncode}",
                "stderr": proc.stderr[-_STDERR_TAIL_CHARS:],
            })

        translate_events(self.events_path, self.chat_log)

        usage = _read_usage(self.events_path)
        if usage is not None:
            record_usage(usage)

        report = self.research_md_path.read_text(encoding="utf-8")
        logger.info(
            "ResearcherAgent read %s (%d chars)", self.research_md_path, len(report)
        )
        return report
