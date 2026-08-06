"""Codex-as-judge for bash-command safety.

Before each shell command MainAgent runs through the `bash` filesystem tool,
a `codex exec` call decides allow-or-block, so the check draws on the user's
Codex subscription rather than a metered API key. Verdicts are cached per
``(command, writable_root)`` so repeat commands are free.

The judge itself runs under codex's ``read-only`` sandbox and is instructed to
reason about the command text without executing it. Failures — codex missing,
rate-limited, unparseable output — **block** the command rather than allowing
it or crashing MainAgent's turn, and are never cached, so one transient outage
cannot permanently poison a command for the rest of the session.
"""

from __future__ import annotations

import logging
from pathlib import Path

from project_config import get_config
from prompts.bash_judge import bash_safety_system
from schemas.bash_safety import BashSafetyVerdict
from utils.codex import run_codex


logger = logging.getLogger(__name__)


_BASH_MAX_LEN = 4000  # bytes — caps prompt size before we even consult the judge.

_CODEX_MODEL = get_config()["subagents"]["codex"]["model"]

# Successful verdicts only. Failure verdicts (judge unavailable) are returned
# but never stored — caching one would permanently block that command for the
# session. Parallel dispatch may race two identical judgements; dict get/set is
# GIL-atomic, so the worst case is a duplicate codex call, which is harmless.
_VERDICTS: dict[tuple[str, str], BashSafetyVerdict] = {}


def judge_bash_command(command: str, writable_root: str) -> BashSafetyVerdict:
    """Ask the codex judge whether `command` is safe; return the verdict.

    The judge is given the agent's ``writable_root`` so it can enforce
    per-agent scope: bash runs with ``cwd=writable_root``, ``cd`` /
    ``pushd`` / ``chdir`` are forbidden, and writes whose targets resolve
    outside ``writable_root`` are blocked, in addition to the existing
    destructive-op rules.

    Two cheap sanity checks short-circuit the codex call entirely: empty
    commands and over-long commands.
    """
    if not command.strip():
        return BashSafetyVerdict(verdict="block", reason="Empty command.")
    if len(command) > _BASH_MAX_LEN:
        return BashSafetyVerdict(
            verdict="block",
            reason=f"Command exceeds {_BASH_MAX_LEN}-byte cap.",
        )

    cache_key = (command, writable_root)
    cached = _VERDICTS.get(cache_key)
    if cached is not None:
        return cached

    logger.info(
        "bash_judge model=%s writable_root=%s command=%r",
        _CODEX_MODEL,
        writable_root,
        command[:200],
    )
    result = run_codex(
        bash_safety_system(writable_root) + f"\n\nCommand:\n```\n{command}\n```",
        cwd=Path(writable_root),
        sandbox="read-only",
        model=_CODEX_MODEL,
        output_schema=BashSafetyVerdict,
    )

    if not result.ok or result.parsed is None:
        # Fail closed, uncached: block now, but let a later retry get a real
        # verdict once codex is reachable again.
        logger.error(
            "bash judge unavailable (rc=%d): %s", result.returncode, result.stderr
        )
        return BashSafetyVerdict(
            verdict="block",
            reason=f"Safety judge unavailable (rc={result.returncode}) — "
            "blocking by default; retry may succeed.",
        )

    _VERDICTS[cache_key] = result.parsed
    return result.parsed
