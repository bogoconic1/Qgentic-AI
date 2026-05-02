"""LLM-as-judge for bash-command safety.

Before each shell command is executed by the developer or researcher
subagent, this module asks a small LLM whether the command is safe to
run inside the sandbox. The verdict is cached per command string so
repeat calls are free.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from project_config import get_config
from prompts.bash_judge import bash_safety_system
from schemas.bash_safety import BashSafetyVerdict
from tools.helpers import call_llm
from utils.llm_utils import append_message


logger = logging.getLogger(__name__)


_BASH_MAX_LEN = 4000  # bytes — caps prompt size before we even consult the judge.


def _judge_model() -> str:
    return get_config()["llm"]["bash_judge_model"]


@lru_cache(maxsize=4096)
def judge_bash_command(command: str, writable_root: str) -> BashSafetyVerdict:
    """Ask the LLM judge whether `command` is safe; return the verdict.

    The judge is given the agent's ``writable_root`` so it can enforce
    per-agent scope: bash runs with ``cwd=writable_root``, ``cd`` /
    ``pushd`` / ``chdir`` are forbidden, and writes whose targets resolve
    outside ``writable_root`` are blocked, in addition to the existing
    destructive-op rules.

    Cached by ``(command, writable_root)`` — identical commands don't
    re-LLM. Two cheap sanity checks short-circuit the LLM call: empty
    commands and over-long commands.
    """
    if not command.strip():
        return BashSafetyVerdict(verdict="block", reason="Empty command.")
    if len(command) > _BASH_MAX_LEN:
        return BashSafetyVerdict(
            verdict="block",
            reason=f"Command exceeds {_BASH_MAX_LEN}-byte cap.",
        )

    logger.info(
        "bash_judge model=%s writable_root=%s command=%r",
        _judge_model(),
        writable_root,
        command[:200],
    )
    return call_llm(
        model=_judge_model(),
        system_instruction=bash_safety_system(writable_root),
        function_declarations=None,
        messages=[append_message("user", f"Command:\n```\n{command}\n```")],
        text_format=BashSafetyVerdict,
    )
