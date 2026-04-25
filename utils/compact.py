"""Conversation compaction for unbounded agent loops.

Triggered after every ``call_llm`` whose returned ``prompt_token_count``
exceeds ``runtime.compaction_threshold_tokens``. Replaces the front of the
``input_list`` with a single user-role summary message produced by a
one-shot Gemini call against the same model the agent is using.

Public API:
    should_compact(last_input_tokens) -> bool
    compact_messages(input_list, *, model) -> list[dict]
"""

from __future__ import annotations

import json
import logging
import re

from project_config import get_config_value
from tools.helpers import call_llm
from utils.llm_utils import append_message


logger = logging.getLogger(__name__)


_COMPACT_PROMPT = """Your task is to create a detailed summary of the conversation so far, paying close attention to the user's explicit requests and your previous actions.
This summary should be thorough in capturing technical details, code patterns, and architectural decisions that would be essential for continuing development work without losing context.

Before providing your final summary, wrap your analysis in <analysis> tags to organize your thoughts and ensure you've covered all necessary points. In your analysis process:

1. Chronologically analyze each message and section of the conversation. For each section thoroughly identify:
   - The user's explicit requests and intents
   - Your approach to addressing the user's requests
   - Key decisions, technical concepts and code patterns
   - Specific details like:
     - file names
     - full code snippets
     - function signatures
     - file edits
   - Errors that you ran into and how you fixed them
   - Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.
2. Double-check for technical accuracy and completeness, addressing each required element thoroughly.

Your summary should include the following sections:

1. Primary Request and Intent: Capture all of the user's explicit requests and intents in detail
2. Key Technical Concepts: List all important technical concepts, technologies, and frameworks discussed.
3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or created. Pay special attention to the most recent messages and include full code snippets where applicable and include a summary of why this file read or edit is important.
4. Errors and fixes: List all errors that you ran into, and how you fixed them. Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.
5. Problem Solving: Document problems solved and any ongoing troubleshooting efforts.
6. All user messages: List ALL user messages that are not tool results. These are critical for understanding the users' feedback and changing intent.
7. Pending Tasks: Outline any pending tasks that you have explicitly been asked to work on.
8. Current Work: Describe in detail precisely what was being worked on immediately before this summary request, paying special attention to the most recent messages from both user and assistant. Include file names and code snippets where applicable.
9. Optional Next Step: List the next step that you will take that is related to the most recent work you were doing. IMPORTANT: ensure that this step is DIRECTLY in line with the user's most recent explicit requests, and the task you were working on immediately before this summary request. If your last task was concluded, then only list next steps if they are explicitly in line with the users request. Do not start on tangential requests or really old requests that were already completed without confirming with the user first.
                       If there is a next step, include direct quotes from the most recent conversation showing exactly what task you were working on and where you left off. This should be verbatim to ensure there's no drift in task interpretation.

Format your output as:

<analysis>
[Your thought process, ensuring all points are covered thoroughly and accurately]
</analysis>

<summary>
1. Primary Request and Intent: ...
2. Key Technical Concepts: ...
3. Files and Code Sections: ...
4. Errors and fixes: ...
5. Problem Solving: ...
6. All user messages: ...
7. Pending Tasks: ...
8. Current Work: ...
9. Optional Next Step: ...
</summary>

The conversation to summarise is provided below inside <conversation> tags as a JSON array of message objects (each with a role and parts). Produce only the <analysis> and <summary> blocks — no other text.

<conversation>
{conversation_json}
</conversation>
"""


_SUMMARY_PREAMBLE = (
    "This session is being continued from a previous conversation that ran out "
    "of context. The summary below covers the earlier portion of the "
    "conversation. Continue from where you left off without asking any further "
    "questions; do not acknowledge the summary or recap what was happening — "
    "pick up the last task as if the break never happened.\n\n"
)


def should_compact(last_input_tokens: int | None) -> bool:
    """Return True iff the most recent call's prompt_token_count exceeds the configured threshold."""
    if last_input_tokens is None:
        return False
    threshold = get_config_value("runtime", "compaction_threshold_tokens")
    if threshold is None:
        raise RuntimeError(
            "config.yaml is missing required key runtime.compaction_threshold_tokens"
        )
    over = last_input_tokens > int(threshold)
    logger.info(
        "[compact] input_tokens=%d threshold=%d → %s",
        last_input_tokens,
        int(threshold),
        "COMPACTING" if over else "ok",
    )
    return over


def compact_messages(input_list: list[dict], *, model: str) -> list[dict]:
    """Summarise the front of ``input_list`` and return ``[summary_user, *kept]``.

    ``kept`` is the last ``runtime.compaction_keep_last`` entries. Gemini's
    "conversation must start with user" rule is satisfied by the prepended
    ``summary_user`` turn, so ``kept`` may start on any role — including an
    orphaned function-response whose function_call was summarised away.
    """
    keep_last = get_config_value("runtime", "compaction_keep_last")
    if keep_last is None:
        raise RuntimeError(
            "config.yaml is missing required key runtime.compaction_keep_last"
        )
    keep_last = int(keep_last)

    if len(input_list) <= keep_last:
        return list(input_list)

    cut = len(input_list) - keep_last
    old, recent = input_list[:cut], input_list[cut:]

    logger.info(
        "[compact] summarising %d old messages, keeping %d verbatim",
        len(old),
        len(recent),
    )

    response = call_llm(
        model=model,
        system_instruction="You are summarising a tool-driven agent conversation.",
        messages=_COMPACT_PROMPT.format(
            conversation_json=json.dumps(old, ensure_ascii=False)
        ),
        function_declarations=None,
        enable_google_search=False,
    )
    summary_text = _format_summary((response.text or "").strip())
    summary_msg = append_message("user", _SUMMARY_PREAMBLE + summary_text)
    return [summary_msg, *recent]


_ANALYSIS_RE = re.compile(r"<analysis>[\s\S]*?</analysis>")
_SUMMARY_RE = re.compile(r"<summary>([\s\S]*?)</summary>")


def _format_summary(raw: str) -> str:
    """Strip <analysis>; unwrap <summary>…</summary> into a `Summary:` header."""
    formatted = _ANALYSIS_RE.sub("", raw)
    match = _SUMMARY_RE.search(formatted)
    if match:
        # `.replace` with a plain string avoids re.sub's replacement-template
        # parsing — the summary body can legitimately contain literal \d, \w,
        # etc. (regex explanations, Windows paths, LaTeX macros) which would
        # otherwise crash `re.sub` with `re.error: bad escape`.
        formatted = formatted.replace(
            match.group(0), f"Summary:\n{match.group(1).strip()}", 1
        )
    return re.sub(r"\n\n+", "\n\n", formatted).strip() or raw
