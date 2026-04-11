"""Standalone goal-mode developer agent.

Reads a free-form goal description, iterates a generate-execute-review loop
until the reviewer reports `done` or the limits are exhausted. Independent
of the existing developer flow at the orchestration level — but reuses the
shared low-level helpers (`call_llm`, `execute_with_monitor`,
`evaluate_guardrails`, `extract_python_code`).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import weave
from google.genai import types

from project_config import get_config
from prompts.goal_developer import (
    codegen_system,
    initial_codegen_user,
    next_codegen_user,
    review_system,
    review_user,
)
from schemas.goal_developer import GoalReview
from tools.developer import execute_with_monitor
from tools.explore import explore_codebase
from tools.helpers import call_llm
from utils.code_utils import extract_python_code
from utils.guardrails import build_block_summary, evaluate_guardrails
from utils.llm_utils import append_message, get_developer_tools


logger = logging.getLogger(__name__)


_CONFIG = get_config()
_LLM_CFG = _CONFIG["llm"]
_RUNTIME_CFG = _CONFIG["runtime"]
_GUARDRAIL_CFG = _CONFIG["guardrails"]

_DEVELOPER_MODEL = _LLM_CFG["developer_model"]
_DEVELOPER_TOOL_MODEL = _LLM_CFG["developer_tool_model"]
_BASELINE_CODE_TIMEOUT = _RUNTIME_CFG["baseline_code_timeout"]
_LOG_MONITOR_INTERVAL = _RUNTIME_CFG["log_monitor_interval"]

_ENABLE_LOGGING_GUARD = bool(_GUARDRAIL_CFG["logging_basicconfig_order"])
_ENABLE_LEAKAGE_GUARD = bool(_GUARDRAIL_CFG["leakage_review"])
_ENABLE_CODE_SAFETY = bool(_GUARDRAIL_CFG["enable_code_safety"])


@dataclass
class HistoryEntry:
    """One iteration of the goal-mode loop, persisted to history.json."""

    version: int
    code: str
    review: GoalReview | None  # None when guardrails blocked the version pre-execution

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "code": self.code,
            "review": self.review.model_dump() if self.review is not None else None,
        }


@weave.op()
def run_goal_mode(
    goal_text: str,
    run_dir: Path,
    max_versions: int = 50,
    max_time_seconds: int = 6 * 3600,
) -> list[HistoryEntry]:
    """Run the goal-mode loop until the reviewer reports done or limits hit.

    Writes per-version artifacts under ``run_dir/v{N}/`` and a summary
    ``history.json`` under ``run_dir/``. Returns the full history.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    history: list[HistoryEntry] = []

    v1_dir = run_dir / "v1"
    input_list: list[dict] = [
        append_message("user", initial_codegen_user(goal_text, v1_dir))
    ]

    deadline = time.monotonic() + max_time_seconds

    for version in range(1, max_versions + 1):
        if time.monotonic() >= deadline:
            logger.info("max_time_seconds reached before v%s", version)
            break

        version_dir = run_dir / f"v{version}"
        version_dir.mkdir(parents=True, exist_ok=True)
        next_dir = run_dir / f"v{version + 1}"

        # 1. Generate code
        try:
            code = _generate_code(input_list)
        except Exception:
            logger.exception("Code generation failed at v%s — aborting loop", version)
            break

        # Echo the canonical code block back into the thread as the assistant turn.
        input_list.append(append_message("assistant", f"```python\n{code}\n```"))

        # 2. Persist train.py before any further checks
        train_py = version_dir / "train.py"
        train_py.write_text(code)
        logger.info("v%s: wrote %s", version, train_py)

        # 3. Pre-execution guardrails (logging order, leakage, code safety)
        guard_report = evaluate_guardrails(
            code_text=code,
            enable_logging_guard=_ENABLE_LOGGING_GUARD,
            enable_leakage_guard=_ENABLE_LEAKAGE_GUARD,
            enable_code_safety=_ENABLE_CODE_SAFETY,
        )
        logger.info("v%s: guardrail decision = %s", version, guard_report["decision"])

        if guard_report["decision"] == "block":
            block_summary = build_block_summary(guard_report)
            (version_dir / "guardrails_block.txt").write_text(block_summary)
            history.append(HistoryEntry(version=version, code=code, review=None))
            input_list.append(
                append_message(
                    "user",
                    f"""<guardrails_block>
{block_summary}
</guardrails_block>

Your previous attempt was blocked by the pre-execution guardrails — it never ran. Fix the issues above and produce a new train.py. Save artifacts under {next_dir}/.""",
                )
            )
            continue

        # 4. Execute with LLM-based log monitoring
        output = execute_with_monitor(
            train_py,
            timeout_seconds=_BASELINE_CODE_TIMEOUT,
            log_monitor_interval=_LOG_MONITOR_INTERVAL,
            logger=logger,
        )
        (version_dir / "train.txt").write_text(output)

        # 5. Review (one-shot LLM call, NOT part of input_list)
        try:
            review = _review(goal_text, code, output)
        except Exception as exc:
            logger.exception(
                "Review LLM call failed at v%s — synthesizing degraded review",
                version,
            )
            review = GoalReview(
                reasoning=f"review call raised: {exc}",
                is_valid=False,
                violations=["review_call_error"],
                score=float("inf"),
                done=False,
                next_step="The review LLM call failed. Try a simpler approach next.",
            )
        (version_dir / "review.json").write_text(review.model_dump_json(indent=2))

        history.append(HistoryEntry(version=version, code=code, review=review))
        logger.info(
            "v%s: score=%s done=%s violations=%s",
            version,
            review.score,
            review.done,
            review.violations,
        )

        if review.done:
            logger.info("v%s: goal reached, terminating loop", version)
            break

        # 6. Append the next user instruction
        input_list.append(
            append_message("user", next_codegen_user(output, review, next_dir))
        )

    (run_dir / "history.json").write_text(
        json.dumps([entry.to_dict() for entry in history], indent=2)
    )
    logger.info("history.json written with %d entries", len(history))
    return history


def _execute_goal_tool_call(function_call) -> str:
    """Dispatch a tool call made during goal-mode code generation."""
    args = dict(function_call.args)

    if function_call.name == "explore_codebase":
        query = args["query"]
        logger.info("explore_codebase called: %s", query[:100])
        return explore_codebase(query)

    return json.dumps({"error": f"Unknown tool: {function_call.name}"})


@weave.op()
def _generate_code(input_list: list[dict]) -> str:
    """Call the codegen LLM and return the extracted python code.

    Mirrors the multi-step tool loop in ``agents/developer.py:_generate_code``:
    up to ``max_tool_steps`` rounds where the LLM may call ``explore_codebase``
    before producing the final ```python``` block. Tool-call results are
    appended to ``input_list`` so the next call sees them.
    """
    tools = get_developer_tools()
    max_tool_steps = 100

    for step in range(max_tool_steps + 1):
        is_last_step = step == max_tool_steps

        response = call_llm(
            model=_DEVELOPER_MODEL,
            system_instruction=codegen_system(),
            messages=input_list,
            text_format=None,
            function_declarations=tools if not is_last_step else [],
            enable_google_search=True,
        )

        parts = response.candidates[0].content.parts
        has_function_calls = any(
            part.function_call
            for part in parts
            if hasattr(part, "function_call")
        )

        if not has_function_calls:
            raw_text = response.text or ""
            code = extract_python_code(raw_text)
            if not code:
                raise ValueError("No ```python code block found in model response")
            return code

        function_responses = []
        for part in parts:
            if hasattr(part, "function_call") and part.function_call:
                tool_result_str = _execute_goal_tool_call(part.function_call)
                function_responses.append(
                    types.Part.from_function_response(
                        name=part.function_call.name,
                        response={"result": tool_result_str},
                    )
                )

        input_list.append(
            response.candidates[0].content.model_dump(mode="json", exclude_none=True)
        )
        if function_responses:
            input_list.append(
                types.Content(role="function", parts=function_responses).model_dump(
                    mode="json", exclude_none=True
                )
            )

    raise RuntimeError("Code generation exhausted tool steps without producing code")


@weave.op()
def _review(goal_text: str, code: str, output: str) -> GoalReview:
    """Run the one-shot review LLM call and return the parsed GoalReview."""
    return call_llm(
        model=_DEVELOPER_TOOL_MODEL,
        system_instruction=review_system(),
        messages=review_user(goal_text, code, output),
        text_format=GoalReview,
        enable_google_search=True,
    )
