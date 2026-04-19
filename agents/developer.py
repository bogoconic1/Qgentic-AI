"""Standalone developer agent.

Reads a free-form goal description, iterates a generate-execute loop until
``max_versions`` is hit. Reuses the shared low-level helpers (``call_llm``,
``execute_with_monitor``, ``evaluate_guardrails``, ``extract_python_code``).

Review is currently a no-op (see ``_review`` below) — the Main Agent will
take over review responsibilities when it lands (#230).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import weave
from google.genai import types

from project_config import get_config
from prompts.developer import (
    codegen_system,
    initial_codegen_user,
    next_codegen_user,
)
from schemas.developer import Review
from tools.developer import (
    _build_resource_header,
    execute_code,
    execute_with_monitor,
)
from agents.explorer import explore_codebase
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
    """One iteration of the developer loop, persisted to history.json."""

    version: int
    code: str
    review: Review | None  # None when guardrails blocked the version pre-execution

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "code": self.code,
            "review": self.review.model_dump() if self.review is not None else None,
        }


@weave.op()
def run_developer(
    goal_text: str,
    run_dir: Path,
    max_versions: int = 50,
) -> list[HistoryEntry]:
    """Run the developer loop until ``max_versions`` is hit.

    Writes per-version artifacts under ``run_dir/v{N}/`` and a summary
    ``history.json`` under ``run_dir/``. Returns the full history.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    history: list[HistoryEntry] = []

    v1_dir = run_dir / "v1"
    input_list: list[dict] = [
        append_message("user", initial_codegen_user(v1_dir))
    ]

    for version in range(1, max_versions + 1):
        version_dir = run_dir / f"v{version}"
        version_dir.mkdir(parents=True, exist_ok=True)
        next_dir = run_dir / f"v{version + 1}"

        # 1. Generate code
        try:
            code = _generate_code(input_list, goal_text, version_dir)
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

        # 5. Review — no-op; Main Agent will take over (#230).
        review = _review(goal_text, code, output)

        history.append(HistoryEntry(version=version, code=code, review=review))

        # 6. Append the next user instruction
        input_list.append(
            append_message("user", next_codegen_user(output, next_dir))
        )

    (run_dir / "history.json").write_text(
        json.dumps([entry.to_dict() for entry in history], indent=2)
    )
    logger.info("history.json written with %d entries", len(history))
    return history


def _execute_developer_tool_call(
    function_call,
    *,
    version_dir: Path,
    step: int,
    call_idx: int,
) -> str:
    """Dispatch a tool call made during code generation."""
    args = dict(function_call.args)

    if function_call.name == "explore_codebase":
        query = args["query"]
        logger.info("explore_codebase called: %s", query[:100])
        return explore_codebase(query)

    if function_call.name == "execute_python":
        code = args["code"]
        timeout = args.get("timeout_seconds", 300)
        logger.info(
            "execute_python called (step %s call %s, %d bytes, timeout=%ds)",
            step,
            call_idx,
            len(code),
            timeout,
        )
        work_dir = version_dir / "codegen_snippets"
        work_dir.mkdir(parents=True, exist_ok=True)
        script_file = work_dir / f"{step}_{call_idx}.py"
        script_file.write_text(_build_resource_header() + code)
        job = execute_code(str(script_file), timeout_seconds=timeout)
        return json.dumps({"output": job.result()})

    return json.dumps({"error": f"Unknown tool: {function_call.name}"})


@weave.op()
def _generate_code(input_list: list[dict], goal_text: str, version_dir: Path) -> str:
    """Call the codegen LLM and return the extracted python code.

    Up to ``max_tool_steps`` rounds where the LLM may call ``explore_codebase``
    or ``execute_python`` before producing the final ```python``` block.
    Tool-call results are appended to ``input_list`` so the next call sees them.

    The goal is embedded in the system prompt (not the user thread) so that
    it is preserved across every call regardless of conversation trimming.
    """
    tools = get_developer_tools()
    max_tool_steps = 100
    system_prompt = codegen_system(goal_text)

    for step in range(max_tool_steps + 1):
        is_last_step = step == max_tool_steps

        response = call_llm(
            model=_DEVELOPER_MODEL,
            system_instruction=system_prompt,
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
        for call_idx, part in enumerate(parts, start=1):
            if hasattr(part, "function_call") and part.function_call:
                tool_result_str = _execute_developer_tool_call(
                    part.function_call,
                    version_dir=version_dir,
                    step=step + 1,
                    call_idx=call_idx,
                )
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


def _review(goal_text: str, code: str, output: str) -> Review:
    """TODO: re-enable when Main Agent takes over review (#230)."""
    return Review()
