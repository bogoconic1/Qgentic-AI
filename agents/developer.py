"""Developer subagent — one training iteration per call.

Parallel to ``ResearcherAgent`` in shape: construct with
``(slug, run_id, dev_iter, goal_text, conda_env)`` and call ``run(idea)`` to
produce one training iteration worth of artifacts. An unbounded internal
retry loop iterates attempts (``developer_<dev_iter>/<k>/``) until one
produces a valid ``train_stats.json``; the subagent only exits on success
or a runtime precondition failure that happens before the loop starts.

Evaluation is the developer's responsibility — the generated ``train.py``
computes its own score and writes ``train_stats.json`` at end-of-run.
There is no separate ``metric.py`` artifact.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any

import weave
from google.genai import types

from agents.explorer import explore_codebase
from project_config import get_config
from prompts.developer import (
    codegen_initial_user,
    codegen_retry_user,
    codegen_system,
)
from tools.developer import (
    _build_resource_header,
    execute_code,
    execute_with_monitor,
    web_search_stack_trace,
)
from tools.helpers import call_llm
from utils.code_utils import extract_python_code
from utils.guardrails import build_block_summary, evaluate_guardrails
from utils.llm_utils import append_message, get_developer_tools


logger = logging.getLogger(__name__)


_CONFIG = get_config()
_LLM_CFG = _CONFIG["llm"]
_RUNTIME_CFG = _CONFIG["runtime"]
_GUARDRAIL_CFG = _CONFIG["guardrails"]
_PATH_CFG = _CONFIG["paths"]

_DEVELOPER_MODEL = _LLM_CFG["developer_model"]
_BASELINE_CODE_TIMEOUT = _RUNTIME_CFG["baseline_code_timeout"]
_LOG_MONITOR_INTERVAL = _RUNTIME_CFG["log_monitor_interval"]
_TASK_ROOT = Path(_PATH_CFG["task_root"])

_ENABLE_LOGGING_GUARD = bool(_GUARDRAIL_CFG["logging_basicconfig_order"])
_ENABLE_LEAKAGE_GUARD = bool(_GUARDRAIL_CFG["leakage_review"])
_ENABLE_CODE_SAFETY = bool(_GUARDRAIL_CFG["enable_code_safety"])

_MAX_CODEGEN_TOOL_STEPS = 100
_STDOUT_TAIL_LINES = 200


def _build_base_dir_header(slug: str) -> str:
    """Prepended to every generated `train.py` so BASE_DIR works locally + on Kaggle."""
    return (
        'import os\n'
        'from pathlib import Path\n'
        f'SLUG = "{slug}"\n'
        'IS_KAGGLE = os.getenv("KAGGLE_KERNEL_RUN_TYPE") is not None\n'
        'BASE_DIR = Path(f"/kaggle/input/{SLUG}") if IS_KAGGLE else Path(f"task/{SLUG}")\n'
        '\n'
    )


def _has_traceback(stderr: str) -> bool:
    return "Traceback (most recent call last)" in stderr


def _extract_hints(enriched: str) -> str | None:
    marker = "This is how you can fix the error:"
    if marker not in enriched:
        return None
    return enriched.split(marker, 1)[1].strip() or None


def _tail(text: str, n_lines: int = _STDOUT_TAIL_LINES) -> str:
    return "\n".join(text.splitlines()[-n_lines:])


class DeveloperAgent:
    """One-training-iteration subagent.

    Parallel to ``Orchestrator`` / ``ResearcherAgent`` in shape. Construct
    with the per-call identity (``slug``, ``run_id``, ``dev_iter``, the
    session's ``goal_text`` from ``GOAL.md``, and an optional ``conda_env``),
    then call ``run(idea)``. Returns a structured payload describing success
    or precondition failure.
    """

    def __init__(
        self,
        slug: str,
        run_id: str,
        dev_iter: int,
        goal_text: str,
        conda_env: str | None = None,
    ):
        self.slug = slug
        self.run_id = run_id
        self.dev_iter = dev_iter
        self.goal_text = goal_text
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

    @weave.op()
    def run(self, idea: str | None = None) -> dict[str, Any]:
        """Run one developer iteration; retry on failure until success."""
        previous_code = self._find_previous_code()
        system_prompt = codegen_system(
            goal_text=self.goal_text,
            idea=idea,
            previous_code=previous_code,
        )

        input_list: list[dict] = [
            append_message("user", codegen_initial_user(self.base_dir / "1")),
        ]

        stderr_cache: dict[str, str | None] = {}
        k = 0
        while True:
            k += 1
            attempt_dir = self.base_dir / str(k)
            attempt_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                "DeveloperAgent slug=%s dev_iter=%d attempt=%d dir=%s",
                self.slug,
                self.dev_iter,
                k,
                attempt_dir,
            )

            try:
                code = self._generate_code(input_list, system_prompt, attempt_dir)
            except Exception as exc:
                logger.exception(
                    "codegen failed at attempt %d — feeding back and retrying", k
                )
                input_list.append(
                    append_message(
                        "user",
                        f"Code generation failed: {exc}. Produce a train.py for attempt {k + 1}.",
                    )
                )
                continue

            input_list.append(append_message("assistant", f"```python\n{code}\n```"))

            final_code = _build_base_dir_header(self.slug) + code
            train_py = attempt_dir / "train.py"
            train_py.write_text(final_code)

            guard_report = evaluate_guardrails(
                code_text=code,
                enable_logging_guard=_ENABLE_LOGGING_GUARD,
                enable_leakage_guard=_ENABLE_LEAKAGE_GUARD,
                enable_code_safety=_ENABLE_CODE_SAFETY,
            )
            if guard_report["decision"] == "block":
                block_summary = build_block_summary(guard_report)
                (attempt_dir / "guardrails_block.txt").write_text(block_summary)
                input_list.append(
                    append_message(
                        "user",
                        f"<guardrails_block>\n{block_summary}\n</guardrails_block>\n\nYour previous attempt was blocked by the pre-execution guardrails — it never ran. Fix the issues and produce a new train.py for attempt {k + 1}.",
                    )
                )
                continue

            output = execute_with_monitor(
                train_py,
                timeout_seconds=_BASELINE_CODE_TIMEOUT,
                log_monitor_interval=_LOG_MONITOR_INTERVAL,
                logger=logger,
                conda_env=self.conda_env,
            )
            (attempt_dir / "train.txt").write_text(output)

            stats_path = attempt_dir / "train_stats.json"
            if stats_path.exists():
                try:
                    stats = json.loads(stats_path.read_text())
                    score = float(stats.get("score"))
                    if math.isfinite(score):
                        return {
                            "status": "success",
                            "code": final_code,
                            "code_path": str(train_py),
                            "summary": {
                                "score": score,
                                "stats": stats,
                                "stdout_tail": _tail(output),
                                "attempts_made": k,
                                "final_error": None,
                            },
                        }
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    logger.warning(
                        "train_stats.json present but unparseable at %s: %s",
                        stats_path,
                        exc,
                    )

            trace_hash = hashlib.md5(output.encode()).hexdigest()
            if trace_hash in stderr_cache:
                hints = stderr_cache[trace_hash]
            else:
                hints = None
                if _has_traceback(output):
                    try:
                        enriched = web_search_stack_trace(output)
                        hints = _extract_hints(enriched)
                    except Exception:
                        logger.exception("web_search_stack_trace failed at attempt %d", k)
                stderr_cache[trace_hash] = hints

            input_list.append(
                append_message(
                    "user",
                    codegen_retry_user(
                        output=output,
                        hints=hints,
                        missing_stats=not stats_path.exists(),
                        attempt_dir=self.base_dir / str(k + 1),
                    ),
                )
            )

    def _generate_code(
        self,
        input_list: list[dict],
        system_prompt: str,
        attempt_dir: Path,
    ) -> str:
        """Inner codegen tool-loop — identical shape to the pre-rewrite version."""
        tools = get_developer_tools()

        for step in range(_MAX_CODEGEN_TOOL_STEPS + 1):
            is_last_step = step == _MAX_CODEGEN_TOOL_STEPS

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
                    raise ValueError(
                        "No ```python code block found in model response"
                    )
                return code

            function_responses = []
            for call_idx, part in enumerate(parts, start=1):
                if hasattr(part, "function_call") and part.function_call:
                    tool_result_str = self._execute_developer_tool_call(
                        part.function_call,
                        attempt_dir=attempt_dir,
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
                response.candidates[0].content.model_dump(
                    mode="json", exclude_none=True
                )
            )
            if function_responses:
                input_list.append(
                    types.Content(
                        role="function", parts=function_responses
                    ).model_dump(mode="json", exclude_none=True)
                )

        raise RuntimeError(
            "Code generation exhausted tool steps without producing code"
        )

    def _execute_developer_tool_call(
        self,
        function_call,
        *,
        attempt_dir: Path,
        step: int,
        call_idx: int,
    ) -> str:
        """Dispatch a tool call made during code generation."""
        args = dict(function_call.args)

        if function_call.name == "explore_codebase":
            query = args["query"]
            logger.info("explore_codebase called: %s", query[:100])
            return explore_codebase(query, goal_text=self.goal_text)

        if function_call.name == "analyze":
            code = args["code"]
            timeout = args.get("timeout_seconds", 300)
            logger.info(
                "analyze called (step %s call %s, %d bytes, timeout=%ds)",
                step,
                call_idx,
                len(code),
                timeout,
            )
            work_dir = attempt_dir / "codegen_snippets"
            work_dir.mkdir(parents=True, exist_ok=True)
            script_file = work_dir / f"{step}_{call_idx}.py"
            script_file.write_text(_build_resource_header() + code)
            job = execute_code(str(script_file), timeout_seconds=timeout)
            return json.dumps({"output": job.result()})

        return json.dumps({"error": f"Unknown tool: {function_call.name}"})
