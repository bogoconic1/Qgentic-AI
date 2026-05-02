"""Developer sub-agent — flat tool loop over `run_solution` + filesystem palette.

Mirrors ResearcherAgent in shape: construct with ``(slug, run_id, dev_iter)``
and call ``run(idea)`` to author SOLUTION.py + SOLUTION.md, execute the
script under guardrails + LLM monitor via ``run_solution``, debug failures
with ``web_search_stack_trace``, maintain SOLUTION.md as a living plan, and
return a structured report read back from SOLUTION.md at termination.

Per-invocation layout (owned by this module):

    task/<slug>/<run_id>/developer_v{dev_iter}/
    ├── SOLUTION.py    # agent-authored training script — scaffolded with the
    │                  # logging stanza at __init__; agent edits below it
    ├── SOLUTION.md    # the agent's living plan / report — scaffolded at
    │                  # run() entry with `# {idea}\n`, populated via
    │                  # write_file/edit_file, read back at termination
    ├── SOLUTION.txt   # produced by SOLUTION.py's FileHandler at runtime
    ├── SOLUTION.json  # produced by SOLUTION.py at end of run; required for
    │                  # run_solution to flag success
    └── submission.csv|.zip|...  # per-task submission artifact
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import weave
from dotenv import load_dotenv
from google.genai import types

from project_config import get_config
from prompts.developer import build_system, build_user
from tools.developer import run_solution as tool_run_solution
from tools.developer import web_search_stack_trace as tool_web_search_stack_trace
from tools.filesystem import execute_filesystem_tool
from tools.helpers import call_llm
from utils.compact import compact_messages, should_compact
from utils.llm_utils import append_message, get_developer_tools
from utils.output import truncate_for_llm


load_dotenv()

logger = logging.getLogger(__name__)


_CONFIG = get_config()
_LLM_CFG = _CONFIG["llm"]
_PATH_CFG = _CONFIG["paths"]

_TASK_ROOT = Path(_PATH_CFG["task_root"])
_DEVELOPER_LLM_MODEL = _LLM_CFG["developer_tool_model"]


_SOLUTION_PY_SCAFFOLD = '''\
"""SOLUTION.py — agent-authored training script.

Edit below the logging stanza. Do not move or remove the basicConfig
call — guardrails enforce that it precedes all third-party imports
and registers a FileHandler for SOLUTION.txt.
"""

import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parent / "SOLUTION.txt", mode="w"),
    ],
    format="%(asctime)s %(levelname)s %(message)s",
)

logger = logging.getLogger(__name__)

# === third-party imports below this line ===
'''


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------


def _execute_tool_call(item, state: dict) -> str:
    """Dispatch one function call from the agent's flat loop."""
    args = dict(item.args)
    tool_name = item.name

    if tool_name == "run_solution":
        result_json = tool_run_solution(state["base_dir"])
        try:
            parsed = json.loads(result_json)
        except json.JSONDecodeError:
            parsed = {}
        state["runs_made"] = state.get("runs_made", 0) + 1
        if parsed.get("success"):
            state["last_score"] = parsed.get("score")
            state["last_stats"] = parsed.get("stats", {})
            state["final_error"] = None
        else:
            state["final_error"] = parsed.get("error_kind") or parsed.get("error")
        elapsed = parsed.get("elapsed_seconds", 0.0) or 0.0
        state["elapsed_total"] = state.get("elapsed_total", 0.0) + elapsed
        return truncate_for_llm(result_json)

    if tool_name == "web_search_stack_trace":
        return truncate_for_llm(tool_web_search_stack_trace(args["query"]))

    fs_result = execute_filesystem_tool(
        tool_name, args, writable_root=state["base_dir"]
    )
    if fs_result is None:
        raise ValueError(f"Unknown tool: {tool_name}")
    return truncate_for_llm(fs_result)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


class DeveloperAgent:
    """Developer sub-agent — flat tool loop over run_solution + filesystem palette.

    Construct with ``(slug, run_id, dev_iter, conda_env?)`` and call
    ``run(idea)`` to author SOLUTION.py, execute it via run_solution,
    debug failures, maintain SOLUTION.md, and return a structured report.
    """

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
        self.solution_py_path = self.base_dir / "SOLUTION.py"
        self.solution_md_path = self.base_dir / "SOLUTION.md"
        self.chat_log = self.base_dir / "developer_chat.jsonl"
        self.base_dir.mkdir(parents=True, exist_ok=True)

        if not self.solution_py_path.exists():
            self.solution_py_path.write_text(
                _SOLUTION_PY_SCAFFOLD, encoding="utf-8"
            )

    def _load_custom_instructions(self) -> str | None:
        """Read ``task/<slug>/DEVELOPER_INSTRUCTIONS.md`` if it exists."""
        path = _TASK_ROOT / self.slug / "DEVELOPER_INSTRUCTIONS.md"
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    @weave.op()
    def run(self, idea: str) -> dict[str, Any]:
        """Run the developer flat tool loop and return a structured report.

        Scaffolds ``SOLUTION.md`` with ``# {idea}\\n`` (idempotent), runs the
        tool loop until the LLM returns a text-only response, reads
        ``SOLUTION.md`` back as the report, and packages a return dict for
        MainAgent.
        """
        if not self.solution_md_path.exists():
            self.solution_md_path.write_text(f"# {idea}\n", encoding="utf-8")

        logger.info(
            "DeveloperAgent.run slug=%s run_id=%s dev_iter=%d dir=%s",
            self.slug,
            self.run_id,
            self.dev_iter,
            self.base_dir,
        )

        system_prompt = build_system(
            writable_root=str(self.base_dir),
            custom_instructions=self._load_custom_instructions(),
        )
        user_prompt = build_user(idea)
        tools = get_developer_tools()
        state: dict = {
            "base_dir": self.base_dir,
            "runs_made": 0,
            "last_score": None,
            "last_stats": {},
            "final_error": None,
            "elapsed_total": 0.0,
        }
        input_list: list[dict] = [append_message("user", user_prompt)]
        last_input_tokens: int | None = None

        step = 0
        while True:
            step += 1
            logger.info("DeveloperAgent step %d", step)

            if should_compact(last_input_tokens):
                input_list = compact_messages(
                    input_list, model=_DEVELOPER_LLM_MODEL
                )

            response, last_input_tokens = call_llm(
                model=_DEVELOPER_LLM_MODEL,
                system_instruction=system_prompt,
                function_declarations=tools,
                messages=input_list,
                enable_google_search=False,
                include_usage=True,
            )

            parts = response.candidates[0].content.parts
            has_function_calls = any(
                part.function_call
                for part in parts
                if hasattr(part, "function_call")
            )

            assistant_content = response.candidates[0].content.model_dump(
                mode="json", exclude_none=True
            )
            input_list.append(assistant_content)
            self._log({"role": "assistant", "content": assistant_content})

            if not has_function_calls:
                logger.info("DeveloperAgent completed at step %d", step)
                break

            function_responses = []
            for part in parts:
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    name = fc.name
                    args = dict(fc.args)
                    tool_result_str = _execute_tool_call(fc, state)
                    function_responses.append(
                        types.Part.from_function_response(
                            name=name,
                            response={"result": tool_result_str},
                        )
                    )
                    self._log({
                        "role": "tool",
                        "name": name,
                        "args": args,
                        "result": tool_result_str,
                    })

            if function_responses:
                input_list.append(
                    types.Content(
                        role="function", parts=function_responses
                    ).model_dump(mode="json", exclude_none=True)
                )

        report = self.solution_md_path.read_text(encoding="utf-8")
        if state["runs_made"] == 0:
            status = "no_run"
        elif state["last_score"] is not None:
            status = "success"
        else:
            status = "failed"
        logger.info(
            "DeveloperAgent.run finished slug=%s run_id=%s dev_iter=%d status=%s",
            self.slug,
            self.run_id,
            self.dev_iter,
            status,
        )

        return {
            "status": status,
            "version_dir": str(self.base_dir),
            "summary": {
                "score": state["last_score"],
                "stats": state["last_stats"],
                "elapsed_seconds": state["elapsed_total"],
                "runs_made": state["runs_made"],
                "final_error": state["final_error"],
            },
            "report": report,
        }

    def _log(self, record: dict) -> None:
        record["ts"] = datetime.now(timezone.utc).isoformat()
        with self.chat_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
