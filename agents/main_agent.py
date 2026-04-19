"""MainAgent — single flat LLM-driven orchestrator.

Infinite function-calling loop. Every step is a tool call — the model picks
one of 11 tools (execute_python, read_file, glob_files, grep_code, list_dir,
bash_readonly, web_research, web_fetch, add_idea, remove_idea, update_idea)
and MainAgent dispatches it to a helper in ``tools/`` or ``utils/idea_pool``.
No sub-agents, no nested LLM loops.

Per-run layout under ``task/<slug>/<run_id>/``:
    main_agent_chat.jsonl    append-only audit of every model turn + tool result
    main_agent_snippets/     NNN.py files written by `execute_python`
    ideas/                   INDEX.md + per-idea markdown files
    web_research/            per-call audit records for `web_research`
    web_fetch/               per-call audit records for `web_fetch`
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import weave
from google.genai import types

from project_config import get_config
from prompts.main_agent import build_system
from tools.filesystem import (
    tool_bash_readonly,
    tool_glob_files,
    tool_grep_code,
    tool_list_dir,
    tool_read_file,
)
from tools.helpers import call_llm
from tools.research_net import (
    render_tool_record_markdown,
    tool_web_fetch,
    tool_web_research,
)
from tools.runtime import _build_resource_header, execute_code
from utils.compact import compact_messages, should_compact
from utils.idea_pool import add_idea, load_index, remove_idea, update_idea
from utils.llm_utils import append_message, get_main_agent_tools
from utils.output import truncate_for_llm


_INITIAL_USER_TURN = (
    "Take your first step. The session goal, the current idea pool (INDEX.md), "
    "and your tool palette are all in the system prompt. Pick a tool and call "
    "it — every step should be a tool call, never a plain text response."
)


logger = logging.getLogger(__name__)

_CONFIG = get_config()
_TASK_ROOT = Path(_CONFIG["paths"]["task_root"])
_MAIN_AGENT_MODEL = _CONFIG["llm"]["main_agent_model"]


class MainAgent:
    """Single flat agent — see module docstring."""

    def __init__(self, slug: str, run_id: str, goal_text: str):
        self.slug = slug
        self.run_id = run_id
        self.goal_text = goal_text
        self.base_dir = _TASK_ROOT / slug / run_id
        self.ideas_dir = self.base_dir / "ideas"
        self.snippets_dir = self.base_dir / "main_agent_snippets"
        self.web_research_dir = self.base_dir / "web_research"
        self.web_fetch_dir = self.base_dir / "web_fetch"
        self.chat_log = self.base_dir / "main_agent_chat.jsonl"
        for d in (
            self.ideas_dir,
            self.snippets_dir,
            self.web_research_dir,
            self.web_fetch_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
        self.tool_seq: dict[str, int] = {}
        # google-genai requires at least one content entry per call; seed with
        # a canonical starter user turn so the first `_step()` has something to send.
        self.input_list: list[dict] = [append_message("user", _INITIAL_USER_TURN)]
        self.last_input_tokens: int | None = None
        if not (self.ideas_dir / "INDEX.md").exists():
            (self.ideas_dir / "INDEX.md").write_text("# Idea pool\n\n")

    @weave.op()
    def run(self) -> None:
        tools = get_main_agent_tools()
        while True:
            self._step(tools)

    def _step(self, tools) -> None:
        system_prompt = build_system(
            slug=self.slug,
            goal_text=self.goal_text,
            index_md=load_index(self.ideas_dir),
        )
        if should_compact(self.last_input_tokens):
            self.input_list = compact_messages(
                self.input_list, model=_MAIN_AGENT_MODEL
            )
        response, self.last_input_tokens = call_llm(
            model=_MAIN_AGENT_MODEL,
            system_instruction=system_prompt,
            messages=self.input_list,
            function_declarations=tools,
            enable_google_search=False,
            include_usage=True,
        )

        parts = response.candidates[0].content.parts
        has_function_calls = any(
            p.function_call for p in parts if hasattr(p, "function_call")
        )

        assistant_dump = response.candidates[0].content.model_dump(
            mode="json", exclude_none=True
        )
        self.input_list.append(assistant_dump)
        self._log({"role": "assistant", "content": assistant_dump})

        if not has_function_calls:
            logger.warning(
                "MainAgent emitted text-only response; nudging back to tool use"
            )
            return

        function_responses = []
        for part in parts:
            if hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                args = dict(fc.args)
                result = self._dispatch(fc.name, args)
                function_responses.append(
                    types.Part.from_function_response(
                        name=fc.name,
                        response={"result": result},
                    )
                )
                self._log({"role": "tool", "name": fc.name, "args": args, "result": result})

        self.input_list.append(
            types.Content(role="function", parts=function_responses).model_dump(
                mode="json", exclude_none=True
            )
        )

    def _dispatch(self, name: str, args: dict) -> str:
        if name == "execute_python":
            return self._tool_execute_python(args)
        if name == "read_file":
            return truncate_for_llm(
                tool_read_file(
                    args["path"],
                    args.get("start_line"),
                    args.get("end_line"),
                )
            )
        if name == "glob_files":
            return truncate_for_llm(tool_glob_files(args["root"], args["pattern"]))
        if name == "grep_code":
            return truncate_for_llm(
                tool_grep_code(
                    args["root"],
                    args["pattern"],
                    args.get("file_glob", "*.py"),
                    args.get("max_results", 20),
                )
            )
        if name == "list_dir":
            return truncate_for_llm(
                tool_list_dir(args["path"], args.get("max_entries", 100))
            )
        if name == "bash_readonly":
            return truncate_for_llm(tool_bash_readonly(args["command"]))
        if name == "web_research":
            return self._tool_web_research(args)
        if name == "web_fetch":
            return self._tool_web_fetch(args)
        if name == "add_idea":
            idea_id = add_idea(self.ideas_dir, args["title"], args["description"])
            return json.dumps({"idea_id": idea_id})
        if name == "remove_idea":
            remove_idea(self.ideas_dir, args["idea_id"])
            return json.dumps({"ok": True})
        if name == "update_idea":
            update_idea(self.ideas_dir, args["idea_id"], args["description"])
            return json.dumps({"ok": True})
        return json.dumps({"error": f"Unknown tool: {name}"})

    # -- Per-tool handlers that need dispatch-local state ------------------

    def _tool_execute_python(self, args: dict) -> str:
        filename = str(args["filename"])
        if (
            "/" in filename
            or "\\" in filename
            or ".." in filename
            or not filename.endswith(".py")
            or filename.startswith(".")
        ):
            return json.dumps(
                {
                    "error": (
                        f"invalid filename {filename!r}: must end with '.py', "
                        f"contain no directory separators, and not start with '.'"
                    )
                }
            )
        script_file = self.snippets_dir / filename
        script_file.write_text(_build_resource_header() + args["code"])
        timeout = int(args.get("timeout_seconds") or 300)
        job = execute_code(str(script_file), timeout_seconds=timeout)
        output = job.result()
        return json.dumps(
            {
                "output": truncate_for_llm(
                    output, script_file.with_suffix(".txt")
                )
            }
        )

    def _tool_web_research(self, args: dict) -> str:
        seq = self._bump_seq("web_research")
        result_json = tool_web_research(args["query"], args.get("num_results"))
        self._write_audit("web_research", seq, args, result_json)
        return truncate_for_llm(result_json)

    def _tool_web_fetch(self, args: dict) -> str:
        seq = self._bump_seq("web_fetch")
        result_json = tool_web_fetch(args["url"])
        self._write_audit("web_fetch", seq, args, result_json)
        return truncate_for_llm(result_json)

    def _bump_seq(self, tool_name: str) -> int:
        seq = self.tool_seq.get(tool_name, 0) + 1
        self.tool_seq[tool_name] = seq
        return seq

    def _write_audit(
        self, tool_name: str, seq: int, args: dict, result_json: str
    ) -> None:
        """Persist full (untruncated) web_research / web_fetch results to disk."""
        if tool_name == "web_research":
            out = self.web_research_dir / f"{seq}.md"
        elif tool_name == "web_fetch":
            out = self.web_fetch_dir / f"{seq}.md"
        else:
            return
        out.write_text(render_tool_record_markdown(tool_name, seq, args, result_json))

    def _log(self, record: dict) -> None:
        record["ts"] = datetime.now(timezone.utc).isoformat()
        with self.chat_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
