"""Main Agent — top-level LLM-driven orchestrator.

Infinite function-calling loop that picks from four tool categories every
step: `developer` / `researcher` (subagents), `analyze` (leaf), and
memory ops (`add_idea` / `remove_idea` / `update_idea`). No termination in
software — user SIGKILLs the process when satisfied.

Session-level context: `GOAL.md` (threaded into every subagent's system
prompt) and `task/<slug>/<run_id>/ideas/INDEX.md` (always-resident, regenerated
after every idea-pool mutation).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import weave
from google.genai import types

from agents.developer import DeveloperAgent
from agents.researcher import ResearcherAgent
from project_config import get_config
from prompts.main_agent import build_system
from tools.developer import _build_resource_header, execute_code
from tools.helpers import call_llm
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
_EXECUTE_PYTHON_TIMEOUT = 600


class MainAgent:
    """Top-level LLM-driven orchestrator. See module docstring."""

    def __init__(self, slug: str, run_id: str, goal_text: str):
        self.slug = slug
        self.run_id = run_id
        self.goal_text = goal_text
        self.base_dir = _TASK_ROOT / slug / run_id
        self.ideas_dir = self.base_dir / "ideas"
        self.snippets_dir = self.base_dir / "main_agent_snippets"
        self.chat_log = self.base_dir / "main_agent_chat.jsonl"
        for d in (self.ideas_dir, self.snippets_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.dev_iter = 0
        self.research_iter = 0
        # google-genai requires at least one content entry per call; seed with
        # a canonical starter user turn so the first `_step()` has something to
        # send. Subsequent steps accumulate model responses + tool results in
        # this list.
        self.input_list: list[dict] = [append_message("user", _INITIAL_USER_TURN)]
        # Ensure INDEX.md exists so `load_index` has something to read.
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
        response = call_llm(
            model=_MAIN_AGENT_MODEL,
            system_instruction=system_prompt,
            messages=self.input_list,
            function_declarations=tools,
            enable_google_search=False,
        )

        parts = response.candidates[0].content.parts
        has_function_calls = any(
            p.function_call for p in parts if hasattr(p, "function_call")
        )

        self.input_list.append(
            response.candidates[0].content.model_dump(mode="json", exclude_none=True)
        )
        self._log({"role": "assistant", "content": response.candidates[0].content.model_dump(mode="json", exclude_none=True)})

        if not has_function_calls:
            logger.warning(
                "MainAgent emitted text-only response; nudging back to tool use"
            )
            return

        function_responses = []
        for part in parts:
            if hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                result = self._dispatch(fc.name, dict(fc.args))
                function_responses.append(
                    types.Part.from_function_response(
                        name=fc.name,
                        response={"result": result},
                    )
                )
                self._log({"role": "tool", "name": fc.name, "args": dict(fc.args), "result": result})

        self.input_list.append(
            types.Content(role="function", parts=function_responses).model_dump(
                mode="json", exclude_none=True
            )
        )

    def _dispatch(self, name: str, args: dict) -> str:
        if name == "develop":
            idea_text: str | None = None
            if "idea_id" in args:
                idea_id = int(args["idea_id"])
                matches = list(self.ideas_dir.glob(f"{idea_id:03d}_*.md"))
                if not matches:
                    return json.dumps({"error": f"unknown idea_id: {idea_id}"})
                idea_text = matches[0].read_text()
            self.dev_iter += 1
            dev = DeveloperAgent(
                slug=self.slug,
                run_id=self.run_id,
                dev_iter=self.dev_iter,
                goal_text=self.goal_text,
            )
            payload = dev.run(idea=idea_text)
            return json.dumps(payload)

        if name == "research":
            self.research_iter += 1
            res = ResearcherAgent(
                slug=self.slug,
                run_id=self.run_id,
                research_iter=self.research_iter,
                goal_text=self.goal_text,
            )
            return truncate_for_llm(res.run(instruction=args["instruction"]))

        if name == "analyze":
            snippet_num = len(list(self.snippets_dir.glob("*.py"))) + 1
            script_file = self.snippets_dir / f"{snippet_num:03d}.py"
            script_file.write_text(_build_resource_header() + args["code"])
            job = execute_code(str(script_file), timeout_seconds=_EXECUTE_PYTHON_TIMEOUT)
            return json.dumps({"output": truncate_for_llm(job.result(), script_file.with_suffix(".txt"))})

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

    def _log(self, record: dict) -> None:
        record["ts"] = datetime.now(timezone.utc).isoformat()
        with self.chat_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
