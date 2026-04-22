"""Main Agent — top-level LLM-driven orchestrator.

Infinite function-calling loop that picks from four tool categories every
step: `developer` / `researcher` (subagents), `analyze` (leaf), and
memory ops (`add_idea` / `remove_idea` / `update_idea`). No termination in
software — user SIGKILLs the process when satisfied.

Session-level context: `GOAL.md` (only in MainAgent's own system prompt —
subagents receive task-scoped strings via `idea` / `instruction` / `query`)
and `task/<slug>/<run_id>/ideas/INDEX.md` (always-resident, regenerated
after every idea-pool mutation).
"""

from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
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
        # Locks protect shared state when multiple function_calls in a single
        # Gemini turn dispatch concurrently (see `_step` parallel path).
        self._iter_lock = threading.Lock()   # dev_iter, research_iter, snippet counter
        self._idea_lock = threading.Lock()   # add_idea / remove_idea / update_idea
        self._log_lock = threading.Lock()    # chat-log append
        # Persistent snippet counter avoids a glob-based race where two parallel
        # `analyze` calls both see N files on disk and both write `N+1.py`.
        self._snippet_counter = len(list(self.snippets_dir.glob("*.py")))
        # google-genai requires at least one content entry per call; seed with
        # a canonical starter user turn so the first `_step()` has something to
        # send. Subsequent steps accumulate model responses + tool results in
        # this list.
        self.input_list: list[dict] = [append_message("user", _INITIAL_USER_TURN)]
        self.last_input_tokens: int | None = None
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

        self.input_list.append(
            response.candidates[0].content.model_dump(mode="json", exclude_none=True)
        )
        self._log({"role": "assistant", "content": response.candidates[0].content.model_dump(mode="json", exclude_none=True)})

        if not has_function_calls:
            logger.warning(
                "MainAgent emitted text-only response; nudging back to tool use"
            )
            self.input_list.append(
                append_message(
                    "user",
                    "Your previous turn had no function_call — every step must "
                    "be a tool call. Pick a tool from the declared palette and "
                    "call it now.",
                )
            )
            return

        call_parts = [
            p for p in parts if hasattr(p, "function_call") and p.function_call
        ]
        results: list[tuple[str, dict, str] | None] = [None] * len(call_parts)

        def _run(idx: int, fc) -> tuple[int, str, dict, str]:
            args = dict(fc.args)
            return idx, fc.name, args, self._dispatch(fc.name, args)

        if len(call_parts) == 1:
            idx, name, args, result = _run(0, call_parts[0].function_call)
            results[idx] = (name, args, result)
        else:
            logger.info(
                "MainAgent dispatching %d function_calls in parallel: %s",
                len(call_parts),
                [cp.function_call.name for cp in call_parts],
            )
            with ThreadPoolExecutor(max_workers=len(call_parts)) as ex:
                futs = [
                    ex.submit(_run, i, cp.function_call)
                    for i, cp in enumerate(call_parts)
                ]
                for fut in futs:
                    idx, name, args, result = fut.result()
                    results[idx] = (name, args, result)

        function_responses = []
        for name, args, result in results:
            function_responses.append(
                types.Part.from_function_response(
                    name=name,
                    response={"result": result},
                )
            )
            self._log({"role": "tool", "name": name, "args": args, "result": result})

        self.input_list.append(
            types.Content(role="function", parts=function_responses).model_dump(
                mode="json", exclude_none=True
            )
        )

    def _dispatch(self, name: str, args: dict) -> str:
        if name == "develop":
            if "idea_id" in args:
                idea_id = int(args["idea_id"])
                matches = list(self.ideas_dir.glob(f"{idea_id:03d}_*.md"))
                if not matches:
                    return json.dumps({"error": f"unknown idea_id: {idea_id}"})
                idea_text = matches[0].read_text()
            else:
                idea_text = self.goal_text
            with self._iter_lock:
                self.dev_iter += 1
                dev_iter_snapshot = self.dev_iter
            dev = DeveloperAgent(
                slug=self.slug,
                run_id=self.run_id,
                dev_iter=dev_iter_snapshot,
            )
            payload = dev.run(idea=idea_text)
            return json.dumps(payload)

        if name == "research":
            with self._iter_lock:
                self.research_iter += 1
                research_iter_snapshot = self.research_iter
            res = ResearcherAgent(
                slug=self.slug,
                run_id=self.run_id,
                research_iter=research_iter_snapshot,
            )
            return truncate_for_llm(res.run(instruction=args["instruction"]))

        if name == "analyze":
            with self._iter_lock:
                self._snippet_counter += 1
                snippet_num = self._snippet_counter
            script_file = self.snippets_dir / f"{snippet_num:03d}.py"
            script_file.write_text(_build_resource_header() + args["code"])
            job = execute_code(str(script_file), timeout_seconds=_EXECUTE_PYTHON_TIMEOUT)
            return json.dumps({"output": truncate_for_llm(job.result(), script_file.with_suffix(".txt"))})

        if name == "add_idea":
            with self._idea_lock:
                idea_id = add_idea(self.ideas_dir, args["title"], args["description"])
            return json.dumps({"idea_id": idea_id})

        if name == "remove_idea":
            with self._idea_lock:
                remove_idea(self.ideas_dir, args["idea_id"])
            return json.dumps({"ok": True})

        if name == "update_idea":
            with self._idea_lock:
                update_idea(self.ideas_dir, args["idea_id"], args["description"])
            return json.dumps({"ok": True})

        return json.dumps({"error": f"Unknown tool: {name}"})

    def _log(self, record: dict) -> None:
        record["ts"] = datetime.now(timezone.utc).isoformat()
        with self._log_lock:
            with self.chat_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
