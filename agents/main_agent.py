"""Main Agent — top-level LLM-driven orchestrator.

Infinite function-calling loop that picks from four tool categories every
step: `developer` / `researcher` (subagents), `analyze` (leaf), and
memory ops (`add_idea` / `remove_idea` / `update_idea`). No termination in
software — user SIGKILLs the process when satisfied.

Session-level context: `GOAL.md` (only in MainAgent's own system prompt —
subagents receive task-scoped strings via `idea` / `instruction` / `query`),
`task/<slug>/<run_id>/ideas/INDEX.md` (always-resident, regenerated after
every idea-pool mutation), and `task/<slug>/<run_id>/MAIN.md` (the agent's
living plan — scaffolded with `# {goal_text}`, maintained by the agent
via `write_file` / `edit_file`).
"""

from __future__ import annotations

import collections
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import weave

from agents.researcher import ResearcherAgent
from project_config import get_config
from prompts.main_agent import build_system
from tools.developer import (
    SOLUTION_PY_SCAFFOLD,
    run_solution as tool_run_solution,
    web_search_stack_trace as tool_web_search_stack_trace,
)
from tools.filesystem import execute_filesystem_tool
from tools.helpers import call_llm
from utils.compact import compact_messages, should_compact
from utils.idea_pool import add_idea, load_index, remove_idea, update_idea
from utils.llm_utils import append_message, get_main_agent_tools
from utils.output import truncate_for_llm
from utils.skills import load_agent_skills, render_skill_catalog


_INITIAL_USER_TURN = (
    "Take your first step. The session goal, the current idea pool (INDEX.md), "
    "and your tool palette are all in the system prompt. Begin with a 1-3 "
    "sentence prose block stating your read of the goal and your opening move, "
    "then issue the tool call(s) — every turn must lead with prose, then act."
)

_STUCK_REPEAT_THRESHOLD = 3

_STUCK_NUDGE = (
    "You've made the same tool call(s) for the last "
    f"{_STUCK_REPEAT_THRESHOLD} turns. That's a sign you're out of ideas, not "
    "that the work is done. **Push on.** Concrete options: add a fresh idea "
    'via `add_idea`; call `research(instruction="...")` to web-search for '
    "unblocking ideas; inspect a `developer_v{N}/` directory you haven't reviewed "
    "yet; reread INDEX.md and pick the next-most-promising idea. **Never "
    "stop** — the session has no termination condition. The user will SIGKILL "
    "when satisfied."
)

_TEXT_ONLY_TERMINATE_THRESHOLD = 3


logger = logging.getLogger(__name__)

_CONFIG = get_config()
_TASK_ROOT = Path(_CONFIG["paths"]["task_root"])
_MAIN_AGENT_MODEL = _CONFIG["llm"]["main_agent_model"]


class MainAgent:
    def __init__(self, slug: str, run_id: str, goal_text: str):
        self.slug = slug
        self.run_id = run_id
        self.goal_text = goal_text
        self.base_dir = _TASK_ROOT / slug / run_id
        self.ideas_dir = self.base_dir / "ideas"
        self.chat_log = self.base_dir / "main_agent_chat.jsonl"
        self.main_md_path = self.base_dir / "MAIN.md"
        self.ideas_dir.mkdir(parents=True, exist_ok=True)
        self.dev_iter = 0
        self.research_iter = 0
        self._iter_lock = threading.Lock()
        self._idea_lock = threading.Lock()
        self._log_lock = threading.Lock()
        self.input_list: list[dict] = [append_message("user", _INITIAL_USER_TURN)]
        self.last_input_tokens: int | None = None
        self._previous_response_id: str | None = None
        self._next_input: list[dict] | None = None
        self._recent_call_sigs: collections.deque[tuple[tuple[str, str], ...]] = (
            collections.deque(maxlen=_STUCK_REPEAT_THRESHOLD)
        )
        self._consecutive_text_only = 0
        self._done = False
        self._skill_catalog = render_skill_catalog(load_agent_skills())
        if not (self.ideas_dir / "INDEX.md").exists():
            (self.ideas_dir / "INDEX.md").write_text("# Idea pool\n\n")
        if not self.main_md_path.exists():
            self.main_md_path.write_text(f"# {goal_text}\n", encoding="utf-8")

    @weave.op()
    def run(self) -> None:
        tools = get_main_agent_tools()
        while not self._done:
            self._step(tools)

    def _step(self, tools) -> None:
        system_prompt = build_system(
            slug=self.slug,
            goal_text=self.goal_text,
            index_md=load_index(self.ideas_dir),
            writable_root=str(self.base_dir),
            skill_catalog=self._skill_catalog,
        )
        if should_compact(self.last_input_tokens):
            self.input_list = compact_messages(self.input_list, model=_MAIN_AGENT_MODEL)
            self._previous_response_id = None
            self._next_input = None
        api_input = (
            self._next_input
            if self._next_input is not None and self._previous_response_id is not None
            else self.input_list
        )
        response, self.last_input_tokens = call_llm(
            model=_MAIN_AGENT_MODEL,
            system_instruction=system_prompt,
            messages=api_input,
            function_declarations=tools,
            enable_google_search=False,
            include_usage=True,
            previous_response_id=self._previous_response_id,
        )
        self._previous_response_id = response.id
        self._next_input = None

        output_items = response.output
        function_calls = [item for item in output_items if item.type == "function_call"]
        has_function_calls = len(function_calls) > 0

        output_text = response.output_text or ""
        if output_text:
            self.input_list.append(append_message("assistant", output_text))
        for fc in function_calls:
            self.input_list.append(
                {
                    "type": "function_call",
                    "call_id": fc.call_id,
                    "name": fc.name,
                    "arguments": fc.arguments,
                }
            )

        self._log(
            {
                "role": "assistant",
                "content": output_text,
                "function_calls": [
                    {"name": fc.name, "arguments": fc.arguments, "call_id": fc.call_id}
                    for fc in function_calls
                ],
            }
        )

        if not has_function_calls:
            self._consecutive_text_only += 1
            if self._consecutive_text_only >= _TEXT_ONLY_TERMINATE_THRESHOLD:
                logger.info(
                    "MainAgent: %d consecutive text-only turns — agent appears "
                    "done; terminating run",
                    self._consecutive_text_only,
                )
                self._done = True
            return
        self._consecutive_text_only = 0

        results: list[tuple[str, dict, str, str] | None] = [None] * len(function_calls)

        def _run(idx: int, fc) -> tuple[int, str, dict, str, str]:
            args = json.loads(fc.arguments)
            return idx, fc.name, args, self._dispatch(fc.name, args), fc.call_id

        if len(function_calls) == 1:
            idx, name, args, result, call_id = _run(0, function_calls[0])
            results[idx] = (name, args, result, call_id)
        else:
            logger.info(
                "MainAgent dispatching %d function_calls in parallel: %s",
                len(function_calls),
                [fc.name for fc in function_calls],
            )
            with ThreadPoolExecutor(max_workers=len(function_calls)) as ex:
                futs = [ex.submit(_run, i, fc) for i, fc in enumerate(function_calls)]
                for fut in futs:
                    idx, name, args, result, call_id = fut.result()
                    results[idx] = (name, args, result, call_id)

        self._next_input = []
        for name, args, result, call_id in results:
            item = {
                "type": "function_call_output",
                "call_id": call_id,
                "output": result,
            }
            self.input_list.append(item)
            self._next_input.append(item)
            self._log({"role": "tool", "name": name, "args": args, "result": result})

        turn_sig = tuple(
            (name, json.dumps(args, sort_keys=True, ensure_ascii=False))
            for name, args, _, _ in results
        )
        self._recent_call_sigs.append(turn_sig)
        if (
            len(self._recent_call_sigs) == _STUCK_REPEAT_THRESHOLD
            and len(set(self._recent_call_sigs)) == 1
        ):
            logger.warning(
                "MainAgent appears stuck: same %d-call turn repeated %d times "
                "(%s) — injecting unstucking nudge",
                len(turn_sig),
                _STUCK_REPEAT_THRESHOLD,
                [n for n, _ in turn_sig],
            )
            nudge_msg = append_message("user", _STUCK_NUDGE)
            self.input_list.append(nudge_msg)
            self._next_input.append(nudge_msg)
            self._log({"role": "user", "content": nudge_msg})
            self._recent_call_sigs.clear()

    def _dispatch(self, name: str, args: dict) -> str:
        if name == "start_dev_session":
            with self._iter_lock:
                self.dev_iter += 1
                dev_iter_snapshot = self.dev_iter
            version_dir = self.base_dir / f"developer_v{dev_iter_snapshot}"
            version_dir.mkdir(parents=True, exist_ok=True)

            solution_py_path = version_dir / "SOLUTION.py"
            if not solution_py_path.exists():
                solution_py_path.write_text(SOLUTION_PY_SCAFFOLD, encoding="utf-8")

            header = "# SOLUTION\n"
            idea_id = args.get("idea_id")
            if idea_id is not None:
                matches = list(self.ideas_dir.glob(f"{int(idea_id):03d}_*.md"))
                if matches:
                    title = matches[0].read_text().splitlines()[0].lstrip("# ").strip()
                    header = f"# {title}\n"
            solution_md_path = version_dir / "SOLUTION.md"
            if not solution_md_path.exists():
                solution_md_path.write_text(header, encoding="utf-8")

            return json.dumps({"version_dir": str(version_dir)})

        if name == "run_solution":
            version_dir = args.get("version_dir")
            if not version_dir:
                return json.dumps(
                    {"error": "version_dir is required. Call start_dev_session first."}
                )
            return truncate_for_llm(tool_run_solution(version_dir))

        if name == "web_search_stack_trace":
            return truncate_for_llm(tool_web_search_stack_trace(args["query"]))

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

        fs_result = execute_filesystem_tool(name, args, writable_root=self.base_dir)
        if fs_result is not None:
            return truncate_for_llm(fs_result)

        return json.dumps({"error": f"Unknown tool: {name}"})

    def _log(self, record: dict) -> None:
        record["ts"] = datetime.now(timezone.utc).isoformat()
        with self._log_lock:
            with self.chat_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
