"""Deep Research sub-agent, exposed as ``ResearcherAgent``.

A sub-agent the Main Agent (or orchestrator) instantiates with a run's
``(slug, run_id, research_iter)`` and then calls ``run(instruction)`` on.
It runs a multi-step tool loop over three inner tools — ``web_research``
(Exa discovery), ``web_fetch`` (Firecrawl scrape), and ``write_python_code``
(subprocess exec) — and emits a markdown report. Gemini's built-in
``google_search`` is disabled inside the sub-agent so every URL the LLM
dereferences is traceable back to a prior tool result (no invented URLs).

Tool outputs (``web_research``, ``web_fetch``, ``write_python_code``) are
truncated to 30k chars before flowing back to the LLM. Full content is
preserved in per-call audit records on disk.

Per-invocation layout (owned by this module):

    task/<slug>/<run_id>/research_<research_iter>/
    ├── scripts/       # executed Python scripts (re-runnable standalone)
    │   └── <seq>.py
    ├── web_research/  # per-call audit records for web_research
    │   └── <seq>.md
    ├── web_fetch/     # per-call audit records for web_fetch
    │   └── <seq>.md
    └── PLAN_<research_iter>.md   # final markdown report
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import weave
from dotenv import load_dotenv
from exa_py import Exa
from firecrawl import Firecrawl
from google.genai import types

from project_config import get_config
from prompts.research import build_system, build_user
from tools.developer import _build_resource_header, execute_code
from tools.filesystem import execute_filesystem_tool
from tools.helpers import call_llm
from utils.compact import compact_messages, should_compact
from utils.llm_utils import append_message, get_deep_research_tools
from utils.output import truncate_for_llm


load_dotenv()

logger = logging.getLogger(__name__)


_CONFIG = get_config()
_LLM_CFG = _CONFIG["llm"]
_RUNTIME_CFG = _CONFIG["runtime"]
_PATH_CFG = _CONFIG["paths"]

_TASK_ROOT = Path(_PATH_CFG["task_root"])
_DEEP_RESEARCH_LLM_MODEL = _LLM_CFG["developer_tool_model"]
_WRITE_PYTHON_CODE_TIMEOUT_SECONDS = _RUNTIME_CFG["write_python_code_timeout_seconds"]


# ---------------------------------------------------------------------------
# Inner tool implementations
# ---------------------------------------------------------------------------


def _tool_web_research(query: str, num_results: int | None) -> str:
    """Exa neural search. Returns full text for each result — no truncation."""
    logger.info("web_research query=%r num_results=%s", query, num_results)

    exa_client = Exa(api_key=os.environ["EXA_API_KEY"])
    search_kwargs = {"type": "auto", "text": True}
    if num_results is not None:
        search_kwargs["num_results"] = num_results

    try:
        search_response = exa_client.search_and_contents(query, **search_kwargs)
    except Exception as exc:
        logger.exception("Exa search_and_contents failed")
        return json.dumps({"error": f"exa search failed: {exc}"})

    results = [
        {
            "url": r.url,
            "title": r.title,
            "text": r.text or "",
            "published_date": r.published_date,
        }
        for r in search_response.results
    ]
    if not results:
        return json.dumps({"error": "no results — try reformulating the query"})

    return json.dumps({"results": results})


def _tool_web_fetch(url: str) -> str:
    """Firecrawl scrape → markdown. Full content, no truncation."""
    logger.info("web_fetch url=%s", url)

    firecrawl_client = Firecrawl(api_key=os.environ["FIRECRAWL_API_KEY"])

    try:
        doc = firecrawl_client.scrape(
            url, only_main_content=True, formats=["markdown"]
        )
    except Exception as exc:
        logger.exception("Firecrawl scrape failed for %s", url)
        return json.dumps({"error": f"firecrawl scrape failed: {exc}"})

    title = doc.metadata.title if doc.metadata is not None else None
    markdown = doc.markdown or ""

    return json.dumps({"url": url, "title": title or url, "markdown": markdown})


def _tool_write_python_code(code: str, seq: int, scripts_dir: Path) -> str:
    """Save + exec a Python script in the scripts dir. Return full stdout/stderr."""
    script_path = scripts_dir / f"{seq}.py"
    script_path.write_text(_build_resource_header() + code)
    logger.info("write_python_code seq=%d path=%s", seq, script_path)

    try:
        job = execute_code(
            str(script_path), timeout_seconds=_WRITE_PYTHON_CODE_TIMEOUT_SECONDS
        )
        output = job.result()
    except Exception as exc:
        logger.exception("write_python_code execution failed")
        return json.dumps({"error": f"execution failed: {exc}"})

    return json.dumps({"output": truncate_for_llm(output, script_path.with_suffix(".txt"))})


def _render_tool_record_markdown(
    tool_name: str, seq: int, args: dict, result_json: str
) -> str:
    """Render one tool call (args + result) as a self-contained markdown audit note."""
    result = json.loads(result_json)
    header = f"# {tool_name} #{seq}\n\n"

    if "error" in result:
        return header + f"**ERROR:** {result['error']}\n"

    if tool_name == "web_research":
        lines = [header, f"**Query:** {args['query']}\n"]
        if args.get("num_results") is not None:
            lines.append(f"**Requested num_results:** {args['num_results']}\n")
        lines.append(f"**Num results returned:** {len(result['results'])}\n\n---\n\n")
        for idx, item in enumerate(result["results"], start=1):
            lines.append(f"## Result {idx}: {item['title'] or '(no title)'}\n\n")
            lines.append(f"- **URL:** {item['url']}\n")
            if item.get("published_date"):
                lines.append(f"- **Published:** {item['published_date']}\n")
            lines.append(f"\n{item['text']}\n\n---\n\n")
        return "".join(lines)

    if tool_name == "web_fetch":
        return (
            header
            + f"**URL:** {result['url']}\n"
            + f"**Title:** {result['title']}\n\n"
            + "---\n\n"
            + f"{result['markdown']}\n"
        )

    raise ValueError(f"Unknown tool_name for record rendering: {tool_name}")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def _execute_tool_call(item, state: dict) -> str:
    args = dict(item.args)
    tool_name = item.name

    tool_seq = state["tool_seq"].get(tool_name, 0) + 1
    state["tool_seq"][tool_name] = tool_seq

    if tool_name == "web_research":
        result_json = _tool_web_research(args["query"], args.get("num_results"))
    elif tool_name == "web_fetch":
        result_json = _tool_web_fetch(args["url"])
    elif tool_name == "write_python_code":
        result_json = _tool_write_python_code(
            args["code"], tool_seq, state["scripts_dir"]
        )
    else:
        fs_result = execute_filesystem_tool(tool_name, args)
        if fs_result is None:
            raise ValueError(f"Unknown tool: {tool_name}")
        return truncate_for_llm(fs_result)

    # Write audit record with FULL result before truncating for the LLM.
    if tool_name in ("web_research", "web_fetch"):
        record_path = state["research_dir"] / tool_name / f"{tool_seq}.md"
        record_path.write_text(
            _render_tool_record_markdown(tool_name, tool_seq, args, result_json)
        )

    return truncate_for_llm(result_json)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


class ResearcherAgent:
    """Deep Research sub-agent.

    Parallel to ``Orchestrator`` in shape: construct with
    ``(slug, run_id, research_iter)`` and call ``run(instruction)`` to
    produce a markdown report.
    """

    def __init__(
        self,
        slug: str,
        run_id: str,
        research_iter: int,
    ):
        self.slug = slug
        self.run_id = run_id
        self.research_iter = research_iter
        self.research_dir = _TASK_ROOT / slug / run_id / f"research_{research_iter}"
        self.scripts_dir = self.research_dir / "scripts"
        self.web_research_dir = self.research_dir / "web_research"
        self.web_fetch_dir = self.research_dir / "web_fetch"

    def _load_custom_instructions(self) -> str | None:
        """Read `task/<slug>/RESEARCHER_INSTRUCTIONS.md` if it exists."""
        path = _TASK_ROOT / self.slug / "RESEARCHER_INSTRUCTIONS.md"
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    @weave.op()
    def run(self, instruction: str) -> str:
        """Run the Deep Research loop and return a markdown report.

        Creates ``task/<slug>/<run_id>/research_<research_iter>/``, runs the
        tool loop, and persists the final markdown as ``PLAN_<research_iter>.md``
        inside that directory. The markdown is also returned.

        Args:
            instruction: Free-form research instruction — as long as needed.

        Returns:
            Free-form markdown report with URL citations.
        """
        for d in (self.scripts_dir, self.web_research_dir, self.web_fetch_dir):
            d.mkdir(parents=True, exist_ok=True)

        logger.info(
            "ResearcherAgent.run slug=%s run_id=%s iter=%d dir=%s instruction=%r",
            self.slug,
            self.run_id,
            self.research_iter,
            self.research_dir,
            instruction,
        )

        system_prompt = build_system(
            custom_instructions=self._load_custom_instructions(),
        )
        user_prompt = build_user(instruction)
        tools = get_deep_research_tools()
        state: dict = {
            "research_dir": self.research_dir,
            "scripts_dir": self.scripts_dir,
            "tool_seq": {},
        }
        input_list: list[dict] = [append_message("user", user_prompt)]
        last_input_tokens: int | None = None

        markdown = ""
        step = 0
        while True:
            step += 1
            logger.info("ResearcherAgent step %d", step)

            if should_compact(last_input_tokens):
                input_list = compact_messages(
                    input_list, model=_DEEP_RESEARCH_LLM_MODEL
                )

            response, last_input_tokens = call_llm(
                model=_DEEP_RESEARCH_LLM_MODEL,
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

            if not has_function_calls:
                logger.info("ResearcherAgent completed at step %d", step)
                markdown = response.text or ""
                break

            function_responses = []
            for part in parts:
                if hasattr(part, "function_call") and part.function_call:
                    tool_result_str = _execute_tool_call(part.function_call, state)
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

        plan_path = self.research_dir / f"PLAN_{self.research_iter}.md"
        plan_path.write_text(markdown)
        logger.info("ResearcherAgent wrote %s (%d chars)", plan_path, len(markdown))

        return markdown
