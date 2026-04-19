"""Deep Research sub-agent.

A sub-agent that the main (or orchestrator) agent can call with a research
instruction. It runs a multi-step tool loop over three tools — `web_research`
(Exa discovery), `web_fetch` (Firecrawl scrape), and `write_python_code`
(subprocess exec) — and emits a markdown report. Gemini's built-in
`google_search` is disabled inside the sub-agent so that every URL the LLM
dereferences is traceable back to a prior tool result (no invented URLs).

No truncation is applied to `web_research` / `web_fetch` outputs — full content
flows back to the LLM so the research can go as deep as the step budget allows.

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

from project_config import get_config, get_instructions
from prompts.research import build_system, build_user
from tools.developer import _build_resource_header, execute_code
from tools.helpers import call_llm
from utils.llm_utils import append_message, get_deep_research_tools


load_dotenv()

logger = logging.getLogger(__name__)


_CONFIG = get_config()
_LLM_CFG = _CONFIG["llm"]
_RUNTIME_CFG = _CONFIG["runtime"]
_PATH_CFG = _CONFIG["paths"]

_TASK_ROOT = Path(_PATH_CFG["task_root"])
_DEEP_RESEARCH_LLM_MODEL = _LLM_CFG["developer_tool_model"]
_DEEP_RESEARCH_MAX_STEPS = _RUNTIME_CFG["deep_research_max_steps"]
_WRITE_PYTHON_CODE_TIMEOUT_SECONDS = _RUNTIME_CFG["write_python_code_timeout_seconds"]
_HITL_INSTRUCTIONS = get_instructions()["# Researcher Instructions"]


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

    return json.dumps({"output": output})


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
        raise ValueError(f"Unknown tool: {tool_name}")

    if tool_name in ("web_research", "web_fetch"):
        record_path = state["research_dir"] / tool_name / f"{tool_seq}.md"
        record_path.write_text(
            _render_tool_record_markdown(tool_name, tool_seq, args, result_json)
        )

    return result_json


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@weave.op()
def deep_research(instruction: str, slug: str, run_id: str, research_iter: int) -> str:
    """Run the Deep Research sub-agent and return a markdown report.

    Creates `task/<slug>/<run_id>/research_<research_iter>/`, runs the tool
    loop, and persists the final markdown as `PLAN_<research_iter>.md` inside
    that directory. The markdown is also returned to the caller.

    Args:
        instruction: Free-form research instruction — as long as needed.
        slug: Competition slug (maps to `task/<slug>/`).
        run_id: Orchestrator run identifier (timestamp dir under task/<slug>/).
        research_iter: Per-caller invocation counter — starts at 1, increments
            each time the caller invokes deep_research for the same run.

    Returns:
        Free-form markdown report with URL citations.
    """
    research_dir = _TASK_ROOT / slug / run_id / f"research_{research_iter}"
    scripts_dir = research_dir / "scripts"
    web_research_dir = research_dir / "web_research"
    web_fetch_dir = research_dir / "web_fetch"
    for d in (scripts_dir, web_research_dir, web_fetch_dir):
        d.mkdir(parents=True, exist_ok=True)

    logger.info(
        "deep_research slug=%s run_id=%s iter=%d dir=%s instruction=%r",
        slug,
        run_id,
        research_iter,
        research_dir,
        instruction,
    )

    system_prompt = build_system(hitl_instructions=_HITL_INSTRUCTIONS)
    user_prompt = build_user(instruction)
    tools = get_deep_research_tools()
    state: dict = {
        "research_dir": research_dir,
        "scripts_dir": scripts_dir,
        "tool_seq": {},
    }
    input_list = [append_message("user", user_prompt)]

    markdown = ""
    for step in range(_DEEP_RESEARCH_MAX_STEPS):
        is_last_step = step == _DEEP_RESEARCH_MAX_STEPS - 1
        logger.info("deep_research step %d/%d", step + 1, _DEEP_RESEARCH_MAX_STEPS)

        response = call_llm(
            model=_DEEP_RESEARCH_LLM_MODEL,
            system_instruction=system_prompt,
            function_declarations=tools if not is_last_step else [],
            messages=input_list,
            enable_google_search=False,
        )

        parts = response.candidates[0].content.parts
        has_function_calls = any(
            part.function_call
            for part in parts
            if hasattr(part, "function_call")
        )

        if not has_function_calls:
            logger.info("deep_research completed at step %d", step + 1)
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

        input_list.append(response.candidates[0].content)
        if function_responses:
            input_list.append(
                types.Content(role="function", parts=function_responses)
            )
    else:
        logger.warning(
            "deep_research exhausted %d steps — forcing final report", _DEEP_RESEARCH_MAX_STEPS
        )
        response = call_llm(
            model=_DEEP_RESEARCH_LLM_MODEL,
            system_instruction=system_prompt
            + "\n\nReturn your final markdown report now, based on everything you have gathered.",
            messages=input_list,
            enable_google_search=False,
        )
        markdown = response.text or ""

    plan_path = research_dir / f"PLAN_{research_iter}.md"
    plan_path.write_text(markdown)
    logger.info("deep_research wrote %s (%d chars)", plan_path, len(markdown))

    return markdown
