"""Utility helpers for executing generated code with rich logging."""

import json
import logging
import os
import re
import subprocess
import traceback

from dotenv import load_dotenv
from typing import List, Dict, Optional, Any, Union
from openai import OpenAI
from project_config import get_config
from tools.helpers import call_llm_with_retry
from prompts.tools_developer import (
    build_stack_trace_prompt as prompt_stack_trace,
    sota_system as prompt_sota_system,
    sota_user as prompt_sota_user,
    ablation_baseline_prompt,
)
import weave

load_dotenv()

# Configure logging once at import. Downstream callers can override if needed.
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


_CONFIG = get_config()
_LLM_CFG = _CONFIG.get("llm", {}) if isinstance(_CONFIG, dict) else {}
_BASE_URL = _LLM_CFG.get("base_url", "https://openrouter.ai/api/v1")
_API_KEY_ENV = _LLM_CFG.get("api_key_env", "OPENROUTER_API_KEY")
_DEVELOPER_MODEL = _LLM_CFG.get("developer_model", "google/gemini-2.5-pro")
_DEVELOPER_TOOL_MODEL = _LLM_CFG.get("developer_tool_model", _DEVELOPER_MODEL)

client = OpenAI(api_key=os.environ.get(_API_KEY_ENV), base_url=_BASE_URL)

@weave.op()
def web_search_stack_trace(query: str) -> str:
    """Research how to fix a bug based on the stack trace and error message."""
    logger.info("Dispatching web search for stack trace remediation guidance.")
    logger.debug("Stack trace query: %s", query)

    messages = [
        {
            "role": "user",
            "content": prompt_stack_trace(query),
        }
    ]
    logger.debug("Web search messages: %s", messages)

    try:
        content = ""
        while content == "":
            completion = call_llm_with_retry(
                client,
                model=_DEVELOPER_TOOL_MODEL,
                messages=messages,
            )
            try:
                msg = completion.choices[0].message
                content = msg.content or ""
            except Exception:
                msg = ""
                content = ""
            logger.debug("Web search raw response: %s", completion)
        logger.info("Received web search response for stack trace query.")
        logger.debug("Web search raw response: %s", content)
    except Exception:
        logger.exception("Web search request for stack trace failed.")
        return "Please try to fix the bug yourself."

    solution_text = content
    parsed_payload = None
    try:
        parsed_payload = json.loads(content)
    except json.JSONDecodeError:
        logger.debug("Raw response is not bare JSON; attempting fenced-block extraction.")
        try:
            match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL | re.IGNORECASE)
            if match:
                parsed_payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            logger.debug("Failed to parse JSON from fenced block in web search response.")

    if isinstance(parsed_payload, dict):
        solution_candidate = parsed_payload.get("solution")
        if isinstance(solution_candidate, str) and solution_candidate.strip():
            logger.debug("Returning solution field extracted from web search response.")
            return solution_candidate.strip()
        logger.debug("Solution field missing or empty in parsed payload.")

    return solution_text

@weave.op()
def search_sota_suggestions(
    description: str,
    failed_ideas: list[str],
    plans: list[str] | None = None,
    ablation_summary: Union[str, Dict[str, Any], List[Dict[str, Any]], None] = None,
) -> str:
    """Request four categorized SOTA suggestions using minimal context blocks."""
    logger.info("Dispatching SOTA suggestion request (minimal prompt)")
    failed_ideas_text = "No prior ideas are blacklisted."
    if failed_ideas:
        filtered = [idea for idea in failed_ideas if idea][-10:]
        if filtered:
            failed_ideas_text = "\n".join(f"- {idea}" for idea in filtered)

    # Optional: include researcher plans as a separate section for plan-aware suggestions
    plans_section = ""
    if plans:
        blocks: list[str] = []
        for idx, p in enumerate(plans, start=1):
            # Truncate to avoid excessive token usage
            text = p if len(p) <= 30000 else p[-30000:]
            blocks.append(f"<plan id=\"{idx}\">\n{text}\n</plan>")
        plans_section = "\n<researcher_plans>\n" + "\n\n".join(blocks) + "\n</researcher_plans>\n"

    # Normalize ablation summary (dict/list -> compact JSON string)
    ablation_summary_text: Optional[str] = ""
    for summary in ablation_summary:
        summary_text = f"""Suggestion Type: {summary.get("suggestion_type")}
Idea: {summary.get("idea")}
Code Summary: {summary.get("code_summary")}
Logs Summary: {summary.get("logs_summary")}
Score: {summary.get("score")}"""
        ablation_summary_text += summary_text + "\n\n"

    system_prompt = prompt_sota_system()
    prompt = prompt_sota_user(
        description=description,
        plans_section=plans_section,
        failed_ideas_text=failed_ideas_text,
        ablation_summary=ablation_summary_text,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    try:
        content = ""
        while content == "":
            completion = call_llm_with_retry(
                client,
                model=_DEVELOPER_TOOL_MODEL,
                messages=messages,
            )
            try:
                msg = completion.choices[0].message
                logger.debug("SOTA search raw response: %s", completion)
                content = msg.content or ""
            except Exception:
                msg = ""
                content = ""
        logger.debug("SOTA search raw response: %s", content)

        return content

    except Exception:
        logger.exception("SOTA suggestions web search failed")
        return ""


@weave.op()
def ablation_summarize_baseline(code: str, logs: str) -> str:
    """Summarize the baseline run as a concise steer for next steps."""
    logger.info("Summarizing baseline for ablation")
    system_prompt = ablation_baseline_prompt()
    user_prompt = f"""<code>
    {code}
    </code>
    <logs>
    {logs}
    </logs>"""
    try:
        completion = call_llm_with_retry(
            client,
            model=_DEVELOPER_TOOL_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        msg = completion.choices[0].message
        content = (msg.content or "").strip()
    except Exception:
        logger.exception("Baseline ablation summarize failed")
        content = ""

    parsed_payload = None
    try:
        parsed_payload = json.loads(content)
    except json.JSONDecodeError:
        logger.debug("Raw response is not bare JSON; attempting fenced-block extraction.")
        try:
            match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL | re.IGNORECASE)
            if match:
                parsed_payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            logger.debug("Failed to parse JSON from fenced block in baseline summary response.")
            return ""
    return parsed_payload

@weave.op()
def execute_code(filepath: str) -> str:
    """Execute a generated Python file and enrich errors with search guidance."""
    logger.info("Executing generated script: %s", filepath)
    try:
        logger.debug("Running subprocess command: python %s", filepath)
        result = subprocess.run(
            ["python", filepath],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            logger.info("Execution succeeded for %s", filepath)
            logger.debug("Execution stdout: %s", result.stdout)
            return result.stdout

        trace = result.stderr
        logger.warning(
            "Execution failed for %s with return code %s", filepath, result.returncode
        )
        logger.debug("Execution stderr: %s", trace)
        search_result = web_search_stack_trace(trace)
        return trace + "\n" + search_result

    except Exception:
        trace = traceback.format_exc()
        logger.exception("Unexpected error while executing %s", filepath)
        search_result = web_search_stack_trace(trace)
        return trace + "\n" + search_result
