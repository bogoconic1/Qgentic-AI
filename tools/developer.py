"""Utility helpers for executing generated code with rich logging."""

import json
import logging
import os
import re
import subprocess
import traceback

from dotenv import load_dotenv
from openai import OpenAI
from project_config import get_config
from tools.helpers import call_llm_with_retry
from prompts.tools_developer import (
    build_stack_trace_prompt as prompt_stack_trace,
    sota_system as prompt_sota_system,
    sota_user as prompt_sota_user,
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
_LLM_CFG = _CONFIG.get("llm")
_DEVELOPER_TOOL_MODEL = _LLM_CFG.get("developer_tool_model")

@weave.op()
def web_search_stack_trace(query: str) -> str:
    """Research how to fix a bug based on the stack trace and error message."""
    logger.info("Dispatching web search for stack trace remediation guidance.")
    trace_index = query.find("Traceback (most recent call last)")
    if trace_index != -1: query = query[trace_index:]
    logger.debug("Stack trace query: %s", query)

    system_prompt = prompt_stack_trace()

    messages = [{"role": "user", "content": "<query>\n" + query + "\n</query>"}]
    logger.debug("Web search messages: %s", messages)

    response = call_llm_with_retry(
        model=_DEVELOPER_TOOL_MODEL,
        instructions=system_prompt,
        tools=[],
        messages=messages,
        web_search_enabled=True
    )

    content = response.output_text or ""

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
            return query + "\n" + "This is how you can fix the error: \n" + solution_candidate.strip()
        logger.debug("Solution field missing or empty in parsed payload.")
    
    return query + "\n" + "This is how you can fix the error: \n" + solution_text

@weave.op()
def search_sota_suggestions(
    description: str,
    context: str,
    executed_suggestion: str | None,
    failed_to_improve_score: bool,
    failed_ideas: list[str],
    executed_code: str | None = None,
    later_recommendations: str | None = None,
) -> str:
    """Use web search to surface potential SOTA improvements for the competition."""
    logger.info("Dispatching SOTA web search")
    failed_ideas_text = "No prior ideas are blacklisted."
    executed_suggestion_text = executed_suggestion or "No previous suggestion executed; this is the first attempt."
    executed_code_text = executed_code or "No explicit code snippet was provided for the last attempt."
    failed_ideas_text = "\n".join(f"- {idea}" for idea in failed_ideas)

    # Include LATER recommendations as context for more advanced suggestions
    suggestions_section = ""
    if later_recommendations:
        suggestions_section = f"\n<suggestions>\n{later_recommendations}\n</suggestions>\n"

    system_prompt = prompt_sota_system()

    outcome_status = "No improvement" if failed_to_improve_score else "Improved or matched"
    prompt = prompt_sota_user(
        description=description,
        plans_section=suggestions_section,
        failed_ideas_text=failed_ideas_text,
        executed_suggestion_text=executed_suggestion_text,
        executed_code_text=executed_code_text,
        context=context,
        outcome_status=outcome_status,
    )

    response = call_llm_with_retry(
        model=_DEVELOPER_TOOL_MODEL,
        instructions=system_prompt,
        tools=[],
        messages=[{"role": "user", "content": prompt}],
        web_search_enabled=True
    )

    return response.output_text or ""

@weave.op()
def execute_code(filepath: str, timeout_seconds: int = 3600) -> str:
    """Execute a generated Python file and enrich errors with search guidance.

    Args:
        filepath: Path to the Python file to execute
        timeout_seconds: Timeout in seconds (default 3600 = 1 hour)

    Returns:
        Execution output or error message
    """
    logger.info("Executing generated script: %s (timeout: %d seconds)", filepath, timeout_seconds)
    try:
        logger.debug("Running subprocess command: python %s", filepath)
        result = subprocess.run(
            ["python", filepath],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
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
        return search_result

    except subprocess.TimeoutExpired:
        logger.error("Execution timed out after %d seconds for %s", timeout_seconds, filepath)
        timeout_minutes = timeout_seconds / 60
        if timeout_minutes >= 60:
            timeout_hours = timeout_minutes / 60
            return f"Code execution timed out after {timeout_hours:.1f} hour(s)"
        elif timeout_minutes >= 1:
            return f"Code execution timed out after {timeout_minutes:.0f} minute(s)"
        else:
            return f"Code execution timed out after {timeout_seconds} second(s)"
    except Exception:
        trace = traceback.format_exc()
        logger.exception("Unexpected error while executing %s", filepath)
        search_result = web_search_stack_trace(trace)
        return search_result