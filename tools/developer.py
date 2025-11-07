"""Utility helpers for executing generated code with rich logging."""

import json
import logging
import os
import re
import subprocess
import time
import traceback

from dotenv import load_dotenv
from openai import OpenAI
from project_config import get_config
from tools.helpers import call_llm_with_retry
from prompts.tools_developer import (
    build_stack_trace_prompt as prompt_stack_trace,
    build_stack_trace_pseudo_prompt as prompt_stack_trace_pseudo,
    red_flags_system as prompt_red_flags_system,
    red_flags_user as prompt_red_flags_user,
    sota_system as prompt_sota_system,
    sota_user as prompt_sota_user,
)
from schemas.developer import StackTraceSolution, SOTAResponse
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
_FINETUNED_CODE_API_MODEL = _LLM_CFG.get("finetuned_code_api_model")
_RUNTIME_CFG = _CONFIG.get("runtime")
_BASELINE_TIME_LIMIT = _RUNTIME_CFG.get("baseline_time_limit")
_ENSEMBLE_TIME_LIMIT = _RUNTIME_CFG.get("ensemble_time_limit")
_BASELINE_CODE_TIMEOUT = 5400  # 1.5 hours for baseline code execution
_ENSEMBLE_CODE_TIMEOUT = 10800  # 3 hours for ensemble code execution
_DEFAULT_CODE_TIMEOUT = _BASELINE_CODE_TIMEOUT  # Default to baseline timeout

@weave.op()
def web_search_stack_trace(query: str) -> str:
    """Research how to fix a bug based on the stack trace and error message."""
    logger.info("Dispatching web search for stack trace remediation guidance.")
    trace_index = query.find("Traceback (most recent call last)")
    if trace_index != -1: query = query[trace_index:]
    logger.debug("Stack trace query: %s", query)

    # Step 1: Try fine-tuned model first (no web search capability)
    logger.info("Attempting fine-tuned model endpoint first...")
    from tools.helpers import call_llm_with_retry_google

    ft_system_prompt = prompt_stack_trace_pseudo()
    ft_user_prompt = "<query>\n" + query + "\n</query>"

    ft_response = call_llm_with_retry_google(
        model=_FINETUNED_CODE_API_MODEL,
        system_instruction=ft_system_prompt,
        user_prompt=ft_user_prompt,
        text_format=StackTraceSolution,
        temperature=1.0,
        max_retries=3,
        enable_google_search=False,
        top_p=1.0,
        thinking_budget=None,
    )

    # Check if fine-tuned model can answer (consider failure if < 35 chars or contains failure message)
    solution_text = ft_response.reasoning_and_solution.strip() if ft_response else ""
    is_valid_response = (
        ft_response
        and len(solution_text) >= 35
        and "I cannot solve this error." not in solution_text
    )

    if is_valid_response:
        logger.info("Fine-tuned model provided a solution, using it.")
        return query + "\n" + "This is how you can fix the error: \n" + solution_text

    # Step 2: Fallback to current workflow with web search
    logger.info("Fine-tuned model cannot answer (response too short or failure message), falling back to web search workflow.")
    system_prompt = prompt_stack_trace()

    messages = [{"role": "user", "content": "<query>\n" + query + "\n</query>"}]
    logger.debug("Web search messages: %s", messages)

    response = call_llm_with_retry(
        model=_DEVELOPER_TOOL_MODEL,
        instructions=system_prompt,
        tools=[],
        messages=messages,
        web_search_enabled=True,
        text_format=StackTraceSolution,
    )

    # Use structured output
    solution_text = ""
    if response and hasattr(response, 'output_parsed') and response.output_parsed:
        solution_text = response.output_parsed.reasoning_and_solution.strip()
        logger.debug("Returning solution from structured output.")
        return query + "\n" + "This is how you can fix the error: \n" + solution_text

    # Fallback to raw output if structured parsing fails
    content = response.output_text or ""
    logger.warning("Structured output parsing failed, falling back to raw content.")
    return query + "\n" + "This is how you can fix the error: \n" + content

@weave.op()
def search_red_flags(
    description: str,
    context: str,
) -> str:
    """Stage 1: Direct analysis to identify red flags in the current approach.

    Args:
        description: Competition description
        context: Current code and logs

    Returns:
        Red flags response text (markdown format with structured analysis)
    """
    logger.info("Dispatching red flags identification via direct analysis")

    system_prompt = prompt_red_flags_system()
    user_prompt = prompt_red_flags_user(
        description=description,
        context=context,
    )

    # Single-pass analysis with web search enabled
    response = call_llm_with_retry(
        model=_DEVELOPER_TOOL_MODEL,
        instructions=system_prompt,
        tools=[],
        messages=[{"role": "user", "content": user_prompt}],
        web_search_enabled=True
    )

    final_content = response.output_text or ""
    logger.info("Red flags identification completed in single pass")
    return final_content

@weave.op()
def search_sota_suggestions(
    description: str,
    context: str,
    red_flags: str,
    executed_suggestion: str | None,
    failed_ideas: list[str],
    later_recommendations: str | None = None,
    shared_suggestions: list[str] | None = None,
    is_ensemble: bool = False,
    external_data_listing: str | None = None,
    plan_content: str | None = None,
) -> str:
    """Stage 2: Use web search to generate SOTA suggestions based on red flags.

    Args:
        description: Competition description
        context: Current code and logs context
        red_flags: Red flags identified from Stage 1 (final summary text)
        executed_suggestion: Most recently executed suggestion
        failed_ideas: List of blacklisted ideas from this model
        later_recommendations: LATER recommendations for progressive improvement
        shared_suggestions: List of all suggestions from all parallel models with outcomes
                          Format: "Model <model> tried <suggestion> (score improved/worsened/remained by X: A -> B)"
        is_ensemble: If True, uses ensemble-specific prompts and constraints
        external_data_listing: Directory listing of external_data_* folders
        plan_content: Content of plan.md file

    Returns:
        SOTA suggestions text with blacklist decision and new suggestion
    """
    logger.info("Dispatching SOTA suggestions (Stage 2) with web search")
    executed_suggestion_text = executed_suggestion or "No previous suggestion executed; this is the first attempt."
    failed_ideas_text = "\n".join(f"- {idea}" for idea in failed_ideas) if failed_ideas else "No prior ideas are blacklisted."
    shared_suggestions_text = "\n".join(f"- {suggestion}" for suggestion in (shared_suggestions or [])) if shared_suggestions else "No shared suggestions yet."

    # Build plans_section: combine plan.md and later_recommendations
    plans_section = ""
    if plan_content:
        plans_section += f"\n<plan>\n{plan_content}\n</plan>\n"
    if later_recommendations:
        plans_section += f"\n<suggestions>\n{later_recommendations}\n</suggestions>\n"

    system_prompt = prompt_sota_system(is_ensemble=is_ensemble)

    user_prompt = prompt_sota_user(
        description=description,
        plans_section=plans_section,
        red_flags=red_flags,
        failed_ideas_text=failed_ideas_text,
        executed_suggestion_text=executed_suggestion_text,
        context=context,
        shared_suggestions_text=shared_suggestions_text,
        external_data_listing=external_data_listing or "No external data directories found.",
    )

    response = call_llm_with_retry(
        model=_DEVELOPER_TOOL_MODEL,
        instructions=system_prompt,
        tools=[],
        messages=[{"role": "user", "content": user_prompt}],
        web_search_enabled=True,
        text_format=SOTAResponse,
    )

    return response

@weave.op()
def execute_code(filepath: str, timeout_seconds: int | None = None, conda_env: str | None = None) -> str:
    """Execute a generated Python file and enrich errors with search guidance.

    Args:
        filepath: Path to the Python file to execute
        timeout_seconds: Timeout in seconds (default: baseline_time_limit // 4 from config)
        conda_env: Conda environment name to use for execution (None = use current env)

    Returns:
        Execution output or error message
    """
    if timeout_seconds is None:
        timeout_seconds = _DEFAULT_CODE_TIMEOUT
    if conda_env:
        cmd = ["conda", "run", "--no-capture-output", "-n", conda_env, "python", filepath]
        logger.info("Executing in conda environment '%s': %s (timeout: %d seconds)", conda_env, filepath, timeout_seconds)
    else:
        cmd = ["python", filepath]
        logger.info("Executing generated script: %s (timeout: %d seconds)", filepath, timeout_seconds)

    try:
        logger.debug("Running subprocess command: %s", " ".join(cmd))
        result = subprocess.run(
            cmd,
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

        # Do web search for suggestions
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
