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
    red_flags_system as prompt_red_flags_system,
    red_flags_user as prompt_red_flags_user,
    sota_system as prompt_sota_system,
    sota_user as prompt_sota_user,
)
from utils.oom_detector import detect_oom, OOMError
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
def search_red_flags(
    description: str,
    context: str,
    data_path: str,
    submission_path: str | None = None,
    max_steps: int = 10,
) -> str:
    """Stage 1: Use ask_eda tool-calling to identify red flags in the current approach.

    Args:
        description: Competition description
        context: Current code and logs
        data_path: Path to task/<slug> directory for EDA
        submission_path: Path to current submission file for analysis
        max_steps: Maximum tool-calling iterations (default 10)

    Returns:
        Red flags response text (markdown format with tool summaries + final summary)
    """
    # Import here to avoid circular dependency
    from tools.researcher import ask_eda

    logger.info("Dispatching red flags identification with tool-calling support")

    system_prompt = prompt_red_flags_system()
    user_prompt = prompt_red_flags_user(
        description=description,
        context=context,
    )

    # Add submission file path context if available
    if submission_path:
        user_prompt += f"\n\n<submission_file_path>\n{submission_path}\n</submission_file_path>\n"

    # Get tools (only ask_eda for red flags stage)
    tools = get_tools()

    # Tool-calling loop
    input_list = [{"role": "user", "content": user_prompt}]

    for step in range(max_steps):
        logger.info(f"[Red Flags] Step {step + 1}/{max_steps}")

        if step == max_steps - 1:
            input_list.append({"role": "user", "content": "This is your FINAL step. Output the final red flags summary now!"})
            logger.info("Reached final step; forcing red flags summary output prompt")

        response = call_llm_with_retry(
            model=_DEVELOPER_TOOL_MODEL,
            instructions=system_prompt,
            tools=tools,
            messages=input_list,
            web_search_enabled=False  # No web search in red flags stage
        )

        input_list += response.output
        tool_calls = False

        for item in response.output:
            if item.type == "function_call":
                tool_calls = True
                if item.name == "ask_eda":
                    try:
                        question = json.loads(item.arguments)["question"]
                    except Exception as e:
                        logger.error("Failed to parse ask_eda arguments: %s", e)
                        question = ""
                    logger.info(f"Red flags agent calling ask_eda: {question}")

                    if not question:
                        tool_output = "Error: Missing question parameter"
                    elif not data_path:
                        tool_output = "Error: EDA not available (data_path not provided)"
                    else:
                        tool_output = ask_eda(
                            question=question,
                            description=description,
                            data_path=data_path,
                            timeout_seconds=60, # Shorter timeout for red flags stage
                        )

                        input_list.append({
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": json.dumps({
                                "insights": tool_output,
                            })
                        })

        if tool_calls:
            continue

        # No tool calls -> final response
        final_content = response.output_text or ""
        logger.info(f"Red flags identification completed at step {step + 1}")
        return final_content

    # Max steps reached
    logger.warning("Red flags identification reached max_steps, returning current response")
    return response.output_text or ""

@weave.op()
def search_sota_suggestions(
    description: str,
    context: str,
    red_flags: str,
    executed_suggestion: str | None,
    failed_to_improve_score: bool,
    failed_ideas: list[str],
    executed_code: str | None = None,
    later_recommendations: str | None = None,
) -> str:
    """Stage 2: Use web search to generate SOTA suggestions based on red flags.

    Args:
        description: Competition description
        context: Current code and logs context
        red_flags: Red flags identified from Stage 1 (final summary text)
        executed_suggestion: Most recently executed suggestion
        failed_to_improve_score: Whether the last attempt failed to improve
        failed_ideas: List of blacklisted ideas
        executed_code: Code snippet from last attempt
        later_recommendations: LATER recommendations for progressive improvement

    Returns:
        SOTA suggestions text with blacklist decision and new suggestion
    """
    logger.info("Dispatching SOTA suggestions (Stage 2) with web search")
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
    user_prompt = prompt_sota_user(
        description=description,
        plans_section=suggestions_section,
        red_flags=red_flags,
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
        messages=[{"role": "user", "content": user_prompt}],
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

        # Efficiency: If OOM, skip web search (retry logic will handle it)
        if detect_oom(trace):
            logger.info("OOM detected - skipping web search (will be handled by retry logic)")
            return trace

        # Non-OOM error: do web search for suggestions
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

        # Efficiency: If OOM in exception trace, skip web search
        if detect_oom(trace):
            logger.info("OOM detected in exception - skipping web search")
            return trace

        search_result = web_search_stack_trace(trace)
        return search_result


@weave.op()
def execute_code_with_oom_retry(
    filepath: str,
    timeout_seconds: int = 3600,
    max_oom_retries: int = 10,
    oom_retry_delay: int = 300
) -> tuple[str, float]:
    """
    Execute code with automatic OOM retry logic.

    If OOM is detected, waits and retries the SAME script up to max_oom_retries times.
    If OOM persists after all retries, returns the output (which includes web search
    suggestions for fixing the OOM), allowing the agent to generate corrected code.

    Args:
        filepath: Path to the Python file to execute
        timeout_seconds: Timeout in seconds for each execution attempt
        max_oom_retries: Maximum number of OOM retries (default 10)
        oom_retry_delay: Seconds to wait between OOM retries (default 300 = 5 min)

    Returns:
        Tuple of (output string, total_wait_time_seconds)
        - output: Execution output or error message from execute_code()
        - total_wait_time: Time spent waiting for OOM retries (to exclude from budget)
    """
    total_wait_time = 0.0

    for attempt in range(max_oom_retries + 1):
        # Execute using the existing execute_code function
        output = execute_code(filepath, timeout_seconds)

        # Check if output indicates OOM
        if detect_oom(output):
            if attempt < max_oom_retries:
                logger.warning(
                    f"OOM detected in {filepath} (attempt {attempt + 1}/{max_oom_retries + 1}). "
                    f"Waiting {oom_retry_delay}s before retry..."
                )

                # Wait for GPU VRAM to free up
                wait_start = time.time()
                time.sleep(oom_retry_delay)
                wait_duration = time.time() - wait_start
                total_wait_time += wait_duration

                logger.info(f"Retrying {filepath} after {wait_duration:.1f}s wait...")
                continue  # Retry the same script
            else:
                # Max retries exhausted - return output with web search suggestions
                # The agent will use these suggestions to fix the code
                logger.warning(
                    f"OOM persisted after {max_oom_retries} retries for {filepath}. "
                    f"Returning output with suggestions for agent to fix."
                )
                return output, total_wait_time

        # No OOM detected - return successfully
        if attempt > 0:
            logger.info(f"Execution succeeded for {filepath} after {attempt} OOM retries")

        return output, total_wait_time

    # Defensive return (technically unreachable)
    return output, total_wait_time

def get_tools():
    return [
        {
            "type": "function",
            "name": "ask_eda",
            "description": "Ask a question to the EDA expert",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The question to ask the EDA expert"}
                },
            },
            "additionalProperties": False,
            "required": ['question']
        }
    ]