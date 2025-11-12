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
_BASELINE_CODE_TIMEOUT = _RUNTIME_CFG.get("baseline_code_timeout")
_ENSEMBLE_CODE_TIMEOUT = _RUNTIME_CFG.get("ensemble_code_timeout")
_DEFAULT_CODE_TIMEOUT = _BASELINE_CODE_TIMEOUT  # Default to baseline timeout

@weave.op()
def web_search_stack_trace(query: str) -> str:
    """Research how to fix a bug based on the stack trace and error message."""
    logger.info("Dispatching web search for stack trace remediation guidance.")
    trace_index = query.find("Traceback (most recent call last)")
    if trace_index != -1: query = query[trace_index:]
    logger.debug("Stack trace query: %s", query)

    # Step 1: Try fine-tuned model first (no web search capability)
    '''
    logger.info("Attempting fine-tuned model endpoint first...")
    from tools.helpers import call_llm_with_retry_google

    try:
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

    except Exception as e:
        logger.warning(f"Fine-tuned model call failed with error: {e}. Falling back to web search workflow.")'''

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

def _get_sota_tools() -> list:
    """Get tools available for SOTA suggestions (subset of researcher tools)."""
    return [
        {
            "type": "function",
            "name": "ask_eda",
            "description": "Run exploratory data analysis to debug issues or validate hypotheses",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The EDA question to answer (e.g. Read train.csv and compute correlation between XXX and YYY, is the relationship really monotonic to justify adding monotonic constraints?)"}
                },
            },
            "additionalProperties": False,
            "required": ['question']
        },
        {
            "type": "function",
            "name": "scrape_web_page",
            "description": "Scrape web pages for implementation guides, documentation, or examples",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to scrape"}
                },
            },
            "additionalProperties": False,
            "required": ["url"]
        },
        {
            "type": "function",
            "name": "read_research_paper",
            "description": "Read research papers to understand techniques deeply",
            "parameters": {
                "type": "object",
                "properties": {
                    "arxiv_link": {"type": "string", "description": "ArXiv paper ID or link"}
                },
            },
            "additionalProperties": False,
            "required": ["arxiv_link"]
        },
        {
            "type": "function",
            "name": "download_external_datasets",
            "description": "Search and download external datasets to augment training data",
            "parameters": {
                "type": "object",
                "properties": {
                    "question_1": {"type": "string", "description": "First phrasing of the dataset query"},
                    "question_2": {"type": "string", "description": "Second phrasing with different wording"},
                    "question_3": {"type": "string", "description": "Third phrasing using alternative keywords"}
                },
            },
            "additionalProperties": False,
            "required": ["question_1", "question_2", "question_3"]
        },
    ]


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
    attempt_number: int = 1,
    slug: str | None = None,
    data_path: str | None = None,
    cpu_core_range: list[int] | None = None,
    gpu_identifier: str | None = None,
    max_tool_steps: int = 10,
) -> str:
    """Stage 2: Use web search and tools to generate SOTA suggestions based on red flags.

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
        attempt_number: Which attempt this is (1, 2, 3, ...) for interleaving strategy
        slug: Competition slug for download_external_datasets
        data_path: Path to task data directory for ask_eda
        cpu_core_range: CPU cores for ask_eda execution
        gpu_identifier: GPU for ask_eda execution
        max_tool_steps: Maximum tool call iterations before forcing final answer

    Returns:
        SOTA suggestions text with blacklist decision and new suggestion
    """
    logger.info("Dispatching SOTA suggestions (Stage 2) with web search (attempt #%d)", attempt_number)
    executed_suggestion_text = executed_suggestion or "No previous suggestion executed; this is the first attempt."
    failed_ideas_text = "\n".join(f"- {idea}" for idea in failed_ideas) if failed_ideas else "No prior ideas are blacklisted."

    # On even attempts, disable shared suggestions to force independent exploration
    if attempt_number % 2 == 0:
        logger.info("Attempt #%d (even): Disabling shared suggestions to encourage novel exploration", attempt_number)
        shared_suggestions_text = "No shared suggestions provided for this attempt (exploring independently)."
    else:
        shared_suggestions_text = "\n".join(f"- {suggestion}" for suggestion in (shared_suggestions or [])) if shared_suggestions else "No shared suggestions yet."

    # On even attempts, strip "Validated Findings" section from plan to focus on other recommendations
    modified_plan_content = plan_content
    if attempt_number % 2 == 0 and plan_content:
        validated_start = "# Validated Findings (A/B Tested)"
        risks_start = "# Risks & Mitigations"

        start_idx = plan_content.find(validated_start)
        end_idx = plan_content.find(risks_start)

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            logger.info("Attempt #%d (even): Stripping 'Validated Findings' section (%d chars)", attempt_number, end_idx - start_idx)
            modified_plan_content = plan_content[:start_idx] + plan_content[end_idx:]
        else:
            logger.warning("Could not find both section headers to strip Validated Findings")

    # Build plans_section: combine plan.md and later_recommendations
    plans_section = ""
    if modified_plan_content:
        plans_section += f"\n<plan>\n{modified_plan_content}\n</plan>\n"
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

    # Tool execution loop
    tools = _get_sota_tools() if (slug and data_path) else []
    input_list = [{"role": "user", "content": user_prompt}]

    for step in range(max_tool_steps):
        logger.info("SOTA suggestion step %d/%d (tools: %s)", step + 1, max_tool_steps, "enabled" if tools else "disabled")

        response = call_llm_with_retry(
            model=_DEVELOPER_TOOL_MODEL,
            instructions=system_prompt,
            tools=tools,
            messages=input_list,
            web_search_enabled=True,
            text_format=SOTAResponse if step == max_tool_steps - 1 else None,  # Force structured output on last step
        )

        # Add response to conversation
        input_list.extend(response.output)

        # Check if response has tool calls
        has_tool_calls = any(
            hasattr(item, 'type') and item.type == "function_call"
            for item in response.output
        )

        if not has_tool_calls:
            # No tool calls, check if we have structured output
            if hasattr(response, 'output_parsed') and response.output_parsed:
                logger.info("SOTA suggestions completed at step %d (no tool calls, structured output present)", step + 1)
                return response
            else:
                # Need to get structured output - make final call
                logger.info("SOTA suggestions completed at step %d (no tool calls), requesting structured output", step + 1)
                response = call_llm_with_retry(
                    model=_DEVELOPER_TOOL_MODEL,
                    instructions=system_prompt,
                    tools=[],  # No more tools
                    messages=input_list,
                    web_search_enabled=True,
                    text_format=SOTAResponse,
                )
                return response

        # Execute tool calls
        for item in response.output:
            if hasattr(item, 'type') and item.type == "function_call":
                tool_result = _execute_sota_tool_call(
                    item=item,
                    description=description,
                    data_path=data_path,
                    slug=slug,
                    cpu_core_range=cpu_core_range,
                    gpu_identifier=gpu_identifier,
                )
                input_list.append({
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": tool_result
                })

    # Reached max steps, force final structured answer
    logger.warning("SOTA reached max_tool_steps=%d, forcing final answer", max_tool_steps)
    response = call_llm_with_retry(
        model=_DEVELOPER_TOOL_MODEL,
        instructions=system_prompt + "\n\nYou have reached the maximum tool usage limit. Provide your final suggestion now based on the information gathered.",
        tools=[],
        messages=input_list,
        web_search_enabled=True,
        text_format=SOTAResponse,
    )

    return response


def _execute_sota_tool_call(item, description, data_path, slug, cpu_core_range, gpu_identifier):
    """Execute a single SOTA tool call and return JSON result."""
    from tools.researcher import ask_eda, scrape_web_page, read_research_paper, download_external_datasets
    import json

    try:
        args = json.loads(item.arguments)
    except Exception as e:
        logger.error("Failed to parse tool arguments: %s", e)
        return json.dumps({"error": f"Failed to parse arguments: {str(e)}"})

    try:
        if item.name == "ask_eda":
            question = args.get("question", "")
            if not question:
                return json.dumps({"error": "question parameter is required"})

            logger.info("SOTA tool: ask_eda(%s)", question[:100])
            result = ask_eda(
                question=question,
                description=description,
                data_path=data_path,
                previous_ab_tests=[],  # SOTA doesn't use AB test history
                cpu_core_range=cpu_core_range,
                gpu_identifier=gpu_identifier,
            )
            return json.dumps({"result": result})

        elif item.name == "scrape_web_page":
            url = args.get("url", "")
            if not url:
                return json.dumps({"error": "url parameter is required"})

            logger.info("SOTA tool: scrape_web_page(%s)", url)
            result = scrape_web_page(url)
            return json.dumps({"content": result})

        elif item.name == "read_research_paper":
            arxiv_link = args.get("arxiv_link", "")
            if not arxiv_link:
                return json.dumps({"error": "arxiv_link parameter is required"})

            logger.info("SOTA tool: read_research_paper(%s)", arxiv_link)
            result = read_research_paper(arxiv_link)
            return json.dumps({"summary": result})

        elif item.name == "download_external_datasets":
            question_1 = args.get("question_1", "")
            question_2 = args.get("question_2", "")
            question_3 = args.get("question_3", "")

            if not (question_1 and question_2 and question_3):
                return json.dumps({"error": "question_1, question_2, and question_3 are all required"})

            logger.info("SOTA tool: download_external_datasets(%s, %s, %s)",
                       question_1[:50], question_2[:50], question_3[:50])
            result = download_external_datasets(
                question_1=question_1,
                question_2=question_2,
                question_3=question_3,
                slug=slug
            )
            return json.dumps({"result": result})

        else:
            logger.error("Unknown SOTA tool: %s", item.name)
            return json.dumps({"error": f"Unknown tool: {item.name}"})

    except Exception as e:
        logger.exception("SOTA tool execution failed for %s", item.name)
        return json.dumps({"error": f"Tool execution failed: {str(e)}"})


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
