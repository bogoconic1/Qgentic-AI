"""Utility helpers for executing generated code with rich logging."""

import base64
import json
import logging
import os
import re
import subprocess
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from project_config import get_config
from tools.helpers import call_llm_with_retry, call_llm_with_retry_anthropic, call_llm_with_retry_google
from utils.llm_utils import detect_provider, extract_text_from_response, append_message
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
_BASELINE_CODE_TIMEOUT = _RUNTIME_CFG.get("baseline_code_timeout")
_PATH_CFG = _CONFIG.get("paths")
_OUTPUTS_DIRNAME = _PATH_CFG.get("outputs_dirname")


def _build_resource_header(cpu_core_range: list[int] | None, gpu_identifier: str | None) -> str:
    """Build a Python header that sets CPU affinity and GPU assignment."""
    lines = ["import os\n"]
    if cpu_core_range:
        lines.append("import psutil")
        lines.append(f"psutil.Process().cpu_affinity({cpu_core_range})\n")
    if gpu_identifier is not None:
        lines.append(f'os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu_identifier}"')
    lines.append('os.environ["OPENBLAS_NUM_THREADS"] = "32"\n')
    return "\n".join(lines) + "\n"

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
        logger.warning(f"Fine-tuned model call failed with error: {e}. Falling back to web search workflow.")

    # Step 2: Fallback to current workflow with web search
    logger.info("Fine-tuned model cannot answer (response too short or failure message), falling back to web search workflow.")
    system_prompt = prompt_stack_trace()

    # Detect provider and create messages in provider-specific format
    provider = detect_provider(_DEVELOPER_TOOL_MODEL)
    messages = [append_message(provider, "user", "<query>\n" + query + "\n</query>")]
    logger.debug("Web search messages: %s", messages)

    if provider == "openai":
        response = call_llm_with_retry(
            model=_DEVELOPER_TOOL_MODEL,
            instructions=system_prompt,
            tools=[],
            messages=messages,
            web_search_enabled=True,
            text_format=StackTraceSolution,
        )

        # Response is already parsed Pydantic object
        solution_text = ""
        if response and hasattr(response, 'reasoning_and_solution'):
            solution_text = response.reasoning_and_solution.strip()
            logger.debug("Returning solution from OpenAI structured output.")
            return query + "\n" + "This is how you can fix the error: \n" + solution_text

        # Fallback to raw output if structured parsing fails
        content = response.output_text or ""
        logger.warning("Structured output parsing failed, falling back to raw content.")
        return query + "\n" + "This is how you can fix the error: \n" + content

    elif provider == "anthropic":
        response = call_llm_with_retry_anthropic(
            model=_DEVELOPER_TOOL_MODEL,
            instructions=system_prompt,
            tools=[],
            messages=messages,
            web_search_enabled=True,
            text_format=StackTraceSolution,
        )

        # Response is already parsed Pydantic object
        if response and hasattr(response, 'reasoning_and_solution'):
            solution_text = response.reasoning_and_solution.strip()
            logger.debug("Returning solution from structured output.")
            return query + "\n" + "This is how you can fix the error: \n" + solution_text

        # Fallback if not a Pydantic object (shouldn't happen with text_format)
        logger.warning("Unexpected response type, attempting to extract text.")
        content = extract_text_from_response(response, provider)
        return query + "\n" + "This is how you can fix the error: \n" + content

    elif provider == "google":
        response = call_llm_with_retry_google(
            model=_DEVELOPER_TOOL_MODEL,
            system_instruction=system_prompt,
            messages=messages,
            enable_google_search=True,
            text_format=StackTraceSolution,
        )

        # Response is already parsed Pydantic object
        if response and hasattr(response, 'reasoning_and_solution'):
            solution_text = response.reasoning_and_solution.strip()
            logger.debug("Returning solution from Gemini structured output.")
            return query + "\n" + "This is how you can fix the error: \n" + solution_text

        # Fallback if not a Pydantic object (shouldn't happen with text_format)
        logger.warning("Unexpected response type, attempting to extract text.")
        content = extract_text_from_response(response, provider)
        return query + "\n" + "This is how you can fix the error: \n" + content

    else:
        raise ValueError(f"Unsupported provider: {provider}")

def _ingest_images_for_llm(images: list[Path], provider: str) -> list[dict] | None:
    """Encode and format images for LLM messages (shared helper for red flags and SOTA tools).

    Args:
        images: List of image paths to ingest
        provider: Provider type ("openai", "anthropic", or "google")

    Returns:
        List of formatted image messages ready to append, or None if no valid images
    """
    from utils.llm_utils import encode_image_to_data_url

    image_content = []
    for img_path in images:
        if not img_path.exists():
            logger.warning(f"Image not found: {img_path}")
            continue

        # Encode image to data URL (with compression for Anthropic's 5MB limit)
        resize_flag = (provider == "anthropic")
        data_url = encode_image_to_data_url(str(img_path), resize_for_anthropic=resize_flag)
        if not data_url:
            logger.warning(f"Failed to encode image: {img_path}")
            continue

        # Format based on provider
        if provider == "anthropic":
            if ";base64," in data_url:
                mime_part, b64_data = data_url.split(";base64,", 1)
                mime_type = mime_part.replace("data:", "")
                image_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": b64_data
                    }
                })
        elif provider == "google":
            from google.genai import types
            if ";base64," in data_url:
                mime_part, b64_data = data_url.split(";base64,", 1)
                mime_type = mime_part.replace("data:", "")
                image_content.append(
                    types.Part(
                        inline_data=types.Blob(
                            mime_type=mime_type,
                            data=base64.b64decode(b64_data)
                        )
                    )
                )
        else:  # OpenAI
            image_content.append({"type": "input_image", "image_url": data_url})

    if not image_content:
        return None

    # Return in provider-specific format
    if provider == "google":
        return [{"role": "user", "parts": image_content}]
    else:
        return [{"role": "user", "content": image_content}]


@weave.op()
def search_red_flags(
    description: str,
    context: str,
    images: list[Path] | None = None,
    train_stats: dict | None = None,
) -> str:
    """Stage 1: Direct analysis to identify red flags in the current approach.

    Args:
        description: Competition description
        context: Current code and logs
        images: Optional list of image paths (e.g., loss_curve.png, metric_curve.png)
        train_stats: Optional training statistics dict from train_stats.json
                    (must include: model_name, cv_scores, cv_mean, cv_std)

    Returns:
        Red flags response text (markdown format with structured analysis)
    """
    logger.info("Dispatching red flags identification via direct analysis")

    # Include train_stats in context if provided
    context_with_stats = context
    if train_stats:
        stats_text = "## Training Statistics (from train_stats.json)\n\n"
        stats_text += f"```json\n{json.dumps(train_stats, indent=2)}\n```\n\n"
        context_with_stats = stats_text + context
        logger.info("Added train_stats to context (%d keys)", len(train_stats))

    system_prompt = prompt_red_flags_system()
    user_prompt = prompt_red_flags_user(
        description=description,
        context=context_with_stats,
    )

    # Detect provider and call appropriate API
    provider = detect_provider(_DEVELOPER_TOOL_MODEL)

    # Start with text message
    messages = [append_message(provider, "user", user_prompt)]

    # Add images if provided
    if images:
        image_messages = _ingest_images_for_llm(images, provider)
        if image_messages:
            messages.extend(image_messages)

    if provider == "openai":
        response = call_llm_with_retry(
            model=_DEVELOPER_TOOL_MODEL,
            instructions=system_prompt,
            tools=[],
            messages=messages,
            web_search_enabled=True
        )
        final_content = response.output_text or ""
    elif provider == "anthropic":
        response = call_llm_with_retry_anthropic(
            model=_DEVELOPER_TOOL_MODEL,
            instructions=system_prompt,
            tools=[],
            messages=messages,
            web_search_enabled=True
        )
        final_content = extract_text_from_response(response, provider)
    elif provider == "google":
        response = call_llm_with_retry_google(
            model=_DEVELOPER_TOOL_MODEL,
            system_instruction=system_prompt,
            function_declarations=[],
            messages=messages,
            enable_google_search=True
        )
        final_content = extract_text_from_response(response, provider)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    logger.info("Red flags identification completed in single pass")
    return final_content

def _get_sota_tools(provider: str = "openai") -> list:
    """Get tools available for SOTA suggestions.

    Args:
        provider: "openai", "anthropic", or "google" for format conversion

    Returns:
        List of tool definitions in provider-specific format
    """
    from utils.llm_utils import get_tools_for_provider
    return get_tools_for_provider(provider)


def _get_tool_name(tool, provider: str) -> str:
    """Extract tool name from tool definition based on provider format.

    Args:
        tool: Tool definition dict or FunctionDeclaration
        provider: Provider type

    Returns:
        Tool name string
    """
    if provider == "openai":
        return tool.get("name", "")
    elif provider == "anthropic":
        return tool.get("name", "")
    elif provider == "google":
        # Gemini uses FunctionDeclaration objects
        return tool.name if hasattr(tool, 'name') else ""
    else:
        return ""


@weave.op()
def search_sota_suggestions(
    description: str,
    context: str,
    red_flags: str,
    executed_suggestion: str | None,
    failed_ideas: list[str],
    later_recommendations: str | None = None,
    shared_suggestions: list[str] | None = None,
    external_data_listing: str | None = None,
    plan_content: str | None = None,
    attempt_number: int = 1,
    slug: str | None = None,
    data_path: str | None = None,
    cpu_core_range: list[int] | None = None,
    gpu_identifier: str | None = None,
    file_suffix: str | None = None,
    max_tool_steps: int = 20,
    version: int = 1,
    images: list[Path] | None = None,
    train_stats: dict | None = None,
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
        external_data_listing: Directory listing of external_data_* folders
        plan_content: Content of plan.md file
        attempt_number: Which attempt this is (1, 2, 3, ...) for interleaving strategy
        slug: Competition slug
        data_path: Path to task data directory
        cpu_core_range: CPU cores for execute_python execution
        gpu_identifier: GPU for execute_python execution
        file_suffix: Suffix for temporary files to prevent race conditions (e.g., "1_1", "1_2")
        max_tool_steps: Maximum tool call iterations before forcing final answer
        images: Optional list of image paths (e.g., loss_curve.png, metric_curve.png)
        train_stats: Optional training statistics dict from train_stats.json
                    (must include: model_name, cv_scores, cv_mean, cv_std)

    Returns:
        Parsed SOTAResponse object with suggestion, blacklist, blacklist_reason, and suggestion_code fields.
        Returns None if parsing fails.
    """
    logger.info("Dispatching SOTA suggestions (Stage 2) with web search (attempt #%d)", attempt_number)

    # Include train_stats in context if provided
    context_with_stats = context
    if train_stats:
        stats_text = "## Training Statistics (from train_stats.json)\n\n"
        stats_text += f"```json\n{json.dumps(train_stats, indent=2)}\n```\n\n"
        context_with_stats = stats_text + context
        logger.info("Added train_stats to context (%d keys)", len(train_stats))
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

    # Convert time limit from seconds to minutes for prompt
    time_limit_minutes = int(_BASELINE_CODE_TIMEOUT / 60)

    system_prompt = prompt_sota_system(time_limit_minutes=time_limit_minutes)

    user_prompt = prompt_sota_user(
        description=description,
        plans_section=plans_section,
        red_flags=red_flags,
        failed_ideas_text=failed_ideas_text,
        executed_suggestion_text=executed_suggestion_text,
        context=context_with_stats,
        shared_suggestions_text=shared_suggestions_text,
        external_data_listing=external_data_listing or "No external data directories found.",
    )

    # Tool execution loop
    provider = detect_provider(_DEVELOPER_TOOL_MODEL)
    tools = _get_sota_tools(provider) if (slug and data_path) else []
    input_list = [append_message(provider, "user", user_prompt)]

    # Add images if provided
    if images:
        image_messages = _ingest_images_for_llm(images, provider)
        if image_messages:
            input_list.extend(image_messages)

    for step in range(max_tool_steps):
        logger.info("SOTA suggestion step %d/%d (tools: %s, provider: %s)", step + 1, max_tool_steps, "enabled" if tools else "disabled", provider)

        if provider == "openai":
            response = call_llm_with_retry(
                model=_DEVELOPER_TOOL_MODEL,
                instructions=system_prompt,
                tools=tools,
                messages=input_list,
                web_search_enabled=True,
                text_format=SOTAResponse if step == max_tool_steps - 1 else None,
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
                if response and hasattr(response, 'suggestion'):
                    logger.info("SOTA suggestions completed at step %d (no tool calls, structured output present)", step + 1)
                    return response  # Already parsed Pydantic object
                else:
                    # Need to get structured output - make final call
                    logger.info("SOTA suggestions completed at step %d (no tool calls), requesting structured output", step + 1)
                    response = call_llm_with_retry(
                        model=_DEVELOPER_TOOL_MODEL,
                        instructions=system_prompt,
                        tools=[],
                        messages=input_list,
                        web_search_enabled=True,
                        text_format=SOTAResponse,
                    )
                    return response  # Already parsed Pydantic object

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
                        file_suffix=file_suffix,
                        step=step,
                        version=version,
                        provider=provider,
                    )
                    input_list.append({
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": tool_result
                    })

        elif provider == "anthropic":
            response = call_llm_with_retry_anthropic(
                model=_DEVELOPER_TOOL_MODEL,
                instructions=system_prompt,
                tools=tools,
                messages=input_list,
                web_search_enabled=True,
                text_format=SOTAResponse if step == max_tool_steps - 1 else None,
            )

            # Check stop_reason
            if response.stop_reason == "tool_use":
                # Extract tool_use blocks
                tool_uses = [
                    block for block in response.content
                    if hasattr(block, 'type') and block.type == 'tool_use'
                ]

                # Execute all tools and collect results
                tool_results = []
                for tool_use in tool_uses:
                    tool_result_str = _execute_sota_tool_call(
                        item=tool_use,
                        description=description,
                        data_path=data_path,
                        slug=slug,
                        cpu_core_range=cpu_core_range,
                        gpu_identifier=gpu_identifier,
                        file_suffix=file_suffix,
                        step=step,
                        version=version,
                        provider=provider,
                    )
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": tool_result_str
                    })

                # Add to messages in Anthropic format
                input_list.append({
                    "role": "assistant",
                    "content": response.content
                })
                input_list.append({
                    "role": "user",
                    "content": tool_results
                })

            else:
                # No tool use, check if we have structured output
                if response and hasattr(response, 'suggestion'):
                    logger.info("SOTA suggestions completed at step %d (no tool use, structured output present)", step + 1)
                    return response  # Already parsed Pydantic object
                else:
                    # Need to get structured output - make final call
                    logger.info("SOTA suggestions completed at step %d (no tool use), requesting structured output", step + 1)
                    response = call_llm_with_retry_anthropic(
                        model=_DEVELOPER_TOOL_MODEL,
                        instructions=system_prompt,
                        tools=[],
                        messages=input_list,
                        web_search_enabled=True,
                        text_format=SOTAResponse,
                    )
                    return response  # Already parsed Pydantic object

        elif provider == "google":
            from google.genai import types

            response = call_llm_with_retry_google(
                model=_DEVELOPER_TOOL_MODEL,
                system_instruction=system_prompt,
                function_declarations=tools if tools else [],
                messages=input_list,
                enable_google_search=True,
                text_format=SOTAResponse if step == max_tool_steps - 1 else None,
            )

            # Check if response has function calls
            has_function_calls = False
            if hasattr(response, 'candidates') and response.candidates:
                parts = response.candidates[0].content.parts
                has_function_calls = any(
                    hasattr(part, 'function_call') and part.function_call
                    for part in parts
                )

            if not has_function_calls:
                # No function calls, check if we have structured output
                if response and hasattr(response, 'suggestion'):
                    logger.info("SOTA suggestions completed at step %d (no function calls, structured output present)", step + 1)
                    return response  # Already parsed Pydantic object
                else:
                    # Need to get structured output - make final call
                    logger.info("SOTA suggestions completed at step %d (no function calls), requesting structured output", step + 1)
                    response = call_llm_with_retry_google(
                        model=_DEVELOPER_TOOL_MODEL,
                        system_instruction=system_prompt,
                        messages=input_list,
                        enable_google_search=True,
                        text_format=SOTAResponse,
                    )
                    return response  # Already parsed Pydantic object

            # Execute function calls
            parts = response.candidates[0].content.parts
            function_responses = []

            for part in parts:
                if hasattr(part, 'function_call') and part.function_call:
                    tool_result_str = _execute_sota_tool_call(
                        item=part.function_call,
                        description=description,
                        data_path=data_path,
                        slug=slug,
                        cpu_core_range=cpu_core_range,
                        gpu_identifier=gpu_identifier,
                        file_suffix=file_suffix,
                        step=step,
                        version=version,
                        provider=provider,
                    )
                    function_responses.append(
                        types.Part.from_function_response(
                            name=part.function_call.name,
                            response={"result": tool_result_str}
                        )
                    )

            # Add assistant message and function results
            input_list.append(append_message(provider, "assistant", response.text if hasattr(response, 'text') else ""))
            if function_responses:
                input_list.append(types.Content(role="function", parts=function_responses))

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    # Reached max steps, force final structured answer
    logger.warning("SOTA reached max_tool_steps=%d, forcing final answer", max_tool_steps)

    if provider == "openai":
        response = call_llm_with_retry(
            model=_DEVELOPER_TOOL_MODEL,
            instructions=system_prompt + "\n\nYou have reached the maximum tool usage limit. Provide your final suggestion now based on the information gathered.",
            tools=[],
            messages=input_list,
            web_search_enabled=True,
            text_format=SOTAResponse,
        )
        return response  # Already parsed Pydantic object
    elif provider == "anthropic":
        response = call_llm_with_retry_anthropic(
            model=_DEVELOPER_TOOL_MODEL,
            instructions=system_prompt + "\n\nYou have reached the maximum tool usage limit. Provide your final suggestion now based on the information gathered.",
            tools=[],
            messages=input_list,
            web_search_enabled=True,
            text_format=SOTAResponse,
        )
        return response  # Already parsed Pydantic object
    elif provider == "google":
        response = call_llm_with_retry_google(
            model=_DEVELOPER_TOOL_MODEL,
            system_instruction=system_prompt + "\n\nYou have reached the maximum tool usage limit. Provide your final suggestion now based on the information gathered.",
            messages=input_list,
            enable_google_search=True,
            text_format=SOTAResponse,
        )
        return response  # Already parsed Pydantic object
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def _execute_sota_tool_call(item, description, data_path, slug, cpu_core_range, gpu_identifier, file_suffix, step=0, version=1, provider="openai"):
    """Execute a single SOTA tool call and return JSON result.

    Args:
        item: Tool call item (OpenAI, Anthropic, or Google format)
        step: Current tool loop step (for unique filenames)
        version: Current developer version (for output directory)
        provider: "openai", "anthropic", or "google"
    """
    from tools.researcher import scrape_web_page, read_research_paper
    import json

    # Parse tool name and arguments based on provider
    if provider == "openai":
        tool_name = item.name
        try:
            args = json.loads(item.arguments)
        except Exception as e:
            logger.error("Failed to parse tool arguments: %s", e)
            return json.dumps({"error": f"Failed to parse arguments: {str(e)}"})
    elif provider == "anthropic":
        tool_name = item.name
        args = item.input  # Already a dict
    elif provider == "google":
        tool_name = item.name
        # Gemini function_call.args is already a dict
        args = dict(item.args) if hasattr(item, 'args') else {}
    else:
        return json.dumps({"error": f"Unsupported provider: {provider}"})

    try:
        if item.name == "execute_python":
            code = args.get("code", "")
            if not code:
                return json.dumps({"error": "code parameter is required"})

            logger.info("SOTA tool: execute_python (code_len=%d, step=%d)", len(code), step)

            # Save to outputs/{iteration}/{version}/execute_python_{step}.py
            version_dir = Path(data_path) / _OUTPUTS_DIRNAME / file_suffix / str(version)
            version_dir.mkdir(parents=True, exist_ok=True)
            script_file = version_dir / f"execute_python_{step}.py"

            resource_header = _build_resource_header(cpu_core_range, gpu_identifier)
            script_file.write_text(resource_header + code)

            output = execute_code(str(script_file), timeout_seconds=300)
            return json.dumps({"output": output})

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

    # Get conda executable path (resolves "conda: command not found" in subprocess)
    conda_exe = os.environ.get('CONDA_EXE', 'conda')

    if conda_env:
        cmd = [conda_exe, "run", "--no-capture-output", "-n", conda_env, "python", filepath]
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
