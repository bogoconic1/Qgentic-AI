import logging
import os
import time
from openai import OpenAI
import openai
import weave

from project_config import get_config

def _build_directory_listing(root: str, num_files: int | None = None) -> str:
    cfg = get_config()
    runtime_cfg = cfg.get("runtime")
    path_cfg = cfg.get("paths")
    limit = num_files if num_files is not None else runtime_cfg.get("directory_listing_max_files")
    lines: list[str] = []
    outputs_dirname = path_cfg.get("outputs_dirname")

    for current_root, dirs, files in os.walk(root):
        rel_root = os.path.relpath(current_root, root)

        # Determine traversal policy for the outputs tree: allow only external_data_*/ contents
        segments = [] if rel_root in (".", "") else rel_root.split(os.sep)
        files_to_show = files

        if segments and segments[0] == outputs_dirname:
            # At <root>/outputs
            if len(segments) == 1:
                # Only descend into iteration directories that are numeric; hide files at this level
                dirs[:] = sorted([d for d in dirs if d.isdigit()])
                files_to_show = []
            # At <root>/outputs/<iteration>
            elif len(segments) == 2:
                # Only descend into external_data_* subdirectories; hide files at this level
                dirs[:] = sorted([d for d in dirs if d.startswith("external_data_")])
                files_to_show = []
            else:
                # At or below <root>/outputs/<iteration>/*
                # Allow full traversal within external_data_*; block others entirely
                if len(segments) >= 3 and segments[2].startswith("external_data_"):
                    dirs[:] = sorted(dirs)
                    files_to_show = files
                else:
                    dirs[:] = []
                    files_to_show = []
        else:
            # Outside outputs subtree: normal traversal
            dirs[:] = sorted(dirs)
            files_to_show = files

        depth = 0 if rel_root in (".", "") else rel_root.count(os.sep) + 1
        indent = "    " * depth
        folder_display = "." if rel_root in (".", "") else os.path.basename(rel_root)
        lines.append(f"{indent}{folder_display}/")

        # Print files according to policy and limit
        for name in files_to_show[:limit]:  # to avoid stuffing context window
            lines.append(f"{indent}    {name}")
        if len(files_to_show) > limit:
            lines.append(f"{indent}    ... ({len(files_to_show) - limit} more files)")

    return "\n".join(lines)

@weave.op()
def call_llm_with_retry_helper(
    model: str,
    instructions: str,
    tools: list,
    messages: list,
    max_retries: int | None = None,
    web_search_enabled: bool = False,
    text_format = None,
):

    client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
    cfg = get_config()
    runtime_cfg = cfg.get("runtime")
    retries = max_retries or runtime_cfg.get("llm_max_retries")
    if retries < 1: retries = 1
    if web_search_enabled: tools.append({"type": "web_search"})

    for attempt in range(retries):
        try:
            if text_format is not None:
                # Use structured outputs with Pydantic model
                response = client.responses.parse(
                    model=model,
                    instructions=instructions,
                    tools=tools,
                    input=messages,
                    text_format=text_format,
                )
            else:
                # Use regular responses API
                response = client.responses.create(
                    model=model,
                    instructions=instructions,
                    tools=tools,
                    input=messages,
                )
            return response
        except openai.InternalServerError as e:
            if "504" in str(e) and attempt < retries - 1:
                wait_time = 2**attempt
                logging.info(f"Retry {attempt + 1}/{retries} in {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                logging.error(f"LLM call failed with error: {e}")
                continue
        except Exception as e:
            logging.error(f"Error calling LLM: {e}")
            continue
    return None

@weave.op()
def call_llm_with_retry(
    model: str,
    instructions: str,
    tools: list,
    messages: list,
    max_retries: int | None = None,
    web_search_enabled: bool = False,
    text_format = None,
):
    result = None

    for _ in range(40):
        result = call_llm_with_retry_helper(
            model=model,
            instructions=instructions,
            tools=tools,
            messages=messages,
            max_retries=max_retries,
            web_search_enabled=web_search_enabled,
            text_format=text_format,
        )
        if result is not None:
            break

    if result is None:
        raise ValueError("LLM call failed after 40 retries") # most likely severe issues like token limit exceeded, should not continue

    return result

@weave.op()
def call_llm_with_retry_google_helper(
    model: str,
    system_instruction: str,
    user_prompt: str,
    text_format = None,
    temperature: float = 1.0,
    max_retries: int | None = None,
    enable_google_search: bool = False,
    top_p: float = 1.0,
    thinking_budget: int | None = None,
):
    """Helper function to call Gemini API with optional structured outputs and retry logic.

    Args:
        model: Gemini model name (e.g., "gemini-2.5-pro")
        system_instruction: System instruction text
        user_prompt: User prompt text
        text_format: Optional Pydantic model for structured output schema. If None, returns raw text.
        temperature: Temperature for generation
        max_retries: Maximum number of retries
        enable_google_search: Enable Google Search tool
        top_p: Nucleus sampling parameter (default: None)
        thinking_budget: Thinking budget for reasoning models (default: None)

    Returns:
        Parsed Pydantic model instance (if text_format provided) or raw text response, or None on failure
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        logging.error(f"Failed to import google.genai: {e}")
        return None

    if model.startswith("gemini-"):
        os.environ['GOOGLE_CLOUD_LOCATION'] = 'global'

    cfg = get_config()
    runtime_cfg = cfg.get("runtime")
    retries = max_retries or runtime_cfg.get("llm_max_retries", 3)
    if retries < 1:
        retries = 1

    for attempt in range(retries):
        try:
            client = genai.Client()

            # Build tools list
            tools = []
            if enable_google_search:
                tools.append(types.Tool(googleSearch=types.GoogleSearch()))

            # Build config
            config_params = {
                "temperature": temperature,
                "top_p": top_p,
                "system_instruction": [types.Part.from_text(text=system_instruction)],
                "tools": tools if tools else None,
            }

            if thinking_budget is not None:
                config_params["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)

            # Add structured output params if text_format provided
            if text_format is not None:
                config_params["response_mime_type"] = "application/json"
                config_params["response_json_schema"] = text_format.model_json_schema()

            response = client.models.generate_content(
                model=model,
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=user_prompt)],
                    )
                ],
                config=types.GenerateContentConfig(**config_params),
            )

            # Parse response based on format
            if text_format is not None:
                # Structured output - parse as Pydantic model
                parsed_result = text_format.model_validate_json(response.text)
                return parsed_result
            else:
                # Unstructured output - return raw text
                return response.text

        except Exception as e:
            if attempt < retries - 1:
                wait_time = 2**attempt
                logging.info(f"Retry {attempt + 1}/{retries} in {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                logging.error(f"Gemini call failed with error: {e}")
                return None

    return None

@weave.op()
def call_llm_with_retry_google(
    model: str,
    system_instruction: str,
    user_prompt: str,
    text_format = None,
    temperature: float = 1.0,
    max_retries: int | None = None,
    enable_google_search: bool = False,
    top_p: float = 1.0,
    thinking_budget: int | None = None,
):
    """Call Gemini API with optional structured outputs and comprehensive retry logic.

    Args:
        model: Gemini model name (e.g., "gemini-2.5-pro")
        system_instruction: System instruction text
        user_prompt: User prompt text
        text_format: Optional Pydantic model for structured output schema. If None, returns raw text.
        temperature: Temperature for generation
        max_retries: Maximum number of retries per attempt
        enable_google_search: Enable Google Search tool
        top_p: Nucleus sampling parameter
        thinking_budget: Thinking budget for reasoning models

    Returns:
        Parsed Pydantic model instance (if text_format provided) or raw text response

    Raises:
        ValueError: If all retries fail
    """
    result = None

    for _ in range(40):
        result = call_llm_with_retry_google_helper(
            model=model,
            system_instruction=system_instruction,
            user_prompt=user_prompt,
            text_format=text_format,
            temperature=temperature,
            max_retries=max_retries,
            enable_google_search=enable_google_search,
            top_p=top_p,
            thinking_budget=thinking_budget,
        )
        if result is not None:
            break

    if result is None:
        raise ValueError("Gemini call failed after 40 retries")

    return result

