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

    # Add web_search without mutating caller's tools list
    if web_search_enabled:
        tools = tools + [{"type": "web_search"}]

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
                # Return parsed Pydantic object directly
                if hasattr(response, 'output_parsed') and response.output_parsed:
                    return response.output_parsed
                elif hasattr(response, 'parsed_output') and response.parsed_output:
                    return response.parsed_output
                else:
                    logging.warning("OpenAI structured output response missing parsed object")
                    return response
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
def call_llm_with_retry_anthropic_helper(
    model: str,
    instructions: str,
    tools: list,
    messages: list,
    max_retries: int | None = None,
    web_search_enabled: bool = False,
    text_format = None,
    max_tokens: int = 16384,
):
    """Helper function to call Anthropic API with retry logic.

    Args:
        model: Anthropic model name (e.g., "claude-sonnet-4-5-20250929")
        instructions: System instruction/prompt
        tools: List of tools in Anthropic format
        messages: List of message dicts with 'role' and 'content'
        max_retries: Maximum number of retries (default from config)
        web_search_enabled: Whether to enable web search tool
        text_format: Optional Pydantic model for structured outputs
        max_tokens: Maximum tokens to generate (default: 8192)

    Returns:
        - If text_format provided: Parsed Pydantic model instance (direct object, not response.parsed_output)
        - If text_format None: Anthropic response object with .content, .stop_reason, .usage
        - On failure: None
    """
    try:
        from anthropic import Anthropic, APIError, RateLimitError, APIConnectionError, InternalServerError, APIStatusError
    except ImportError as e:
        logging.error(f"Failed to import anthropic: {e}")
        return None

    client = Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))
    cfg = get_config()
    runtime_cfg = cfg.get("runtime")
    retries = max_retries or runtime_cfg.get("llm_max_retries")
    if retries < 1:
        retries = 1

    # Add web search tool if enabled (don't mutate caller's tools list)
    if web_search_enabled:
        WEB_SEARCH_TOOL = {
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 20
        }
        tools = tools + [WEB_SEARCH_TOOL]

    for attempt in range(retries):
        try:
            if text_format is not None:
                # Use beta API for structured outputs
                response = client.beta.messages.parse(
                    model=model,
                    betas=["structured-outputs-2025-11-13"],
                    system=instructions,
                    messages=messages,
                    max_tokens=max_tokens,
                    output_format=text_format,
                    tools=tools if tools else [],
                    thinking={
                        "type": "enabled",
                        "budget_tokens": 4096
                    },
                )
                # Return parsed Pydantic object directly
                if hasattr(response, 'parsed_output') and response.parsed_output:
                    return response.parsed_output
                else:
                    logging.warning("Anthropic structured output response missing parsed_output")
                    return response
            else:
                # Use regular messages API
                response = client.messages.create(
                    model=model,
                    system=instructions,
                    messages=messages,
                    max_tokens=max_tokens,
                    tools=tools if tools else [],
                    thinking={
                        "type": "enabled",
                        "budget_tokens": 4096
                    },
                )
                return response

        except RateLimitError as e:
            if attempt < retries - 1:
                wait_time = 2**attempt
                logging.info(f"Anthropic rate limit. Retry {attempt + 1}/{retries} in {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                logging.error(f"Anthropic rate limit exceeded: {e}")
                return None

        except APIConnectionError as e:
            if attempt < retries - 1:
                wait_time = 2**attempt
                logging.info(f"Anthropic connection error. Retry {attempt + 1}/{retries} in {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                logging.error(f"Anthropic connection failed: {e}")
                return None

        except InternalServerError as e:
            if attempt < retries - 1:
                wait_time = 2**attempt
                logging.info(f"Anthropic server error. Retry {attempt + 1}/{retries} in {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                logging.error(f"Anthropic server error: {e}")
                return None

        except APIStatusError as e:
            # Other API errors (4xx, etc.)
            logging.error(f"Anthropic API status error: {e}")
            if attempt < retries - 1:
                wait_time = 2**attempt
                time.sleep(wait_time)
                continue
            return None

        except APIError as e:
            # General API errors
            logging.error(f"Anthropic API error: {e}")
            if attempt < retries - 1:
                wait_time = 2**attempt
                time.sleep(wait_time)
                continue
            return None

        except Exception as e:
            logging.error(f"Unexpected error calling Anthropic: {e}")
            if attempt < retries - 1:
                wait_time = 2**attempt
                time.sleep(wait_time)
                continue
            return None

    return None

@weave.op()
def call_llm_with_retry_anthropic(
    model: str,
    instructions: str,
    tools: list,
    messages: list,
    max_retries: int | None = None,
    web_search_enabled: bool = False,
    text_format = None,
    max_tokens: int = 16384,
):
    """Call Anthropic API with comprehensive retry logic.

    This function wraps call_llm_with_retry_anthropic_helper with an outer retry loop,
    attempting up to 40 times before giving up.

    Args:
        model: Anthropic model name (e.g., "claude-sonnet-4-5-20250929")
        instructions: System instruction/prompt
        tools: List of tools in Anthropic format
        messages: List of message dicts with 'role' and 'content'
        max_retries: Maximum number of retries per attempt (default from config)
        web_search_enabled: Whether to enable web search tool
        text_format: Optional Pydantic model for structured outputs
        max_tokens: Maximum tokens to generate (default: 8192)

    Returns:
        - If text_format provided: Parsed Pydantic model instance (direct object)
        - If text_format None: Anthropic response object with .content, .stop_reason, .usage

    Raises:
        ValueError: If all retries fail after 40 attempts

    Example:
        # Regular call
        response = call_llm_with_retry_anthropic(
            model="claude-sonnet-4-5-20250929",
            instructions="You are a helpful assistant.",
            tools=[],
            messages=[{"role": "user", "content": "Hello"}],
        )
        text = response.content[0].text

        # Structured output call
        from pydantic import BaseModel
        class Output(BaseModel):
            answer: str

        result = call_llm_with_retry_anthropic(
            model="claude-sonnet-4-5-20250929",
            instructions="Extract the answer.",
            tools=[],
            messages=[{"role": "user", "content": "What is 2+2?"}],
            text_format=Output,
        )
        print(result.answer)  # Direct access - result is already the Pydantic object!
    """
    result = None

    for _ in range(40):
        result = call_llm_with_retry_anthropic_helper(
            model=model,
            instructions=instructions,
            tools=tools,
            messages=messages,
            max_retries=max_retries,
            web_search_enabled=web_search_enabled,
            text_format=text_format,
            max_tokens=max_tokens,
        )
        if result is not None:
            break

    if result is None:
        raise ValueError("Anthropic call failed after 40 retries")

    return result

@weave.op()
def call_llm_with_retry_google_helper(
    model: str,
    system_instruction: str,
    user_prompt: str | list = None,
    text_format = None,
    temperature: float = 1.0,
    max_retries: int | None = None,
    enable_google_search: bool = False,
    enable_url_context: bool = False,
    top_p: float = 1.0,
    thinking_budget: int | None = None,
    thinking_level: str | None = None,
    include_thoughts: bool = False,
    function_declarations: list = None,
):
    """Helper function to call Gemini API with full feature support and retry logic.

    Args:
        model: Gemini model name (e.g., "gemini-3-pro-preview", "gemini-2.5-pro")
        system_instruction: System instruction text
        user_prompt: User prompt text (str) or multi-turn conversation (list of dicts)
        text_format: Optional Pydantic model for structured output schema. If None, returns raw text.
        temperature: Temperature for generation (default: 1.0)
        max_retries: Maximum number of retries
        enable_google_search: Enable Google Search tool
        enable_url_context: Enable URL context tool for reading web pages
        top_p: Nucleus sampling parameter (default: 1.0)
        thinking_budget: Thinking budget in tokens for Gemini 2.5 (128-32768, -1 for dynamic)
        thinking_level: Thinking level for Gemini 3 ("low" or "high")
        include_thoughts: Whether to return thought summaries (default: False)
        function_declarations: List of function declarations for function calling

    Returns:
        - If text_format provided: Parsed Pydantic model instance
        - If text_format None: Raw text response string
        - On failure: None
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        logging.error(f"Failed to import google.genai: {e}")
        return None

    # Set location based on model
    if model.startswith("gemini"):
        os.environ['GOOGLE_CLOUD_LOCATION'] = 'global'
    else:
        os.environ['GOOGLE_CLOUD_LOCATION'] = 'us-central1'

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
                tools.append(types.Tool(google_search=types.GoogleSearch()))
            if enable_url_context:
                tools.append(types.Tool(url_context=types.UrlContext()))
            if function_declarations:
                tools.append(types.Tool(function_declarations=function_declarations))

            # Build config params
            config_params = {
                "temperature": temperature,
                "top_p": top_p,
                "system_instruction": system_instruction,  # Direct string, not wrapped
                "tools": tools if tools else None,
            }

            # Add thinking configuration
            if thinking_level is not None or thinking_budget is not None or include_thoughts:
                thinking_config_params = {}
                if thinking_level is not None:
                    thinking_config_params["thinking_level"] = thinking_level
                if thinking_budget is not None:
                    thinking_config_params["thinking_budget"] = thinking_budget
                if include_thoughts:
                    thinking_config_params["include_thoughts"] = include_thoughts
                config_params["thinking_config"] = types.ThinkingConfig(**thinking_config_params)

            # Add structured output params if text_format provided
            if text_format is not None:
                config_params["response_mime_type"] = "application/json"
                config_params["response_json_schema"] = text_format.model_json_schema()

            # Handle user_prompt: can be string or list (multi-turn)
            if isinstance(user_prompt, str):
                contents = user_prompt
            elif isinstance(user_prompt, list):
                contents = user_prompt
            else:
                contents = user_prompt

            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(**config_params),
            )

            # Parse response based on format
            if text_format is not None:
                # Structured output - parse as Pydantic model with fallback
                try:
                    parsed_result = text_format.model_validate_json(response.text)
                    return parsed_result
                except Exception as parse_error:
                    # Fallback: strip markdown code fences if present
                    logging.warning(f"Initial JSON parsing failed, attempting to clean response: {parse_error}")
                    cleaned_text = response.text.lstrip("```json").lstrip("```").rstrip("```").strip()
                    parsed_result = text_format.model_validate_json(cleaned_text)
                    return parsed_result
            else:
                # Unstructured output - return raw text
                return response.text

        except Exception as e:
            if attempt < retries - 1:
                wait_time = 2**attempt
                logging.info(f"Gemini retry {attempt + 1}/{retries} in {wait_time}s... Error: {str(e)[:100]}")
                time.sleep(wait_time)
                continue
            else:
                logging.error(f"Gemini call failed after {retries} attempts: {e}")
                return None

    return None

@weave.op()
def call_llm_with_retry_google(
    model: str,
    system_instruction: str,
    user_prompt: str | list = None,
    text_format = None,
    temperature: float = 1.0,
    max_retries: int | None = None,
    enable_google_search: bool = False,
    enable_url_context: bool = False,
    top_p: float = 1.0,
    thinking_budget: int | None = None,
    thinking_level: str | None = None,
    include_thoughts: bool = False,
    function_declarations: list = None,
):
    """Call Gemini API with full feature support and comprehensive retry logic.

    This function provides access to all Gemini capabilities including:
    - Text generation with system instructions
    - Thinking/reasoning mode (Gemini 2.5 & 3)
    - Structured output with Pydantic schemas
    - Google Search and URL Context tools (unique to Gemini)
    - Function calling
    - Multi-turn conversations

    Args:
        model: Gemini model name (e.g., "gemini-3-pro-preview", "gemini-2.5-pro")
        system_instruction: System instruction text
        user_prompt: User prompt text (str) or multi-turn conversation (list)
        text_format: Optional Pydantic model for structured output. If None, returns raw text.
        temperature: Temperature for generation (default: 1.0)
        max_retries: Maximum number of retries per attempt
        enable_google_search: Enable Google Search tool (Gemini native)
        enable_url_context: Enable URL context tool for reading web pages (Gemini native)
        top_p: Nucleus sampling parameter (default: 1.0)
        thinking_budget: Thinking budget in tokens for Gemini 2.5 (128-32768, -1 for dynamic)
        thinking_level: Thinking level for Gemini 3 ("low" or "high")
        include_thoughts: Whether to return thought summaries
        function_declarations: List of function declarations for function calling

    Returns:
        - If text_format provided: Parsed Pydantic model instance
        - If text_format None: Raw text response string

    Raises:
        ValueError: If all retries fail after 40 attempts

    Examples:
        # Basic text generation
        response = call_llm_with_retry_google(
            model="gemini-3-pro-preview",
            system_instruction="You are a helpful assistant.",
            user_prompt="Explain quantum computing."
        )

        # With Google Search
        response = call_llm_with_retry_google(
            model="gemini-3-pro-preview",
            system_instruction="Research assistant",
            user_prompt="What are the latest AI developments in 2025?",
            enable_google_search=True,
            enable_url_context=True,
        )

        # With thinking mode (Gemini 3)
        response = call_llm_with_retry_google(
            model="gemini-3-pro-preview",
            system_instruction="Math tutor",
            user_prompt="Solve this complex problem...",
            thinking_level="high",
            include_thoughts=True,
        )

        # Structured output
        from pydantic import BaseModel
        class Analysis(BaseModel):
            summary: str
            score: float

        result = call_llm_with_retry_google(
            model="gemini-3-pro-preview",
            system_instruction="Analyze this data",
            user_prompt="...",
            text_format=Analysis,
            enable_google_search=True,
        )
        print(result.summary, result.score)
    """
    result = None

    for _ in range(40):  # Match Anthropic's 40 retries
        result = call_llm_with_retry_google_helper(
            model=model,
            system_instruction=system_instruction,
            user_prompt=user_prompt,
            text_format=text_format,
            temperature=temperature,
            max_retries=max_retries,
            enable_google_search=enable_google_search,
            enable_url_context=enable_url_context,
            top_p=top_p,
            thinking_budget=thinking_budget,
            thinking_level=thinking_level,
            include_thoughts=include_thoughts,
            function_declarations=function_declarations,
        )
        if result is not None:
            break

    if result is None:
        raise ValueError("Gemini call failed after 40 retries")

    return result