import logging
import os
import time

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
import httpx
import weave

from project_config import get_config

RETRYABLE_HTTP_STATUS_CODES = {429, 500, 502, 503, 504}


RETRYABLE_EXCEPTIONS = (
    TimeoutError,
    ConnectionError,
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
    httpx.HTTPStatusError,
    genai_errors.ServerError,
    genai_errors.ClientError,
)


def _is_non_retryable_http_status(exc):
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code not in RETRYABLE_HTTP_STATUS_CODES
    return False


def _retry_with_backoff(func, *, max_retries, backoff_sequence):
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except RETRYABLE_EXCEPTIONS as e:
            if _is_non_retryable_http_status(e):
                raise
            last_exception = e
            if attempt < max_retries:
                backoff = backoff_sequence[min(attempt, len(backoff_sequence) - 1)]
                logging.warning(
                    "API call failed (attempt %d/%d): %s: %s. Retrying in %.1fs...",
                    attempt + 1, max_retries + 1, type(e).__name__, str(e), backoff,
                )
                time.sleep(backoff)
            else:
                logging.error(
                    "API call failed after %d attempts: %s: %s",
                    max_retries + 1, type(e).__name__, str(e),
                )
        except Exception as e:
            logging.error("Non-retryable error: %s: %s", type(e).__name__, str(e))
            raise

    raise last_exception


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
def call_llm(
    model: str,
    system_instruction: str,
    messages: str | list = None,
    text_format = None,
    temperature: float = 1.0,
    max_retries: int | None = None,
    enable_google_search: bool = False,
    top_p: float = 1.0,
    thinking_budget: int | None = None,
    thinking_level: str | None = None,
    include_thoughts: bool = False,
    function_declarations: list = None,
    include_usage: bool = False,
):
    runtime_cfg = get_config().get("runtime")
    retries = max_retries or runtime_cfg.get("llm_max_retries")
    backoff_seq = tuple(runtime_cfg.get("llm_backoff_sequence"))

    def _call():
        client = genai.Client()

        # Build tools list - combine into single Tool object
        tool_list = []
        tool_params = {}
        if enable_google_search:
            tool_params["google_search"] = genai_types.GoogleSearch()
        if function_declarations:
            tool_params["function_declarations"] = function_declarations

        if tool_params:
            tool_list = [genai_types.Tool(**tool_params)]

        config_params = {
            "temperature": temperature,
            "top_p": top_p,
            "system_instruction": system_instruction,
            "tools": tool_list if tool_list else None,
        }

        if thinking_level is not None or thinking_budget is not None or include_thoughts:
            thinking_config_params = {}
            if thinking_level is not None:
                thinking_config_params["thinking_level"] = thinking_level
            if thinking_budget is not None:
                thinking_config_params["thinking_budget"] = thinking_budget
            if include_thoughts:
                thinking_config_params["include_thoughts"] = include_thoughts
            config_params["thinking_config"] = genai_types.ThinkingConfig(**thinking_config_params)

        if text_format is not None:
            config_params["response_mime_type"] = "application/json"
            config_params["response_json_schema"] = text_format.model_json_schema()

        response = client.models.generate_content(
            model=model,
            contents=messages,
            config=genai_types.GenerateContentConfig(**config_params),
        )

        input_tokens = None
        if include_usage and hasattr(response, 'usage_metadata') and response.usage_metadata:
            input_tokens = getattr(response.usage_metadata, 'prompt_token_count', None)

        if text_format is not None:
            try:
                parsed_result = text_format.model_validate_json(response.text)
                return (parsed_result, input_tokens) if include_usage else parsed_result
            except Exception as parse_error:
                logging.warning(f"Initial JSON parsing failed, attempting to clean response: {parse_error}")
                cleaned_text = response.text.lstrip("```json").lstrip("```").rstrip("```").strip()
                parsed_result = text_format.model_validate_json(cleaned_text)
                return (parsed_result, input_tokens) if include_usage else parsed_result
        else:
            return (response, input_tokens) if include_usage else response

    return _retry_with_backoff(_call, max_retries=retries, backoff_sequence=backoff_seq)
