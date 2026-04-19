import json
import logging
import time

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
import httpx
import pydantic
import weave

from project_config import get_config
from utils.llm_utils import append_message

_MAX_STRUCTURED_OUTPUT_RETRIES = 3


class StructuredOutputError(Exception):
    """Raised when the LLM fails to produce valid JSON after retries."""

    pass


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


_503_POLL_INTERVAL = 300  # 5 minutes


def _is_503_unavailable(exc):
    if isinstance(exc, genai_errors.ServerError) and exc.code == 503:
        return True
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 503:
        return True
    return False


def _retry_with_backoff(func, *, max_retries, backoff_sequence):
    last_exception = None
    attempt = 0

    while attempt <= max_retries:
        try:
            return func()
        except RETRYABLE_EXCEPTIONS as e:
            if _is_non_retryable_http_status(e):
                raise

            if _is_503_unavailable(e):
                logging.warning(
                    "503 Unavailable (attempt %d): %s. Polling again in %ds...",
                    attempt + 1,
                    str(e),
                    _503_POLL_INTERVAL,
                )
                time.sleep(_503_POLL_INTERVAL)
                continue

            last_exception = e
            if attempt < max_retries:
                backoff = backoff_sequence[min(attempt, len(backoff_sequence) - 1)]
                logging.warning(
                    "API call failed (attempt %d/%d): %s: %s. Retrying in %.1fs...",
                    attempt + 1,
                    max_retries + 1,
                    type(e).__name__,
                    str(e),
                    backoff,
                )
                time.sleep(backoff)
            else:
                logging.error(
                    "API call failed after %d attempts: %s: %s",
                    max_retries + 1,
                    type(e).__name__,
                    str(e),
                )
            attempt += 1
        except Exception as e:
            logging.error("Non-retryable error: %s: %s", type(e).__name__, str(e))
            raise

    raise last_exception


@weave.op()
def call_llm(
    model: str,
    system_instruction: str,
    messages: str | list = None,
    text_format=None,
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
    runtime_cfg = get_config()["runtime"]
    retries = max_retries or runtime_cfg["llm_max_retries"]
    backoff_seq = tuple(runtime_cfg["llm_backoff_sequence"])

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

    thinking_config_params = {}
    if thinking_level is not None:
        thinking_config_params["thinking_level"] = thinking_level
    if thinking_budget is not None:
        thinking_config_params["thinking_budget"] = thinking_budget
    if include_thoughts:
        thinking_config_params["include_thoughts"] = include_thoughts
    config_params["thinking_config"] = genai_types.ThinkingConfig(
        **thinking_config_params
    )

    if text_format is not None:
        config_params["response_mime_type"] = "application/json"
        config_params["response_json_schema"] = text_format.model_json_schema()

    config = genai_types.GenerateContentConfig(**config_params)

    def _make_request(contents):
        def _attempt():
            client = genai.Client()
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )

        return _retry_with_backoff(
            _attempt, max_retries=retries, backoff_sequence=backoff_seq
        )

    response = _make_request(messages)
    input_tokens = (
        response.usage_metadata.prompt_token_count if include_usage else None
    )

    if text_format is None:
        return (response, input_tokens) if include_usage else response

    # Structured output: parse JSON, retry with corrective feedback on failure
    current_messages = (
        [append_message("user", messages)] if isinstance(messages, str) else list(messages)
    )

    for attempt in range(_MAX_STRUCTURED_OUTPUT_RETRIES):
        raw_text = response.text
        if raw_text is not None:
            try:
                parsed_result = text_format.model_validate_json(raw_text)
                return (
                    (parsed_result, input_tokens) if include_usage else parsed_result
                )
            except (pydantic.ValidationError, json.JSONDecodeError) as e:
                error_msg = str(e)
        else:
            error_msg = "Empty response (no text)"

        if attempt == _MAX_STRUCTURED_OUTPUT_RETRIES - 1:
            raise StructuredOutputError(
                f"Structured output failed after {_MAX_STRUCTURED_OUTPUT_RETRIES} "
                f"attempts: {error_msg}"
            )

        logging.warning(
            "Structured output invalid (attempt %d/%d): %s. Response: %.500s",
            attempt + 1,
            _MAX_STRUCTURED_OUTPUT_RETRIES,
            error_msg,
            raw_text or "",
        )

        current_messages = current_messages + [
            append_message("assistant", raw_text or ""),
            append_message(
                "user",
                f"Your response was not valid JSON conforming to the required schema. "
                f"Error: {error_msg}\n\n"
                f"Please regenerate your response as valid JSON matching the schema.",
            ),
        ]

        response = _make_request(current_messages)
        input_tokens = (
            response.usage_metadata.prompt_token_count if include_usage else None
        )
