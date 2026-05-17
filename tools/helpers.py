import json
import logging
import time

import httpx
import openai
import pydantic
import weave

from project_config import get_config
from utils.llm_utils import append_message

_MAX_STRUCTURED_OUTPUT_RETRIES = 3


class StructuredOutputError(Exception):
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
    openai.APIError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
)


def _is_non_retryable_http_status(exc):
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code not in RETRYABLE_HTTP_STATUS_CODES
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code not in RETRYABLE_HTTP_STATUS_CODES
    return False


_503_POLL_INTERVAL = 300  # 5 minutes


def _is_503_unavailable(exc):
    if isinstance(exc, openai.APIStatusError) and exc.status_code == 503:
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
    max_retries: int | None = None,
    enable_google_search: bool = False,
    thinking_level: str | None = "xhigh",
    function_declarations: list = None,
    include_usage: bool = False,
    previous_response_id: str | None = None,
):
    runtime_cfg = get_config()["runtime"]
    retries = max_retries or runtime_cfg["llm_max_retries"]
    backoff_seq = tuple(runtime_cfg["llm_backoff_sequence"])

    tool_list = []
    if enable_google_search:
        tool_list.append({"type": "web_search_preview"})
    if function_declarations:
        tool_list.extend(function_declarations)

    create_params = {
        "model": model,
        "instructions": system_instruction,
    }
    if tool_list:
        create_params["tools"] = tool_list

    reasoning_params = {}
    if thinking_level is not None:
        reasoning_params["effort"] = thinking_level
    if reasoning_params:
        create_params["reasoning"] = reasoning_params

    use_parse = text_format is not None

    if use_parse and not function_declarations:
        create_params["text"] = {
            "format": {
                "type": "json_schema",
                "name": text_format.__name__,
                "schema": text_format.model_json_schema(),
                "strict": True,
            }
        }

    if previous_response_id is not None:
        create_params["previous_response_id"] = previous_response_id

    def _make_request(contents):
        def _attempt():
            client = openai.OpenAI()
            create_params["input"] = contents
            return client.responses.create(**create_params)

        return _retry_with_backoff(
            _attempt, max_retries=retries, backoff_sequence=backoff_seq
        )

    response = _make_request(messages)
    input_tokens = response.usage.input_tokens if include_usage else None

    if text_format is None:
        return (response, input_tokens) if include_usage else response

    current_messages = (
        [append_message("user", messages)]
        if isinstance(messages, str)
        else list(messages)
    )

    for attempt in range(_MAX_STRUCTURED_OUTPUT_RETRIES):
        raw_text = response.output_text
        if raw_text is not None:
            try:
                parsed_result = text_format.model_validate_json(raw_text)
                return (parsed_result, input_tokens) if include_usage else parsed_result
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
        input_tokens = response.usage.input_tokens if include_usage else None
