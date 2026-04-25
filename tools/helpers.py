import json
import logging
import os
import sys
import time
from pathlib import Path

import httpx
import pydantic
import weave

from project_config import get_config
from utils.llm_utils import append_message, extract_text_from_response

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_OPENROUTER_SRC = _REPO_ROOT / "libraries" / "python-sdk" / "src"
if _LOCAL_OPENROUTER_SRC.exists():
    sys.path.insert(0, str(_LOCAL_OPENROUTER_SRC))

from openrouter import OpenRouter, errors as openrouter_errors  # noqa: E402

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
    openrouter_errors.OpenRouterError,
)


def _is_non_retryable_http_status(exc):
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code not in RETRYABLE_HTTP_STATUS_CODES
    if isinstance(exc, openrouter_errors.OpenRouterError):
        return exc.status_code not in RETRYABLE_HTTP_STATUS_CODES
    return False


_503_POLL_INTERVAL = 300  # 5 minutes


def _is_503_unavailable(exc):
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 503:
        return True
    if isinstance(exc, openrouter_errors.OpenRouterError) and exc.status_code == 503:
        return True
    return False


def _normalize_messages(messages: str | list | None) -> list[dict]:
    if messages is None:
        return []
    if isinstance(messages, str):
        return [append_message("user", messages)]
    return list(messages)


def _prompt_tokens(response) -> int | None:
    if response.usage is None:
        return None
    return response.usage.prompt_tokens


def _json_schema_response_format(text_format) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": text_format.__name__,
            "schema": text_format.model_json_schema(),
            "strict": True,
        },
    }


def _web_search_tool() -> dict:
    return {
        "type": "openrouter:web_search",
        "parameters": {
            "engine": "auto",
            "max_results": 5,
            "max_total_results": 10,
        },
    }


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
    enable_web_search: bool = False,
    top_p: float = 1.0,
    reasoning_effort: str | None = None,
    function_declarations: list = None,
    include_usage: bool = False,
):
    runtime_cfg = get_config()["runtime"]
    retries = max_retries or runtime_cfg["llm_max_retries"]
    backoff_seq = tuple(runtime_cfg["llm_backoff_sequence"])

    tool_list = []
    if function_declarations:
        tool_list.extend(function_declarations)
    if enable_web_search:
        tool_list.append(_web_search_tool())

    response_format = (
        _json_schema_response_format(text_format) if text_format is not None else None
    )
    reasoning = {"effort": reasoning_effort} if reasoning_effort is not None else None

    def _make_request(contents):
        def _attempt():
            with OpenRouter(
                api_key=os.environ["OPENROUTER_API_KEY"],
                x_open_router_title="Qgentic-AI",
                x_open_router_categories="cli-agent,kaggle-agent",
            ) as client:
                request_messages = []
                if system_instruction:
                    request_messages.append(
                        {"role": "system", "content": system_instruction}
                    )
                request_messages.extend(_normalize_messages(contents))

                return client.chat.send(
                    model=model,
                    messages=request_messages,
                    tools=tool_list or None,
                    tool_choice="auto" if tool_list else None,
                    parallel_tool_calls=True if function_declarations else None,
                    temperature=temperature,
                    top_p=top_p,
                    response_format=response_format,
                    reasoning=reasoning,
                    stream=False,
                    timeout_ms=600_000,
                )

        return _retry_with_backoff(
            _attempt, max_retries=retries, backoff_sequence=backoff_seq
        )

    response = _make_request(messages)
    input_tokens = _prompt_tokens(response) if include_usage else None

    if text_format is None:
        return (response, input_tokens) if include_usage else response

    # Structured output: parse JSON, retry with corrective feedback on failure
    current_messages = _normalize_messages(messages)

    for attempt in range(_MAX_STRUCTURED_OUTPUT_RETRIES):
        raw_text = extract_text_from_response(response)
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
        input_tokens = _prompt_tokens(response) if include_usage else None
