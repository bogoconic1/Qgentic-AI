import logging
import os
import time
import json
from openai import OpenAI
import openai
import weave
import requests

from project_config import get_config


def _build_directory_listing(root: str, num_files: int | None = None) -> str:
    cfg = get_config()
    runtime_cfg = cfg.get("runtime")
    path_cfg = cfg.get("paths")
    limit = num_files if num_files is not None else runtime_cfg.get(
        "directory_listing_max_files")
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
                dirs[:] = sorted(
                    [d for d in dirs if d.startswith("external_data_")])
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
        folder_display = "." if rel_root in (
            ".", "") else os.path.basename(rel_root)
        lines.append(f"{indent}{folder_display}/")

        # Print files according to policy and limit
        for name in files_to_show[:limit]:  # to avoid stuffing context window
            lines.append(f"{indent}    {name}")
        if len(files_to_show) > limit:
            lines.append(
                f"{indent}    ... ({len(files_to_show) - limit} more files)")

    return "\n".join(lines)


@weave.op()
def call_llm_with_retry_helper(
    model: str,
    instructions: str,
    tools: list,
    messages: list,
    max_retries: int | None = None,
    web_search_enabled: bool = False
):

    client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
    cfg = get_config()
    runtime_cfg = cfg.get("runtime")
    retries = max_retries or runtime_cfg.get("llm_max_retries")
    if retries < 1:
        retries = 1
    if web_search_enabled:
        tools.append({"type": "web_search"})

    for attempt in range(retries):
        try:
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
                logging.info(
                    f"Retry {attempt + 1}/{retries} in {wait_time}s...")
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
    web_search_enabled: bool = False
):
    result = None

    for _ in range(40):
        result = call_llm_with_retry_helper(
            model=model,
            instructions=instructions,
            tools=tools,
            messages=messages,
            max_retries=max_retries,
            web_search_enabled=web_search_enabled
        )
        if result is not None:
            break

    while result is None:
        result = call_llm_with_retry_helper(
            model=model,
            instructions=instructions,
            tools=tools,
            messages=messages,
            max_retries=max_retries,
            web_search_enabled=web_search_enabled
        )
        time.sleep(60)

    return result


@weave.op()
def call_gemini_api(
    messages: list,
    model: str = "gemini-2.5-pro",
    max_retries: int = 3,
    timeout: int = 30
) -> str | None:
    """
    Calls the Google Gemini API with retry logic.

    Args:
        messages: A list of messages in the Gemini format, e.g.,
                  [{"role": "user", "parts": [{"text": "Hello"}]}]
        model: The Gemini model to use (default: "gemini-pro").
        max_retries: Maximum number of retries (default: 3).
        timeout: Request timeout in seconds (default: 30).

    Returns:
        The generated text content as a string, or None if the call fails.
    """
    logger = logging.getLogger(__name__)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable not set.")
        return None

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    # Adapt OpenAI-style messages to Gemini format if needed
    if messages and "role" in messages[0] and messages[0]["role"] == "user" and "content" in messages[0]:
        gemini_contents = [{"role": "user", "parts": [{"text": msg["content"]}]} if msg["role"] == "user"
                           else {"role": "model", "parts": [{"text": msg["content"]}]}
                           for msg in messages]
    else:
        gemini_contents = messages  # Assume already in Gemini format

    payload = {"contents": gemini_contents}

    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            response_data = response.json()

            text_content = response_data.get('candidates', [{}])[0].get(
                'content', {}).get('parts', [{}])[0].get('text')
            if text_content:
                return text_content
            else:
                logger.warning(
                    "Gemini API call succeeded but returned no text content.")
                return None
        except requests.exceptions.HTTPError as e:
            logger.warning(
                f"Gemini API call failed (attempt {attempt+1}/{max_retries}): {e}")
            if e.response.status_code in (500, 503, 504) and attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1)
                logger.info(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(
                    f"Gemini API call failed permanently: {e.response.text}")
                return None
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during Gemini API call: {e}")
            return None

    logger.error("Gemini API call failed after all retries.")
    return None


@weave.op()
def tavily_search(
    query: str,
    search_depth: str = "advanced",
    include_answer: bool = True,
    max_results: int = 5,
    max_retries: int = 3,
    timeout: int = 30
) -> dict | None:
    """
    Performs a search using the Tavily API with retry logic.

    Args:
        query: The search query string.
        search_depth: "basic" or "advanced" (default: "advanced").
        include_answer: Whether to include a generated answer (default: True).
        max_results: Number of search results to return (default: 5).
        max_retries: Maximum number of retries (default: 3).
        timeout: Request timeout in seconds (default: 30).

    Returns:
        The structured JSON response from Tavily as a dict, or None if it fails.
    """
    logger = logging.getLogger(__name__)
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.error("TAVILY_API_KEY environment variable not set.")
        return None

    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": search_depth,
        "include_answer": include_answer,
        "max_results": max_results
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            logger.warning(
                f"Tavily API call failed (attempt {attempt+1}/{max_retries}): {e}")
            if e.response.status_code in (500, 503, 504) and attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1)
                logger.info(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(
                    f"Tavily API call failed permanently: {e.response.text}")
                return None
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during Tavily API call: {e}")
            return None

    logger.error("Tavily API call failed after all retries.")
    return None
