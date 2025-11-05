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
    web_search_enabled: bool = False
):
    cfg = get_config()
    runtime_cfg = cfg.get("runtime")
    retries = max_retries or runtime_cfg.get("llm_max_retries")
    if retries < 1: retries = 1
    if web_search_enabled: tools.append({"type": "web_search"})

    # Detect model provider and API format
    use_chat_completions = model.startswith("gemini-") # NOTE this only works for ask_eda() - if you set it for anything else it will break

    if use_chat_completions:
        # Gemini uses Chat Completions API via OpenAI compatibility layer
        client = OpenAI(
            api_key=os.environ.get('GOOGLE_API_KEY'),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        logging.debug(f"Using Gemini API (Chat Completions) for model: {model}")
    else:
        # OpenAI uses Responses API
        client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
        logging.debug(f"Using OpenAI API (Responses) for model: {model}")

    for attempt in range(retries):
        try:
            if use_chat_completions:
                # Chat Completions API (Gemini)
                # Convert instructions to system message
                chat_messages = [{"role": "system", "content": instructions}] + messages
                response = client.chat.completions.create(
                    model=model,
                    messages=chat_messages,
                    tools=tools if tools else None,
                )

                # Adapt Chat Completions response to Responses API format
                class ResponseAdapter:
                    def __init__(self, chat_response):
                        self.output_text = chat_response.choices[0].message.content
                        self.output = [{"role": "assistant", "content": self.output_text}]
                        # Handle tool calls if present
                        if hasattr(chat_response.choices[0].message, 'tool_calls') and chat_response.choices[0].message.tool_calls:
                            self.tool_calls = chat_response.choices[0].message.tool_calls

                return ResponseAdapter(response)
            else:
                # Responses API (OpenAI)
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

