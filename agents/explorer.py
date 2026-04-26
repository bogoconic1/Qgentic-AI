"""Codebase exploration sub-agent.

A read-only sub-agent that the developer codegen LLM calls when it needs to
understand existing code before writing new code. Uses a multi-step tool
loop to read files, glob, grep, list directories, and run safety-judged
shell commands across a configurable set of allowed root paths.

Exposed as a plain module-level function ``explore_codebase(query) -> str``
(no state to carry, so no class wrapper). Free-form markdown output (no
Pydantic schema).
"""

from __future__ import annotations

import json
import logging

import weave
from google.genai import types

from project_config import get_config
from prompts.explore import build_system, build_user
from tools.filesystem import (
    _ALLOWED_ROOTS,
    execute_filesystem_tool,
)
from tools.helpers import call_llm
from utils.llm_utils import append_message, get_explore_tools
from utils.output import truncate_for_llm


logger = logging.getLogger(__name__)


_CONFIG = get_config()
_LLM_CFG = _CONFIG["llm"]
_DEVELOPER_TOOL_MODEL = _LLM_CFG["developer_tool_model"]


def _execute_tool_call(item) -> str:
    args = dict(item.args)
    result = execute_filesystem_tool(item.name, args)
    if result is not None:
        return result
    return json.dumps({"error": f"Unknown tool: {item.name}"})


@weave.op()
def explore_codebase(query: str) -> str:
    """Run the codebase exploration sub-agent and return a markdown report.

    Args:
        query: Natural language question about the codebase or libraries.

    Returns:
        Free-form markdown report (with file:line citations).
    """
    logger.info("Starting codebase exploration: %s", query[:100])

    allowed_roots_display = [str(r) for r in _ALLOWED_ROOTS]
    system_prompt = build_system(allowed_roots_display)
    user_prompt = build_user(query)
    tools = get_explore_tools()
    input_list = [append_message("user", user_prompt)]

    step = 0
    while True:
        step += 1
        logger.info("Explore step %d", step)

        response = call_llm(
            model=_DEVELOPER_TOOL_MODEL,
            system_instruction=system_prompt,
            function_declarations=tools,
            messages=input_list,
            enable_google_search=True,
        )

        parts = response.candidates[0].content.parts
        has_function_calls = any(
            part.function_call for part in parts if hasattr(part, "function_call")
        )

        if not has_function_calls:
            logger.info("Explore completed at step %d", step)
            return truncate_for_llm(response.text or "")

        function_responses = []
        for part in parts:
            if hasattr(part, "function_call") and part.function_call:
                tool_result_str = _execute_tool_call(part.function_call)
                function_responses.append(
                    types.Part.from_function_response(
                        name=part.function_call.name,
                        response={"result": tool_result_str},
                    )
                )

        input_list.append(response.candidates[0].content)
        if function_responses:
            input_list.append(
                types.Content(role="function", parts=function_responses)
            )
