"""Codebase exploration sub-agent.

A read-only sub-agent that the developer codegen LLM calls when it needs to
understand existing code before writing new code. Uses a multi-step tool
loop to read files, glob, grep, list directories, and run read-only shell
commands across a configurable set of allowed root paths.

Exposed as a plain module-level function ``explore_codebase(query) -> str``
(no state to carry, so no class wrapper). Free-form markdown output (no
Pydantic schema).
"""

from __future__ import annotations

import json
import logging
import shlex
import subprocess
import sysconfig
from pathlib import Path

import weave
from google.genai import types

from project_config import get_config
from prompts.explore import build_system, build_user
from tools.helpers import call_llm
from utils.llm_utils import append_message, get_explore_tools


logger = logging.getLogger(__name__)


_CONFIG = get_config()
_LLM_CFG = _CONFIG["llm"]
_RUNTIME_CFG = _CONFIG["runtime"]
_DEVELOPER_TOOL_MODEL = _LLM_CFG["developer_tool_model"]
_MAX_STEPS = _RUNTIME_CFG["explore_max_steps"]

_USER_ROOTS = [Path(r).resolve() for r in _RUNTIME_CFG["explore_allowed_roots"]]
_SITE_PACKAGES = Path(sysconfig.get_path("purelib")).resolve()
_ALLOWED_ROOTS: list[Path] = _USER_ROOTS + [_SITE_PACKAGES]


# ---------------------------------------------------------------------------
# Bash command allowlist
# ---------------------------------------------------------------------------


_BASH_ALLOWED_COMMANDS = frozenset(
    {
        "ls",
        "cat",
        "head",
        "tail",
        "wc",
        "file",
        "find",
        "grep",
        "tree",
        "du",
        "stat",
    }
)
_BASH_ALLOWED_GIT_SUBCOMMANDS = frozenset(
    {
        "status",
        "log",
        "diff",
        "show",
        "blame",
        "ls-files",
        "ls-tree",
    }
)
_BASH_FORBIDDEN_SUBSTRINGS = (";", "&&", "||", "|", ">", "<", "`", "$(")
_BASH_OUTPUT_CAP_BYTES = 8 * 1024
_BASH_TIMEOUT_SECONDS = 15


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_path(path: Path) -> str | None:
    """Return an error string if the resolved path is outside every allowed root."""
    try:
        resolved = path.resolve()
    except Exception as exc:
        return f"Failed to resolve path: {exc}"
    for root in _ALLOWED_ROOTS:
        try:
            resolved.relative_to(root)
            return None
        except ValueError:
            continue
    return (
        f"Path {resolved} is outside the allowed roots "
        f"({', '.join(str(r) for r in _ALLOWED_ROOTS)})"
    )


def _validate_bash_command(command: str) -> str | None:
    """Validate a bash command against the read-only allowlist.

    Rejects forbidden substrings (pipes, redirection, chaining, backticks,
    command substitution) and any leading command not in the allowlist.
    """
    if not command or not command.strip():
        return "Empty command"

    for token in _BASH_FORBIDDEN_SUBSTRINGS:
        if token in command:
            return (
                f"Forbidden token {token!r} in command. Pipes, redirection, "
                f"chaining, backticks, and $() are not allowed — use the "
                f"dedicated read_file/glob_files/grep_code/list_dir tools instead."
            )

    try:
        parts = shlex.split(command)
    except ValueError as exc:
        return f"Failed to parse command: {exc}"
    if not parts:
        return "Empty command after parsing"

    head = parts[0]
    if head == "git":
        if len(parts) < 2:
            return "git command requires a subcommand"
        subcmd = parts[1]
        if subcmd not in _BASH_ALLOWED_GIT_SUBCOMMANDS:
            return (
                f"git subcommand {subcmd!r} is not allowed. Allowed: "
                f"{sorted(_BASH_ALLOWED_GIT_SUBCOMMANDS)}"
            )
        return None

    if head not in _BASH_ALLOWED_COMMANDS:
        return (
            f"Command {head!r} is not allowed. Allowed commands: "
            f"{sorted(_BASH_ALLOWED_COMMANDS)} (and 'git' with read-only subcommands)"
        )
    return None


# ---------------------------------------------------------------------------
# Tool implementations (read-only, allowlisted paths)
# ---------------------------------------------------------------------------


def _tool_read_file(
    path: str, start_line: int | None = None, end_line: int | None = None
) -> str:
    full_path = Path(path)
    error = _validate_path(full_path)
    if error:
        return json.dumps({"error": error})

    resolved = full_path.resolve()
    if not resolved.exists():
        return json.dumps({"error": f"File not found: {path}"})
    if not resolved.is_file():
        return json.dumps({"error": f"Not a file: {path}"})

    try:
        lines = resolved.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        return json.dumps({"error": f"Failed to read file: {exc}"})

    total_lines = len(lines)

    if start_line is not None or end_line is not None:
        s = (start_line or 1) - 1
        e = end_line or total_lines
        lines = lines[s:e]
        line_offset = s + 1
    else:
        line_offset = 1
        if total_lines > 300:
            lines = lines[:300]
            numbered = "\n".join(
                f"{i + line_offset}: {line}" for i, line in enumerate(lines)
            )
            return json.dumps(
                {
                    "content": numbered,
                    "total_lines": total_lines,
                    "truncated": True,
                    "showing": "1-300",
                }
            )

    numbered = "\n".join(f"{i + line_offset}: {line}" for i, line in enumerate(lines))
    return json.dumps({"content": numbered, "total_lines": total_lines})


def _tool_glob_files(root: str, pattern: str) -> str:
    root_path = Path(root)
    error = _validate_path(root_path)
    if error:
        return json.dumps({"error": error})

    resolved = root_path.resolve()
    if not resolved.exists():
        return json.dumps({"error": f"Root not found: {root}"})
    if not resolved.is_dir():
        return json.dumps({"error": f"Root is not a directory: {root}"})

    try:
        matches = sorted(str(p.relative_to(resolved)) for p in resolved.glob(pattern))
    except Exception as exc:
        return json.dumps({"error": f"Glob failed: {exc}"})

    total = len(matches)
    if total > 50:
        matches = matches[:50]

    return json.dumps({"matches": matches, "total": total, "truncated": total > 50})


def _tool_grep_code(
    root: str,
    pattern: str,
    file_glob: str = "*.py",
    max_results: int = 20,
) -> str:
    root_path = Path(root)
    error = _validate_path(root_path)
    if error:
        return json.dumps({"error": error})

    resolved = root_path.resolve()
    if not resolved.exists():
        return json.dumps({"error": f"Root not found: {root}"})
    if not resolved.is_dir():
        return json.dumps({"error": f"Root is not a directory: {root}"})

    try:
        result = subprocess.run(
            ["grep", "-rn", f"--include={file_glob}", "-E", pattern, str(resolved)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        lines = result.stdout.strip().splitlines()
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "Search timed out"})
    except Exception as exc:
        return json.dumps({"error": f"Search failed: {exc}"})

    cleaned = lines[:max_results]
    return json.dumps(
        {
            "matches": cleaned,
            "total_matches": len(lines),
            "showing": min(max_results, len(lines)),
        }
    )


def _tool_list_dir(path: str, max_entries: int = 100) -> str:
    full_path = Path(path)
    error = _validate_path(full_path)
    if error:
        return json.dumps({"error": error})

    resolved = full_path.resolve()
    if not resolved.exists():
        return json.dumps({"error": f"Path not found: {path}"})
    if not resolved.is_dir():
        return json.dumps({"error": f"Not a directory: {path}"})

    try:
        children = sorted(resolved.iterdir(), key=lambda p: p.name)
    except Exception as exc:
        return json.dumps({"error": f"Failed to list directory: {exc}"})

    total = len(children)
    truncated = total > max_entries
    if truncated:
        children = children[:max_entries]

    entries = [f"{p.name}/" if p.is_dir() else p.name for p in children]
    return json.dumps(
        {
            "entries": entries,
            "total": total,
            "showing": len(entries),
            "truncated": truncated,
        }
    )


def _tool_bash_readonly(command: str) -> str:
    error = _validate_bash_command(command)
    if error:
        return json.dumps({"error": error})

    try:
        parts = shlex.split(command)
    except ValueError as exc:
        return json.dumps({"error": f"Failed to parse command: {exc}"})

    try:
        result = subprocess.run(
            parts,
            capture_output=True,
            text=True,
            timeout=_BASH_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return json.dumps({"error": f"Command timed out after {_BASH_TIMEOUT_SECONDS}s"})
    except FileNotFoundError as exc:
        return json.dumps({"error": f"Command not found: {exc}"})
    except Exception as exc:
        return json.dumps({"error": f"Command failed: {exc}"})

    output = (result.stdout or "") + (result.stderr or "")
    truncated = len(output) > _BASH_OUTPUT_CAP_BYTES
    if truncated:
        output = output[:_BASH_OUTPUT_CAP_BYTES] + "\n[output truncated]"

    return json.dumps(
        {
            "output": output,
            "returncode": result.returncode,
            "truncated": truncated,
        }
    )


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------


def _execute_tool_call(item) -> str:
    args = dict(item.args)

    if item.name == "read_file":
        return _tool_read_file(
            args["path"],
            args.get("start_line"),
            args.get("end_line"),
        )
    if item.name == "glob_files":
        return _tool_glob_files(args["root"], args["pattern"])
    if item.name == "grep_code":
        return _tool_grep_code(
            args["root"],
            args["pattern"],
            args.get("file_glob", "*.py"),
            args.get("max_results", 20),
        )
    if item.name == "list_dir":
        return _tool_list_dir(args["path"], args.get("max_entries", 100))
    if item.name == "bash_readonly":
        return _tool_bash_readonly(args["command"])

    return json.dumps({"error": f"Unknown tool: {item.name}"})


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


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

    for step in range(_MAX_STEPS):
        is_last_step = step == _MAX_STEPS - 1
        logger.info("Explore step %d/%d", step + 1, _MAX_STEPS)

        response = call_llm(
            model=_DEVELOPER_TOOL_MODEL,
            system_instruction=system_prompt,
            function_declarations=tools if not is_last_step else [],
            messages=input_list,
            enable_google_search=True,
        )

        parts = response.candidates[0].content.parts
        has_function_calls = any(
            part.function_call for part in parts if hasattr(part, "function_call")
        )

        if not has_function_calls:
            logger.info("Explore completed at step %d", step + 1)
            return response.text or ""

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

    # Exhausted steps — force a final text response
    logger.warning("Explore exhausted %d steps, forcing final report", _MAX_STEPS)
    response = call_llm(
        model=_DEVELOPER_TOOL_MODEL,
        system_instruction=system_prompt
        + "\n\nReturn your findings now based on what you have gathered.",
        messages=input_list,
        enable_google_search=True,
    )
    return response.text or ""
