"""
LLM utility functions for OpenRouter response handling and tool definitions.
"""

import base64
import json
import logging
import mimetypes
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def encode_image_to_data_url(
    image_path: str | Path, max_size_bytes: int = 4_500_000
) -> str:
    """Encode an image file to a data URL.

    Args:
        image_path: Path to the image file
        max_size_bytes: Maximum file size in bytes (default 4.5MB)

    Returns:
        Data URL string in format: data:{mime};base64,{data}
    """
    path = Path(image_path) if isinstance(image_path, str) else image_path

    mime, _ = mimetypes.guess_type(path.name)
    if mime is None:
        suffix = path.suffix.lower()
        _SUFFIX_TO_MIME = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }
        mime = _SUFFIX_TO_MIME.get(suffix)
        if mime is None:
            raise ValueError(f"Unsupported image type: {path.name}")

    image_bytes = path.read_bytes()

    if len(image_bytes) > max_size_bytes:
        raise ValueError(
            f"Image {path.name} exceeds size limit ({len(image_bytes) / 1024 / 1024:.2f}MB > {max_size_bytes / 1024 / 1024:.2f}MB)"
        )

    data = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{data}"


def extract_text_from_response(response) -> str:
    """Extract text from an OpenRouter chat completion response."""
    content = response.choices[0].message.content
    if content is None:
        return ""
    if not isinstance(content, str):
        raise TypeError(f"Expected text content, got {type(content).__name__}")
    return content


def append_message(role: str, message: str) -> dict:
    """Create an OpenRouter/OpenAI-compatible chat message."""
    return {"role": role, "content": message}


def make_tool_message(tool_call_id: str, result: str | dict) -> dict:
    """Create a chat message that answers one model tool call."""
    content = result if isinstance(result, str) else json.dumps(result)
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def assistant_message_from_response(response) -> dict:
    """Serialize the first assistant message for reuse in chat history."""
    return response.choices[0].message.model_dump(
        mode="json", by_alias=True, exclude_none=True
    )


def get_response_tool_calls(response) -> list:
    """Return tool calls from the first assistant message."""
    tool_calls = response.choices[0].message.tool_calls
    return [] if tool_calls is None else list(tool_calls)


def tool_call_id(tool_call) -> str:
    return tool_call.id


def tool_call_name(tool_call) -> str:
    return tool_call.function.name


def tool_call_args(tool_call) -> dict:
    return json.loads(tool_call.function.arguments)


def _function_tool(name: str, description: str, parameters: dict[str, Any]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def get_tools():
    """
    Get tools as OpenRouter function tool definitions.

    Returns:
        List of OpenRouter tool dictionaries
    """
    return [
        _function_tool(
            "analyze",
            "Write and execute a Python script. The script runs in the task data directory with access to all data files, model outputs, and predictions. Print results to stdout.",
            {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Complete Python script to execute.",
                    }
                },
                "required": ["code"],
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Monitor tools (execute_bash for system diagnostics during training)
# ---------------------------------------------------------------------------


def get_monitor_tools():
    """Get monitor tools (execute_bash) as OpenRouter function tools."""
    return [
        _function_tool(
            "execute_bash",
            "Execute a bash command for system diagnostics. Use for checking GPU utilization (nvidia-smi), process status (ps, top), memory (free), disk (df), etc.",
            {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Bash command to execute.",
                    }
                },
                "required": ["command"],
            },
        )
    ]


# ---------------------------------------------------------------------------
# Deep Research tools (web_research + web_fetch + write_python_code)
# ---------------------------------------------------------------------------


def get_deep_research_tools():
    """Inner tools available to the Deep Research sub-agent."""
    return [
        _function_tool(
            "web_research",
            (
                "Discover web pages for a query via Exa neural search. "
                "Returns a list of results, each with url, title, full page text, "
                "and published_date — not a snippet, the whole text. This is the "
                "ONLY way to discover URLs; never guess or reconstruct URLs."
            ),
            {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language search query.",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": (
                            "Optional number of results to return. Omit to use "
                            "the Exa default."
                        ),
                    },
                },
                "required": ["query"],
            },
        ),
        _function_tool(
            "web_fetch",
            (
                "Fetch a single URL's main content as markdown via Firecrawl. "
                "Full page content is returned — no truncation. Only call with "
                "URLs you got from a prior `web_research` result or from a "
                "markdown link inside a prior `web_fetch` result."
            ),
            {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Absolute URL to fetch.",
                    },
                },
                "required": ["url"],
            },
        ),
        _function_tool(
            "write_python_code",
            (
                "Write a Python script to the research scripts dir and execute "
                "it in a subprocess. Full stdout/stderr is returned. Use for "
                "EDA, API probing, quick computations — anything you'd run in "
                "a notebook to validate an idea."
            ),
            {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Complete Python source to execute.",
                    },
                },
                "required": ["code"],
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Explore tools (read-only, scoped to runtime.explore_allowed_roots)
# ---------------------------------------------------------------------------


def get_explore_tools():
    """Get the read-only tools available to the codebase exploration sub-agent."""
    return [
        _function_tool(
            "read_file",
            "Read a source file. Path is absolute or relative to the current working directory. Returns numbered lines. Files over 300 lines are truncated unless start_line/end_line are specified.",
            {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path or path relative to cwd (e.g. '/workspace/Qgentic-AI/agents/developer.py').",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Start line number (1-indexed). Optional.",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "End line number (1-indexed, inclusive). Optional.",
                    },
                },
                "required": ["path"],
            },
        ),
        _function_tool(
            "glob_files",
            "Find files matching a glob pattern under a root directory. Returns up to 50 matches as paths relative to the root.",
            {
                "type": "object",
                "properties": {
                    "root": {
                        "type": "string",
                        "description": "Directory under one of the allowed roots to glob from.",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (e.g., '**/*.py', '**/sft*.py').",
                    },
                },
                "required": ["root", "pattern"],
            },
        ),
        _function_tool(
            "grep_code",
            "Recursively search a regex pattern in files under a root directory. Returns matching lines with file paths and line numbers.",
            {
                "type": "object",
                "properties": {
                    "root": {
                        "type": "string",
                        "description": "Directory under one of the allowed roots to search in.",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Regular expression pattern (extended regex; passed to grep -E).",
                    },
                    "file_glob": {
                        "type": "string",
                        "description": "File glob to restrict the search (default '*.py'; pass '*' for all files).",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of matching lines to return (default: 20).",
                    },
                },
                "required": ["root", "pattern"],
            },
        ),
        _function_tool(
            "list_dir",
            "List the immediate children of a directory. Directories are suffixed with '/'.",
            {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path under one of the allowed roots.",
                    },
                    "max_entries": {
                        "type": "integer",
                        "description": "Maximum number of entries to return (default: 100).",
                    },
                },
                "required": ["path"],
            },
        ),
        _function_tool(
            "bash_readonly",
            (
                "Run a single read-only shell command. ONLY these commands are allowed: "
                "ls, cat, head, tail, wc, file, find, grep, tree, du, stat, "
                "git status, git log, git diff, git show, git blame, git ls-files, git ls-tree. "
                "Pipes (|), redirection (>, <, >>), command chaining (;, &&, ||), backticks, and $() are forbidden — "
                "use the dedicated read_file/glob_files/grep_code/list_dir tools instead."
            ),
            {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to run (e.g. 'ls -la /workspace/Qgentic-AI/agents').",
                    },
                },
                "required": ["command"],
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Developer tools (explore_codebase for code generation)
# ---------------------------------------------------------------------------


def get_main_agent_tools():
    """Get the tool palette available to the Main Agent."""
    return [
        _function_tool(
            "develop",
            (
                "Runs one developer iteration and OWNS SUBMISSION AUTHORING: the "
                "developer subagent writes a `train.py` that produces whatever "
                "artifact the session goal requires (CSV, ONNX graph, model "
                "weights, generated text, ZIP bundle, …) and dumps "
                "`train_stats.json` with a score. Retries internally until "
                "valid stats land. Returns a structured payload with the final "
                "code, its path on disk, and a summary (score, stats, "
                "stdout_tail, attempts_made). Omit `idea_id` on the very first "
                "call (baseline from the session goal); otherwise pass the "
                "integer id of the entry you selected from INDEX.md (the "
                "`[NNN]` prefix) — the framework resolves it to the full "
                "idea body. DO NOT hand-author submission artifacts yourself "
                "via `analyze` — that is always wrong; it belongs here."
            ),
            {
                "type": "object",
                "properties": {
                    "idea_id": {
                        "type": "integer",
                        "description": "Id of the idea entry to develop (the `[NNN]` prefix in INDEX.md). Omit for baseline.",
                    },
                },
                "required": [],
            },
        ),
        _function_tool(
            "research",
            (
                "Runs one Deep Research iteration: web_fetch + web_search + internal "
                "Python to produce a markdown report answering the instruction. Use "
                "for domain grounding, library docs, prior-art sweeps, empirical "
                "sniff-tests on the dataset."
            ),
            {
                "type": "object",
                "properties": {
                    "instruction": {
                        "type": "string",
                        "description": "Free-form research instruction.",
                    },
                },
                "required": ["instruction"],
            },
        ),
        _function_tool(
            "analyze",
            (
                "Run a Python snippet in a fresh subprocess for INSPECTION and "
                "ANALYSIS. Returns stdout/stderr. Legitimate uses: read files, "
                "inspect artifacts produced by prior `develop` / `research` "
                "calls, reproduce a reported score, grep code for leakage "
                "patterns, compute analyses (per-class F1, calibration, "
                "confusion matrices) over `valid_preds.csv`, list directory "
                "contents. NOT for authoring submission artifacts (call "
                "`develop`), NOT for modifying the Python environment "
                "(`pip install`, `apt`, `conda`), NOT for long training runs. "
                "If you find yourself iterating variants of a solution via "
                "this tool, stop and call `develop(idea=...)` instead."
            ),
            {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Complete Python source to execute.",
                    },
                },
                "required": ["code"],
            },
        ),
        _function_tool(
            "add_idea",
            "Add a new entry to the idea pool. Returns the assigned integer id.",
            {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Short title — becomes the filename slug and the INDEX.md entry.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Full markdown body of the idea.",
                    },
                },
                "required": ["title", "description"],
            },
        ),
        _function_tool(
            "remove_idea",
            "Remove an idea from the pool by id.",
            {
                "type": "object",
                "properties": {
                    "idea_id": {"type": "integer"},
                },
                "required": ["idea_id"],
            },
        ),
        _function_tool(
            "update_idea",
            "Replace the body of an existing idea. Title stays the same.",
            {
                "type": "object",
                "properties": {
                    "idea_id": {"type": "integer"},
                    "description": {"type": "string"},
                },
                "required": ["idea_id", "description"],
            },
        ),
    ]


def get_developer_tools():
    """Get tools available to the developer agents during code generation."""
    return [
        _function_tool(
            "explore_codebase",
            (
                "Ask a natural-language question about the codebase or installed Python "
                "packages and get back a markdown report with file:line citations. The "
                "sub-agent reads source files using read_file/glob_files/grep_code/list_dir "
                "across configured roots — it does NOT execute Python or shell commands. "
                "To verify whether a snippet runs, use `analyze` instead.\n\n"
                "Use this for static investigation: 'How does X work?', 'What's the "
                "signature of Y?', 'Where is Z defined?', 'Show me callers of W'. Brief "
                "the sub-agent like a smart colleague who just walked into the room — it "
                "hasn't seen this conversation. Terse command-style prompts produce "
                "shallow, generic work."
            ),
            {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language question about the codebase or installed libraries.",
                    }
                },
                "required": ["query"],
            },
        ),
        _function_tool(
            "analyze",
            (
                "Run a Python snippet in a fresh subprocess and get back stdout, "
                "stderr, and the exit code. Use this to VERIFY behavior — test an "
                "import, probe an API, check a library version, prototype a function, "
                "run a quick computation. The snippet runs with a timeout. Use this "
                "tool when you would otherwise be guessing whether something works."
            ),
            {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python source code to execute.",
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "Hard timeout in seconds (default: 300).",
                    },
                },
                "required": ["code"],
            },
        ),
    ]
