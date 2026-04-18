"""
LLM utility functions for Gemini response handling and tool definitions.
"""

import base64
import logging
import mimetypes
from pathlib import Path

from google.genai import types

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
    """
    Extract text from a Gemini response.

    Args:
        response: Google Gemini response object

    Returns:
        Extracted text string

    Examples:
        >>> text = extract_text_from_response(gemini_response)
    """
    return response.text


def append_message(role: str, message: str) -> dict:
    """
    Create a message in Gemini format.

    Args:
        role: Message role ("user", "assistant", "model", etc.)
        message: Message content (text string)

    Returns:
        Formatted message dict for Gemini

    Examples:
        >>> append_message("user", "Hello")
        {'role': 'user', 'parts': [{'text': 'Hello'}]}

        >>> append_message("assistant", "Hi")
        {'role': 'model', 'parts': [{'text': 'Hi'}]}
    """
    gemini_role = "model" if role == "assistant" else role
    return {"role": gemini_role, "parts": [{"text": message}]}


def get_tools():
    """
    Get tools as Gemini FunctionDeclaration objects.

    Returns:
        List of FunctionDeclaration objects
    """
    return [
        types.FunctionDeclaration(
            name="execute_python",
            description="Write and execute a Python script. The script runs in the task data directory with access to all data files, model outputs, and predictions. Print results to stdout.",
            parameters_json_schema={
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
    """Get monitor tools (execute_bash) as Gemini FunctionDeclaration objects."""
    return [
        types.FunctionDeclaration(
            name="execute_bash",
            description="Execute a bash command for system diagnostics. Use for checking GPU utilization (nvidia-smi), process status (ps, top), memory (free), disk (df), etc.",
            parameters_json_schema={
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
        types.FunctionDeclaration(
            name="web_research",
            description=(
                "Discover web pages for a query via Exa neural search. "
                "Returns a list of results, each with url, title, full page text, "
                "and published_date — not a snippet, the whole text. This is the "
                "ONLY way to discover URLs; never guess or reconstruct URLs."
            ),
            parameters_json_schema={
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
        types.FunctionDeclaration(
            name="web_fetch",
            description=(
                "Fetch a single URL's main content as markdown via Firecrawl. "
                "Full page content is returned — no truncation. Only call with "
                "URLs you got from a prior `web_research` result or from a "
                "markdown link inside a prior `web_fetch` result."
            ),
            parameters_json_schema={
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
        types.FunctionDeclaration(
            name="write_python_code",
            description=(
                "Write a Python script to the research scripts dir and execute "
                "it in a subprocess. Full stdout/stderr is returned. Use for "
                "EDA, API probing, quick computations — anything you'd run in "
                "a notebook to validate an idea."
            ),
            parameters_json_schema={
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
        types.FunctionDeclaration(
            name="read_file",
            description="Read a source file. Path is absolute or relative to the current working directory. Returns numbered lines. Files over 300 lines are truncated unless start_line/end_line are specified.",
            parameters_json_schema={
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
        types.FunctionDeclaration(
            name="glob_files",
            description="Find files matching a glob pattern under a root directory. Returns up to 50 matches as paths relative to the root.",
            parameters_json_schema={
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
        types.FunctionDeclaration(
            name="grep_code",
            description="Recursively search a regex pattern in files under a root directory. Returns matching lines with file paths and line numbers.",
            parameters_json_schema={
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
        types.FunctionDeclaration(
            name="list_dir",
            description="List the immediate children of a directory. Directories are suffixed with '/'.",
            parameters_json_schema={
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
        types.FunctionDeclaration(
            name="bash_readonly",
            description=(
                "Run a single read-only shell command. ONLY these commands are allowed: "
                "ls, cat, head, tail, wc, file, find, grep, tree, du, stat, "
                "git status, git log, git diff, git show, git blame, git ls-files, git ls-tree. "
                "Pipes (|), redirection (>, <, >>), command chaining (;, &&, ||), backticks, and $() are forbidden — "
                "use the dedicated read_file/glob_files/grep_code/list_dir tools instead."
            ),
            parameters_json_schema={
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


def get_developer_tools():
    """Get tools available to the DeveloperAgent during code generation."""
    return [
        types.FunctionDeclaration(
            name="explore_codebase",
            description=(
                "Ask a natural-language question about the codebase or installed Python "
                "packages and get back a markdown report with file:line citations. The "
                "sub-agent reads source files using read_file/glob_files/grep_code/list_dir "
                "across configured roots — it does NOT execute Python or shell commands. "
                "To verify whether a snippet runs, use `execute_python` instead.\n\n"
                "Use this for static investigation: 'How does X work?', 'What's the "
                "signature of Y?', 'Where is Z defined?', 'Show me callers of W'. Brief "
                "the sub-agent like a smart colleague who just walked into the room — it "
                "hasn't seen this conversation. Terse command-style prompts produce "
                "shallow, generic work."
            ),
            parameters_json_schema={
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
        types.FunctionDeclaration(
            name="execute_python",
            description=(
                "Run a Python snippet in a fresh subprocess and get back stdout, "
                "stderr, and the exit code. Use this to VERIFY behavior — test an "
                "import, probe an API, check a library version, prototype a function, "
                "run a quick computation. The snippet runs with a timeout. Use this "
                "tool when you would otherwise be guessing whether something works."
            ),
            parameters_json_schema={
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
