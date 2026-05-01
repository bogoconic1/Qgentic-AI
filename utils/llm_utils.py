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
# Deep Research tools (web_research + web_fetch)
# ---------------------------------------------------------------------------


def get_deep_research_tools():
    """Inner tools available to the Deep Research sub-agent.

    The two research-specific tools (web_research / web_fetch) plus the
    shared filesystem tools (read_file / glob_files / grep_code / list_dir /
    bash). Use ``bash`` for any scripted execution (`python -c "..."` or
    `python script.py`).
    """
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
        *get_filesystem_tools(),
    ]


# ---------------------------------------------------------------------------
# Filesystem + explore tools (scoped to runtime.explore_allowed_roots)
# ---------------------------------------------------------------------------


def get_filesystem_tools():
    """Five Linux-style tools (read/glob/grep/list/bash) shared across subagents.

    The four read-only tools (`read_file`, `glob_files`, `grep_code`,
    `list_dir`) reject paths outside the configured allowed roots. The
    fifth tool, `bash`, is widened: it goes through `bash -c` so pipes,
    redirection, and chaining all work — but every command is run past
    an LLM safety judge first. The judge blocks destructive operations
    (`rm -rf /`, `dd` to a disk, fork bombs, pipe-to-shell, writes to
    system paths, etc.).
    """
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
            name="bash",
            description=(
                "Run a shell command via `bash -c`. Pipes, redirection, chaining, "
                "backticks, and $() all work. An LLM safety judge inspects every "
                "command first and blocks destructive operations (rm -rf /, dd "
                "against block devices, mkfs, fork bombs, pipe-to-shell, writes "
                "to /etc/usr/lib/boot/sys/proc/dev/var/root, force-pushes, "
                "shutdown/reboot, etc.). Use this for the long tail of operations "
                "the dedicated tools don't cover: `cp`, `mv`, `mkdir`, `rm` of "
                "project files, `tar`, `pip install`, `python script.py | tee log`."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to run (passed verbatim to `bash -c`).",
                    },
                },
                "required": ["command"],
            },
        ),
        types.FunctionDeclaration(
            name="write_file",
            description=(
                "Write a file to the local filesystem. Creates parent "
                "directories as needed and overwrites any existing file at "
                "`path`. Prefer `edit_file` for in-place modifications since "
                "it only sends the diff; use `write_file` for new files or "
                "full rewrites. Path must resolve under one of the configured "
                "allowed roots."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path or path relative to cwd.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full file content to write.",
                    },
                },
                "required": ["path", "content"],
            },
        ),
        types.FunctionDeclaration(
            name="edit_file",
            description=(
                "Perform an exact-string replacement inside a file. "
                "`old_string` must match a unique substring; if the file "
                "contains more than one occurrence, set `replace_all` to "
                "true. Curly quotes are normalized so a straight-quote "
                "`old_string` matches typographic content. To create a new "
                "file pass an empty `old_string` and the desired content as "
                "`new_string`. Path must resolve under one of the configured "
                "allowed roots."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path or path relative to cwd.",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "Substring to replace; must be unique unless replace_all is true. Empty string + missing file = create.",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement text. Empty string deletes old_string.",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace every occurrence of old_string instead of failing on multiple matches (default: false).",
                    },
                },
                "required": ["path", "old_string", "new_string"],
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Main Agent tools
# ---------------------------------------------------------------------------


def get_main_agent_tools():
    """Get the tool palette available to the Main Agent."""
    return [
        types.FunctionDeclaration(
            name="develop",
            description=(
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
                "— that is always wrong; it belongs here."
            ),
            parameters_json_schema={
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
        types.FunctionDeclaration(
            name="research",
            description=(
                "Runs one Deep Research iteration: web_fetch + web_search + internal "
                "Python to produce a markdown report answering the instruction. Use "
                "for domain grounding, library docs, prior-art sweeps, empirical "
                "sniff-tests on the dataset."
            ),
            parameters_json_schema={
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
        types.FunctionDeclaration(
            name="add_idea",
            description="Add a new entry to the idea pool. Returns the assigned integer id.",
            parameters_json_schema={
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
        types.FunctionDeclaration(
            name="remove_idea",
            description="Remove an idea from the pool by id.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "idea_id": {"type": "integer"},
                },
                "required": ["idea_id"],
            },
        ),
        types.FunctionDeclaration(
            name="update_idea",
            description="Replace the body of an existing idea. Title stays the same.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "idea_id": {"type": "integer"},
                    "description": {"type": "string"},
                },
                "required": ["idea_id", "description"],
            },
        ),
        *get_filesystem_tools(),
    ]


