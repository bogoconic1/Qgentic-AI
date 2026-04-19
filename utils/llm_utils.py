"""LLM utility functions — response helpers + MainAgent tool declarations."""

from __future__ import annotations

import base64
import logging
import mimetypes
from pathlib import Path

from google.genai import types


logger = logging.getLogger(__name__)


def encode_image_to_data_url(
    image_path: str | Path, max_size_bytes: int = 4_500_000
) -> str:
    """Encode an image file to a data URL: ``data:{mime};base64,{data}``."""
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
            f"Image {path.name} exceeds size limit "
            f"({len(image_bytes) / 1024 / 1024:.2f}MB > {max_size_bytes / 1024 / 1024:.2f}MB)"
        )

    data = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{data}"


def append_message(role: str, message: str) -> dict:
    """Create a Gemini-format message dict.

    ``role='assistant'`` is translated to ``'model'`` (Gemini's wire term).
    """
    gemini_role = "model" if role == "assistant" else role
    return {"role": gemini_role, "parts": [{"text": message}]}


def get_main_agent_tools():
    """Full tool palette exposed to MainAgent (11 tools).

    Union of every tool the four former agents exposed, minus the sub-agent
    wrapper tools themselves. MainAgent's dispatcher routes each call to a
    helper in ``tools/runtime.py``, ``tools/filesystem.py``,
    ``tools/research_net.py``, or ``utils/idea_pool.py``.
    """
    return [
        types.FunctionDeclaration(
            name="execute_python",
            description=(
                "Write a Python snippet to `main_agent_snippets/<filename>` and "
                "run it in a fresh subprocess. You pick the filename — e.g. "
                "'probe_task_012.py', 'build_task001.py', 'validate_all.py' — "
                "so you can re-run it later, evolve it, or reference it in "
                "logs. Must end with `.py`; no directory separators (no '/', "
                "no '..'); re-using a filename overwrites the previous version. "
                "Returns combined stdout/stderr (long output is truncated — the "
                "full file stays on disk). On non-zero exit, stderr is "
                "auto-enriched with a web-searched remediation hint. Use for "
                "EDA, probes, authoring per-task builders (the snippet can "
                "write files via `Path.write_text`), running an ONNX-"
                "construction script, validating a candidate network, etc."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": (
                            "Filename ending in '.py'. Saved under "
                            "`main_agent_snippets/`. Re-using a name overwrites "
                            "the previous script."
                        ),
                    },
                    "code": {
                        "type": "string",
                        "description": "Complete Python source to execute.",
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "Hard timeout in seconds (default: 300).",
                    },
                },
                "required": ["filename", "code"],
            },
        ),
        types.FunctionDeclaration(
            name="read_file",
            description=(
                "Read a source file. Path is absolute or relative to cwd. Returns "
                "numbered lines. Files over 300 lines are truncated unless "
                "start_line/end_line are specified."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path or path relative to cwd.",
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
            description=(
                "Find files matching a glob pattern under a root directory. "
                "Returns up to 50 matches as paths relative to the root."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "root": {
                        "type": "string",
                        "description": "Directory under one of the allowed roots to glob from.",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (e.g., '**/*.py').",
                    },
                },
                "required": ["root", "pattern"],
            },
        ),
        types.FunctionDeclaration(
            name="grep_code",
            description=(
                "Recursively search a regex pattern in files under a root "
                "directory. Returns matching lines with file paths and line numbers."
            ),
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
                        "description": "File glob to restrict the search (default '*.py').",
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
            description=(
                "List the immediate children of a directory. Directories are "
                "suffixed with '/'."
            ),
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
                "Run an arbitrary shell command via `bash -c`. No allowlist — "
                "use for `cp` / `mv` / `mkdir` / pipes / redirection / `zip` / "
                "`unzip` / anything else the session needs. Output is capped "
                "at 8 KB; use `execute_python` if you need larger output or "
                "structured return values. Timeout: 600 seconds."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to run.",
                    },
                },
                "required": ["command"],
            },
        ),
        types.FunctionDeclaration(
            name="web_research",
            description=(
                "Discover web pages for a query via Exa neural search. Returns a "
                "list of results, each with url, title, full page text, and "
                "published_date — not a snippet, the whole text. This is the "
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
                        "description": "Optional number of results to return.",
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
    ]
