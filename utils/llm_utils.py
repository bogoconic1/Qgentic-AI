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
        types.FunctionDeclaration(
            name="read_research_paper",
            description="Read and summarize academic research papers from arXiv relevant to the machine learning task. Returns structured markdown summary with Abstract, Introduction, Related Work, Method/Architecture, Experiments/Results, and Conclusion sections.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "arxiv_link": {
                        "type": "string",
                        "description": "ArXiv paper link (e.g., 'https://arxiv.org/pdf/2510.22916' or just '2510.22916')",
                    }
                },
                "required": ["arxiv_link"],
            },
        ),
        types.FunctionDeclaration(
            name="scrape_web_page",
            description="Scrape web pages and return markdown content from technical documentation, blog posts, and tutorials. Useful for accessing domain-specific knowledge, implementation guides, or best practices that complement academic papers.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The webpage URL to scrape (e.g., 'https://developer.nvidia.com/blog/...')",
                    }
                },
                "required": ["url"],
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
# Library investigator tools (read-only, scoped to site-packages)
# ---------------------------------------------------------------------------


def get_library_investigator_tools():
    """Get tools for the library investigator sub-agent.

    Four read-only tools scoped to the venv's site-packages directory.
    """
    return [
        types.FunctionDeclaration(
            name="list_packages",
            description="List all installed Python packages in the virtual environment's site-packages directory. Returns package directory names (excludes dist-info, egg-info, and private directories).",
            parameters_json_schema={
                "type": "object",
                "properties": {},
            },
        ),
        types.FunctionDeclaration(
            name="read_file",
            description="Read a Python source file from site-packages. Path is relative to site-packages (e.g., 'transformers/trainer.py'). Returns numbered lines. Files over 300 lines are truncated unless start_line/end_line are specified.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to site-packages (e.g., 'trl/trainer/sft_trainer.py')",
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
            description="Find files matching a glob pattern within a package directory. For example, glob_files('transformers', '**/*trainer*.py') finds all trainer-related files.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "package": {
                        "type": "string",
                        "description": "Package directory name (e.g., 'transformers', 'trl')",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (e.g., '**/*.py', '**/sft*.py')",
                    },
                },
                "required": ["package", "pattern"],
            },
        ),
        types.FunctionDeclaration(
            name="grep_code",
            description="Search for a regex pattern in Python source files within a package. Returns matching lines with file paths and line numbers.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "package": {
                        "type": "string",
                        "description": "Package directory name (e.g., 'transformers', 'xgboost')",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Regular expression pattern to search for (e.g., 'class SFTTrainer', 'def forward')",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of matching lines to return (default: 20)",
                    },
                },
                "required": ["package", "pattern"],
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Developer tools (investigate_library for code generation)
# ---------------------------------------------------------------------------


def get_developer_tools():
    """Get tools available to the DeveloperAgent during code generation."""
    return [
        types.FunctionDeclaration(
            name="investigate_library",
            description="Investigate the actual source code of an installed Python library to understand its API before writing code. A sub-agent will explore the library's source files in site-packages and return accurate constructor signatures, method signatures, and usage patterns. Use this when you need to understand how to use a library correctly.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query describing what you need to understand (e.g., 'How to use trl SFTTrainer with a Qwen3-4B model, including SFTConfig and data formatting')",
                    }
                },
                "required": ["query"],
            },
        ),
    ]
