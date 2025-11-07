"""
Utility functions for working with generated code files.
"""

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_python_code(content: str) -> str:
    """Extract Python code from markdown fenced code blocks with fallback patterns.

    Tries old pattern first (handles most cases), then falls back to new pattern
    if compilation fails (e.g., when ``` appears inside strings).

    Args:
        content: Text content that may contain ```python code blocks

    Returns:
        Extracted Python code, or original content if no code blocks found
    """
    # Try old pattern first (handles most cases)
    pattern_old = r'```python\s*(.*?)\s*```'
    matches = re.findall(pattern_old, content, re.DOTALL | re.IGNORECASE)

    if matches:
        code = "\n\n".join(matches).strip()
        # Try to compile to check for syntax errors
        try:
            compile(code, '<string>', 'exec')
            logger.debug("Extracted code with old pattern, compilation succeeded")
            return code
        except SyntaxError:
            # Old pattern resulted in syntax error, try new pattern
            logger.debug("Code from old pattern failed compilation, trying new pattern")
            pattern_new = r'```python\s*\n(.*?)\n^```'
            matches_new = re.findall(pattern_new, content, re.DOTALL | re.MULTILINE | re.IGNORECASE)
            if matches_new:
                code_new = "\n\n".join(matches_new).strip()
                logger.debug("Extracted code with new pattern")
                return code_new
            # If new pattern also fails to find, return the code from old pattern anyway
            return code

    logger.debug("No fenced code blocks found, returning original content")
    return content.strip()


def strip_header_from_code(code_path: Path) -> str:
    """Read code file and strip the header lines based on metadata.

    Args:
        code_path: Path to the .py code file

    Returns:
        Clean code without headers
    """
    # Read the code
    with open(code_path, 'r') as f:
        code_text = f.read()

    # Read metadata to get number of header lines
    metadata_path = code_path.with_suffix('.json')
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        num_header_lines = metadata.get("num_header_lines", 0)
    else:
        num_header_lines = 0

    if num_header_lines == 0:
        return code_text

    # Strip header lines
    lines = code_text.split('\n')
    return '\n'.join(lines[num_header_lines:])
