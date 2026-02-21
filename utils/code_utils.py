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

    Tries both patterns and selects the best extraction:
    1. Keep whichever extracts MORE code AND compiles successfully
    2. If only one compiles, use that one
    3. If neither compiles, use the longer extraction
    4. If neither extracts anything, return original content

    Args:
        content: Text content that may contain ```python code blocks

    Returns:
        Extracted Python code, or original content if no code blocks found
    """
    # Try old pattern (matches any ``` including in strings/comments)
    pattern_old = r'```python\s*(.*?)\s*```'
    matches_old = re.findall(pattern_old, content, re.DOTALL | re.IGNORECASE)
    code_old = "\n\n".join(matches_old).strip() if matches_old else ""

    # Try new pattern (only matches ``` at start of line)
    pattern_new = r'```python\s*\n(.*?)\n^```'
    matches_new = re.findall(pattern_new, content, re.DOTALL | re.MULTILINE | re.IGNORECASE)
    code_new = "\n\n".join(matches_new).strip() if matches_new else ""

    # If neither found anything, return original content
    if not code_old and not code_new:
        logger.debug("No fenced code blocks found, returning original content")
        return ""

    # Check compilation for both
    old_compiles = False
    new_compiles = False

    if code_old:
        try:
            compile(code_old, '<string>', 'exec')
            old_compiles = True
        except SyntaxError:
            pass

    if code_new:
        try:
            compile(code_new, '<string>', 'exec')
            new_compiles = True
        except SyntaxError:
            pass

    if new_compiles and old_compiles:
        if len(code_new) >= len(code_old):
            logger.debug(f"Both patterns compile, using new pattern ({len(code_new)} chars vs {len(code_old)} chars)")
            return code_new
        else:
            logger.debug(f"Both patterns compile, using old pattern ({len(code_old)} chars vs {len(code_new)} chars)")
            return code_old
    elif new_compiles:
        logger.debug(f"Only new pattern compiles, using new pattern ({len(code_new)} chars)")
        return code_new
    elif old_compiles:
        logger.debug(f"Only old pattern compiles, using old pattern ({len(code_old)} chars)")
        return code_old
    else:
        if len(code_new) >= len(code_old):
            logger.debug(f"Neither compiles, using new pattern as longer ({len(code_new)} chars vs {len(code_old)} chars)")
            return code_new
        else:
            logger.debug(f"Neither compiles, using old pattern as longer ({len(code_old)} chars vs {len(code_new)} chars)")
            return code_old


def strip_header_from_code(code_path: Path) -> str:
    """Read code file and strip the header lines based on metadata.

    Args:
        code_path: Path to the .py code file

    Returns:
        Clean code without headers
    """
    with open(code_path, 'r') as f:
        code_text = f.read()

    metadata_path = code_path.with_suffix('.json')
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        num_header_lines = metadata.get("num_header_lines", 0)
    else:
        num_header_lines = 0

    if num_header_lines == 0:
        return code_text

    lines = code_text.split('\n')
    return '\n'.join(lines[num_header_lines:])
