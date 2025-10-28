"""
Utility functions for working with generated code files.
"""

import json
from pathlib import Path


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
