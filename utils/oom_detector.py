"""OOM detection utility for subprocess execution."""

import logging

logger = logging.getLogger(__name__)


class OOMError(Exception):
    """Raised when subprocess runs out of memory."""
    pass


def detect_oom(output: str) -> bool:
    """
    Detect if execution output indicates an out-of-memory error.

    Args:
        output: Combined stdout/stderr from execution

    Returns:
        True if OOM detected, False otherwise
    """
    if not output:
        return False

    output_lower = output.lower()

    # Common OOM patterns across different frameworks
    oom_patterns = [
        "cuda out of memory",
        "cuda error: out of memory",
        "torch.cuda.outofmemoryerror",
        "outofmemoryerror",
    ]

    for pattern in oom_patterns:
        if pattern in output_lower:
            logger.debug(f"OOM pattern detected: '{pattern}'")
            return True

    return False
