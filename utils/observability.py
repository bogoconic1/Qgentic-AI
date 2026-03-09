"""Observability helpers backed by Langfuse.

Provides a tiny compatibility layer so modules can use:
- @op() decorator for traced functions
- init(project_name=...) during startup
- finish() during shutdown
"""

from __future__ import annotations

import logging
from functools import wraps
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

try:
    from langfuse import Langfuse
    from langfuse.decorators import observe
except Exception:  # pragma: no cover - graceful fallback if optional dep missing
    Langfuse = None
    observe = None


_langfuse_client: Optional[Any] = None


def init(project_name: Optional[str] = None) -> None:
    """Initialise Langfuse client.

    `project_name` is accepted for backward compatibility with previous setup.
    """

    global _langfuse_client

    if Langfuse is None:
        logger.warning("Langfuse is not available; observability disabled.")
        return

    if _langfuse_client is not None:
        return

    try:
        # Langfuse config is read from env vars; keep project_name as a no-op arg.
        _langfuse_client = Langfuse()
        logger.info("Langfuse observability initialized")
    except Exception:
        logger.exception("Failed to initialize Langfuse; observability disabled.")
        _langfuse_client = None


def finish() -> None:
    """Flush and shutdown Langfuse client."""

    global _langfuse_client

    if _langfuse_client is None:
        return

    try:
        _langfuse_client.flush()
    except Exception:
        logger.exception("Failed to flush Langfuse traces")
    finally:
        _langfuse_client = None


def op(*args: Any, **kwargs: Any):
    """Decorator compatible with `@weave.op()` usage."""

    if observe is None:
        # No-op fallback preserving call signature for both @op and @op(...)
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(func)
            def _wrapped(*f_args: Any, **f_kwargs: Any) -> Any:
                return func(*f_args, **f_kwargs)

            return _wrapped

        return _decorator

    # Map to Langfuse's observe decorator.
    return observe(*args, **kwargs)
