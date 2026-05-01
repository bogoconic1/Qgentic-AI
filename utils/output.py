"""Output truncation for LLM context windows.

Caps tool/execution output at a character limit before it enters the
conversation thread. When truncated, the full text is optionally persisted
to a file and the truncated version includes a reference so the LLM can
read more via ``read_file`` / ``grep_code`` if needed.
"""

from __future__ import annotations

from pathlib import Path


MAX_OUTPUT_CHARS = 30_000


def truncate_for_llm(
    text: str,
    full_output_path: Path | None = None,
    max_chars: int = MAX_OUTPUT_CHARS,
) -> str:
    """Cap *text* for LLM context. Persist full output to disk if over limit.

    If *full_output_path* is given and the file does not already exist, the
    full text is written there. If it already exists (e.g. the caller wrote
    ``train.txt`` before calling this), the write is skipped — the path is
    only used in the truncation reference message.
    """
    if len(text) <= max_chars:
        return text
    if full_output_path and not full_output_path.exists():
        full_output_path.parent.mkdir(parents=True, exist_ok=True)
        full_output_path.write_text(text)
    total_lines = len(text.splitlines())
    kept_lines = len(text[:max_chars].splitlines())
    truncated_lines = total_lines - kept_lines
    ref = f" Full output: {full_output_path}" if full_output_path else ""
    return text[:max_chars] + f"\n\n... [{truncated_lines} lines truncated.{ref}]"
