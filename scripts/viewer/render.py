"""Jinja-friendly rendering helpers (markdown, truncation, classification)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import markdown as _markdown
from markupsafe import Markup, escape

# 4 KB inline cap before "Show full" reveal.
INLINE_CAP = 4096
# Result/args bodies above this share of non-printable bytes are treated as binary.
BINARY_THRESHOLD = 0.30


@dataclass(frozen=True)
class TruncatedBody:
    head: str
    full: str
    truncated: bool
    full_size: int


def render_markdown(text: str) -> Markup:
    """Render markdown with fenced code, tables, and single-newline → <br>.

    `nl2br` matches how the agents (and most chat tools) write soft wraps:
    one newline ends a visible line. Code blocks and tables keep their own
    rules.
    """
    if not text:
        return Markup("")
    html = _markdown.markdown(text, extensions=["fenced_code", "tables", "nl2br"])
    return Markup(html)


def format_args_json(args: Any) -> str:
    return json.dumps(args, indent=2, ensure_ascii=False, sort_keys=True)


def truncate_body(text: str, cap: int = INLINE_CAP) -> TruncatedBody:
    truncated = len(text) > cap
    head = text[:cap] + ("…" if truncated else "")
    return TruncatedBody(head=head, full=text, truncated=truncated, full_size=len(text))


def is_binary_blob(text: str) -> tuple[bool, int]:
    """Return (is_binary, byte_size). Heuristic: non-printable ratio."""
    if not text:
        return False, 0
    sample = text[:8192]
    printable = sum(
        1 for ch in sample
        if ch == "\n" or ch == "\t" or ch == "\r" or (32 <= ord(ch) < 127) or ord(ch) > 127
    )
    ratio_nonprintable = 1.0 - (printable / len(sample))
    return ratio_nonprintable > BINARY_THRESHOLD, len(text)


_PRIMARY_TEXT_KEYS = ("output", "output_tail", "content", "report", "error")
_PRIMARY_LIST_KEYS = ("entries", "matches")


def _format_list_item(item: Any) -> str:
    if isinstance(item, str):
        return item
    return json.dumps(item, ensure_ascii=False)


def _extract_primary_body(text: str) -> tuple[str, list[tuple[str, Any]], bool]:
    """Inspect a tool result string and unwrap its primary payload.

    Returns (body_text, meta_pairs, is_error). body_text is "" for schemas
    whose values are all primitives (rendered as meta-only). Tools wrap their
    output in JSON like `{"output": "...", "returncode": 0, "truncated": false}`;
    we unwrap a known textual or list key and surface the rest as key=value pills.
    """
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text, [], False

    for key in _PRIMARY_TEXT_KEYS:
        if isinstance(parsed.get(key), str):
            meta = [(k, parsed[k]) for k in parsed if k != key]
            return parsed[key], meta, key == "error"

    for key in _PRIMARY_LIST_KEYS:
        value = parsed.get(key)
        if isinstance(value, list):
            body = "\n".join(_format_list_item(item) for item in value)
            meta = [(k, parsed[k]) for k in parsed if k != key]
            return body, meta, False

    return "", list(parsed.items()), False


def classify_result(text: str) -> dict[str, Any]:
    """Bundle render-ready info for a tool result string."""
    raw_size = len(text)
    binary, _ = is_binary_blob(text)
    if binary:
        return {
            "binary": True,
            "size": raw_size,
            "body": None,
            "meta": [],
            "is_error": False,
        }
    body_text, meta, is_error = _extract_primary_body(text)
    return {
        "binary": False,
        "size": raw_size,
        "body": truncate_body(body_text),
        "meta": meta,
        "is_error": is_error,
    }


def fmt_meta_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (str, int, float)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def fmt_bytes(n: int | None) -> str:
    if n is None:
        return "0 B"
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def escape_pre(text: str) -> Markup:
    return Markup(f"<pre>{escape(text)}</pre>")


def fmt_mtime(epoch: float) -> str:
    dt = datetime.fromtimestamp(epoch, tz=timezone.utc).astimezone()
    return dt.strftime("%Y-%m-%d %H:%M")


def bash_command(name: str, args: dict[str, Any]) -> str | None:
    """Return the bash `command` string iff this is a bash call carrying one.

    The agents' bash tools take a `command` string field; pretty-printing the
    full args dict double-escapes shell quoting and is unreadable. Render the
    command directly when we can.
    """
    if name != "bash":
        return None
    cmd = args.get("command")
    return cmd if isinstance(cmd, str) else None


def args_without_command(args: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in args.items() if k != "command"}
