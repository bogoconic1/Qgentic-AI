"""Parse agent chat JSONL files into typed records.

Live writes from agents append to these files without holding a reader-side
lock, so the *last* line may be partial. We swallow that case silently and
only surface mid-file malformations as `RawRecord`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class AssistantPart:
    text: str | None
    function_call: dict[str, Any] | None
    has_thought: bool
    thought_size: int


@dataclass(frozen=True)
class AssistantRecord:
    ts: str | None
    parts: list[AssistantPart] = field(default_factory=list)


@dataclass(frozen=True)
class ToolRecord:
    ts: str | None
    name: str
    args: dict[str, Any]
    result: str


@dataclass(frozen=True)
class RawRecord:
    ts: str | None
    line: str
    error: str


Record = AssistantRecord | ToolRecord | RawRecord


def iter_records(path: Path) -> Iterator[Record]:
    """Yield parsed records from a chat JSONL file.

    - Empty / missing file → empty iterator.
    - Mid-file malformed line → yields `RawRecord` and continues.
    - Last non-empty line malformed → silently dropped (live-write race).
    """
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8", errors="replace")

    raw_lines = text.split("\n")
    # Drop trailing pure-empty lines so "last" is the last meaningful line.
    while raw_lines and raw_lines[-1] == "":
        raw_lines.pop()

    last_idx = len(raw_lines) - 1
    for idx, line in enumerate(raw_lines):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            if idx == last_idx:
                # Live-write race: agent is mid-append. Swallow silently.
                return
            yield RawRecord(ts=None, line=line, error=str(exc))
            continue

        ts = obj.get("ts")
        role = obj.get("role")
        if role == "assistant":
            yield _parse_assistant(obj, ts)
        elif role == "tool":
            yield ToolRecord(
                ts=ts,
                name=obj["name"],
                args=obj["args"],
                result=obj["result"],
            )
        else:
            yield RawRecord(ts=ts, line=line, error=f"unknown role: {role!r}")


def _parse_assistant(obj: dict[str, Any], ts: str | None) -> AssistantRecord:
    content = obj["content"]
    parts: list[AssistantPart] = []
    for raw in content.get("parts", []):
        thought = raw.get("thought_signature")
        parts.append(
            AssistantPart(
                text=raw.get("text"),
                function_call=raw.get("function_call"),
                has_thought=bool(thought),
                thought_size=len(thought) if thought else 0,
            )
        )
    return AssistantRecord(ts=ts, parts=parts)
