"""Tests for scripts.viewer.parser."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.viewer import parser


def _write(path: Path, *records: dict | str) -> None:
    lines: list[str] = []
    for r in records:
        lines.append(r if isinstance(r, str) else json.dumps(r))
    path.write_text("\n".join(lines) + "\n")


def test_iter_records_empty_file(tmp_path: Path):
    p = tmp_path / "x.jsonl"
    p.write_text("")
    assert list(parser.iter_records(p)) == []


def test_iter_records_missing_file(tmp_path: Path):
    assert list(parser.iter_records(tmp_path / "nope.jsonl")) == []


def test_assistant_record_with_text_function_call_and_thought(tmp_path: Path):
    p = tmp_path / "x.jsonl"
    _write(p, {
        "role": "assistant",
        "content": {"parts": [
            {"text": "hello"},
            {
                "function_call": {"id": "c1", "name": "read_file", "args": {"path": "/x"}},
                "thought_signature": "A" * 64,
            },
        ]},
        "ts": "2026-05-02T07:25:00+00:00",
    })

    records = list(parser.iter_records(p))
    assert len(records) == 1
    rec = records[0]
    assert isinstance(rec, parser.AssistantRecord)
    assert rec.ts == "2026-05-02T07:25:00+00:00"
    assert len(rec.parts) == 2

    text_part, fc_part = rec.parts
    assert text_part.text == "hello"
    assert text_part.function_call is None
    assert text_part.has_thought is False

    assert fc_part.text is None
    assert fc_part.function_call == {"id": "c1", "name": "read_file", "args": {"path": "/x"}}
    assert fc_part.has_thought is True
    assert fc_part.thought_size == 64


def test_tool_record(tmp_path: Path):
    p = tmp_path / "x.jsonl"
    _write(p, {
        "role": "tool",
        "name": "read_file",
        "args": {"path": "/x"},
        "result": "{\"content\": \"hi\"}",
        "ts": "2026-05-02T07:25:01+00:00",
    })

    [rec] = list(parser.iter_records(p))
    assert isinstance(rec, parser.ToolRecord)
    assert rec.name == "read_file"
    assert rec.args == {"path": "/x"}
    assert rec.result == "{\"content\": \"hi\"}"
    assert rec.ts == "2026-05-02T07:25:01+00:00"


def test_mid_file_malformed_yields_raw_then_continues(tmp_path: Path):
    p = tmp_path / "x.jsonl"
    _write(
        p,
        {"role": "tool", "name": "a", "args": {}, "result": "1", "ts": "t1"},
        "{not json}",
        {"role": "tool", "name": "b", "args": {}, "result": "2", "ts": "t2"},
    )

    records = list(parser.iter_records(p))
    assert len(records) == 3
    assert isinstance(records[0], parser.ToolRecord) and records[0].name == "a"
    assert isinstance(records[1], parser.RawRecord)
    assert records[1].line == "{not json}"
    assert "Expecting" in records[1].error or "delim" in records[1].error
    assert isinstance(records[2], parser.ToolRecord) and records[2].name == "b"


def test_last_line_malformed_dropped_silently(tmp_path: Path):
    # Simulate live-write race: prior records are valid, last line truncated mid-write.
    p = tmp_path / "x.jsonl"
    text = json.dumps({"role": "tool", "name": "a", "args": {}, "result": "1", "ts": "t1"}) + "\n"
    text += '{"role": "tool", "name": "b", "args"'  # truncated
    p.write_text(text)

    records = list(parser.iter_records(p))
    assert len(records) == 1
    assert isinstance(records[0], parser.ToolRecord)
    assert records[0].name == "a"


def test_missing_ts_renders_as_none(tmp_path: Path):
    p = tmp_path / "x.jsonl"
    _write(p, {
        "role": "tool",
        "name": "a",
        "args": {},
        "result": "ok",
    })
    [rec] = list(parser.iter_records(p))
    assert rec.ts is None


def test_unknown_role_yields_raw(tmp_path: Path):
    p = tmp_path / "x.jsonl"
    _write(p, {"role": "system", "content": "x", "ts": "t"})
    [rec] = list(parser.iter_records(p))
    assert isinstance(rec, parser.RawRecord)
    assert "system" in rec.error


def test_assistant_with_no_parts(tmp_path: Path):
    p = tmp_path / "x.jsonl"
    _write(p, {"role": "assistant", "content": {}, "ts": "t"})
    [rec] = list(parser.iter_records(p))
    assert isinstance(rec, parser.AssistantRecord)
    assert rec.parts == []


def test_blank_lines_are_ignored(tmp_path: Path):
    p = tmp_path / "x.jsonl"
    p.write_text(
        "\n"
        + json.dumps({"role": "tool", "name": "a", "args": {}, "result": "1"})
        + "\n\n\n"
    )
    records = list(parser.iter_records(p))
    assert len(records) == 1
