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


def test_tool_record(tmp_path: Path):
    p = tmp_path / "x.jsonl"
    _write(
        p,
        {
            "role": "tool",
            "name": "read_file",
            "args": {"path": "/x"},
            "result": '{"content": "hi"}',
            "ts": "2026-05-02T07:25:01+00:00",
        },
    )

    [rec] = list(parser.iter_records(p))
    assert isinstance(rec, parser.ToolRecord)
    assert rec.name == "read_file"
    assert rec.args == {"path": "/x"}
    assert rec.result == '{"content": "hi"}'
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
    text = (
        json.dumps({"role": "tool", "name": "a", "args": {}, "result": "1", "ts": "t1"})
        + "\n"
    )
    text += '{"role": "tool", "name": "b", "args"'  # truncated
    p.write_text(text)

    records = list(parser.iter_records(p))
    assert len(records) == 1
    assert isinstance(records[0], parser.ToolRecord)
    assert records[0].name == "a"


def test_unknown_role_yields_raw(tmp_path: Path):
    p = tmp_path / "x.jsonl"
    _write(p, {"role": "system", "content": "x", "ts": "t"})
    [rec] = list(parser.iter_records(p))
    assert isinstance(rec, parser.RawRecord)
    assert "system" in rec.error


def test_blank_lines_are_ignored(tmp_path: Path):
    p = tmp_path / "x.jsonl"
    p.write_text(
        "\n"
        + json.dumps({"role": "tool", "name": "a", "args": {}, "result": "1"})
        + "\n\n\n"
    )
    records = list(parser.iter_records(p))
    assert len(records) == 1
