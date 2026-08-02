"""Tests for the Deep Research sub-agent.

The agent itself is a `codex exec` subprocess call, so the only logic worth
testing is the event translator: codex's JSONL stream in, agent chat records
out. The fixture is captured from real `codex exec --json` runs rather than
hand-written, so it keeps failing if codex's event schema drifts.

The records asserted here are the shape `MainAgent._log` writes. They are not
yet asserted against `scripts.viewer.parser`, which still expects the
pre-#304 `content.parts[]` form and raises `AttributeError` on both agents'
logs; add that round-trip once the parser is fixed.
"""

from __future__ import annotations

import json
from pathlib import Path

from agents.researcher import translate_events


_FIXTURE = Path(__file__).parent / "fixtures" / "codex_events.jsonl"


def _translate(tmp_path) -> list[dict]:
    chat_log = tmp_path / "researcher_chat.jsonl"
    translate_events(_FIXTURE, chat_log)
    return [json.loads(line) for line in chat_log.read_text().splitlines()]


def test_codex_items_become_main_agent_shaped_records(tmp_path):
    records = _translate(tmp_path)

    assert [r["role"] for r in records] == [
        "tool",       # web_search
        "assistant",  # agent_message
        "tool",       # command_execution
        "tool",
        "tool",
        "tool",       # file_change
        "tool",
        "assistant",
    ]

    for record in records:
        assert record["ts"]
        if record["role"] == "assistant":
            assert isinstance(record["content"], str)
            assert record["function_calls"] == []
        else:
            assert isinstance(record["args"], dict)
            assert isinstance(record["result"], str)


def test_search_queries_and_shell_output_reach_the_transcript(tmp_path):
    tools = [r for r in _translate(tmp_path) if r["role"] == "tool"]

    search = next(t for t in tools if t["name"] == "web_research")
    assert search["args"] == {"query": "2026 FIFA World Cup winner"}

    # aggregated_output is the only record of what a command printed — codex
    # keeps no other copy once the process exits.
    bash = [t for t in tools if t["name"] == "bash"]
    assert any("wc -l" in t["args"]["command"] for t in bash)
    assert any(t["result"].strip() for t in bash)
