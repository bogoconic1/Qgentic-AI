"""Unit tests for utils.compact."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from utils import compact
from utils.compact import compact_messages, should_compact


@pytest.fixture(autouse=True)
def configure_thresholds(monkeypatch):
    monkeypatch.setattr(
        compact,
        "get_config_value",
        lambda *keys: {
            ("runtime", "compaction_threshold_tokens"): 100,
            ("runtime", "compaction_keep_last"): 4,
        }[keys],
    )


def _msg(role: str, text: str = "x") -> dict:
    return {"role": role, "content": text}


def test_should_compact_threshold_boundary():
    assert should_compact(None) is False
    assert should_compact(50) is False
    assert should_compact(100) is False
    assert should_compact(101) is True


def test_should_compact_raises_when_threshold_unset(monkeypatch):
    monkeypatch.setattr(compact, "get_config_value", lambda *_: None)
    with pytest.raises(RuntimeError, match="compaction_threshold_tokens"):
        should_compact(999)


def test_compact_messages_short_input_is_noop(monkeypatch):
    called = []
    monkeypatch.setattr(
        compact,
        "call_llm",
        lambda **kw: called.append(kw) or SimpleNamespace(output_text="x"),
    )
    msgs = [_msg("user"), _msg("assistant"), _msg("user"), _msg("assistant")]
    out = compact_messages(msgs, model="gpt-5.5")
    assert out == msgs
    assert called == []


def test_compact_messages_summarises_and_keeps_last_n(monkeypatch):
    monkeypatch.setattr(
        compact,
        "call_llm",
        lambda **kw: SimpleNamespace(
            output_text="<analysis>thinking…</analysis><summary>S1\nS2</summary>"
        ),
    )
    msgs = [
        _msg("user", "u1"),
        _msg("assistant", "m1"),
        _msg("user", "f1"),
        _msg("user", "u2"),
        _msg("assistant", "m2"),
        _msg("user", "f2"),
        _msg("user", "u3"),
        _msg("assistant", "m3"),
    ]
    out = compact_messages(msgs, model="gpt-5.5")

    assert len(out) == 5
    assert out[0]["role"] == "user"
    assert "Summary:\nS1\nS2" in out[0]["content"]
    assert out[1:] == msgs[4:]


def test_compact_messages_handles_backslash_escapes_in_summary(monkeypatch):
    troubling_body = r"regex: \d+\w+ and Windows path C:\Users\x; LaTeX \dfrac{a}{b}"
    monkeypatch.setattr(
        compact,
        "call_llm",
        lambda **kw: SimpleNamespace(
            output_text=f"<summary>{troubling_body}</summary>"
        ),
    )
    msgs = [
        _msg("user", "u1"),
        _msg("assistant", "m1"),
        _msg("user", "f1"),
        _msg("user", "u2"),
        _msg("assistant", "m2"),
        _msg("user", "f2"),
        _msg("user", "u3"),
        _msg("assistant", "m3"),
    ]

    out = compact_messages(msgs, model="gpt-5.5")

    assert out[0]["role"] == "user"
    assert r"\d+" in out[0]["content"]
    assert r"C:\Users\x" in out[0]["content"]
    assert r"\dfrac" in out[0]["content"]


def test_compact_messages_summarises_even_when_kept_starts_on_non_user(monkeypatch):
    called = []
    monkeypatch.setattr(
        compact,
        "call_llm",
        lambda **kw: (
            called.append(kw) or SimpleNamespace(output_text="<summary>ok</summary>")
        ),
    )
    msgs = [
        _msg("user", "u1"),
        _msg("assistant", "m1"),
        _msg("user", "f1"),
        _msg("assistant", "m2"),
        _msg("user", "f2"),
    ]
    out = compact_messages(msgs, model="gpt-5.5")

    assert len(called) == 1
    assert len(out) == 5
    assert out[0]["role"] == "user"
    assert out[1:] == msgs[1:]
