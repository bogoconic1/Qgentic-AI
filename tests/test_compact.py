"""Unit tests for utils.compact."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from utils import compact
from utils.compact import compact_messages, should_compact


@pytest.fixture(autouse=True)
def configure_thresholds(monkeypatch):
    """All tests run against a deterministic threshold + keep-last."""
    monkeypatch.setattr(
        compact,
        "get_config_value",
        lambda *keys: {
            ("runtime", "compaction_threshold_tokens"): 100,
            ("runtime", "compaction_keep_last"): 4,
        }[keys],
    )


def _msg(role: str, text: str = "x") -> dict:
    return {"role": role, "parts": [{"text": text}]}


def test_should_compact_threshold_boundary():
    assert should_compact(None) is False
    assert should_compact(50) is False
    assert should_compact(100) is False  # strictly greater than
    assert should_compact(101) is True


def test_should_compact_raises_when_threshold_unset(monkeypatch):
    monkeypatch.setattr(compact, "get_config_value", lambda *_: None)
    with pytest.raises(RuntimeError, match="compaction_threshold_tokens"):
        should_compact(999)


def test_compact_messages_short_input_is_noop(monkeypatch):
    """When ≤ keep_last entries exist, the list is returned unchanged."""
    called = []
    monkeypatch.setattr(
        compact, "call_llm", lambda **kw: called.append(kw) or SimpleNamespace(text="x")
    )
    msgs = [_msg("user"), _msg("model"), _msg("function"), _msg("user")]
    out = compact_messages(msgs, model="gemini-3.1-pro-preview")
    assert out == msgs
    assert called == []  # no summariser call when nothing to summarise


def test_compact_messages_summarises_and_keeps_last_n(monkeypatch):
    """Happy path: 8-message list → [summary_user, last 4]."""
    monkeypatch.setattr(
        compact,
        "call_llm",
        lambda **kw: SimpleNamespace(
            text="<analysis>thinking…</analysis><summary>S1\nS2</summary>"
        ),
    )
    msgs = [
        _msg("user", "u1"),
        _msg("model", "m1"),
        _msg("function", "f1"),
        _msg("user", "u2"),
        _msg("model", "m2"),
        _msg("function", "f2"),
        _msg("user", "u3"),
        _msg("model", "m3"),
    ]
    out = compact_messages(msgs, model="gemini-3.1-pro-preview")

    # cut = 8 - 4 = 4; kept = msgs[4:] (4 entries); output = [summary, *kept].
    assert len(out) == 5
    assert out[0]["role"] == "user"
    assert "Summary:\nS1\nS2" in out[0]["parts"][0]["text"]
    assert out[1:] == msgs[4:]


def test_compact_messages_handles_backslash_escapes_in_summary(monkeypatch):
    """Regression: summary body containing \\d / \\w / \\n must not crash.

    `re.sub` used to receive the summary body as a replacement template and
    choke on any `\\<letter>` sequence with `re.error: bad escape`. The LLM's
    summary can legitimately contain such sequences (regex explanations,
    Windows paths, LaTeX). Use `.replace` instead.
    """
    troubling_body = r"regex: \d+\w+ and Windows path C:\Users\x; LaTeX \dfrac{a}{b}"
    monkeypatch.setattr(
        compact,
        "call_llm",
        lambda **kw: SimpleNamespace(
            text=f"<summary>{troubling_body}</summary>"
        ),
    )
    msgs = [
        _msg("user", "u1"),
        _msg("model", "m1"),
        _msg("function", "f1"),
        _msg("user", "u2"),
        _msg("model", "m2"),
        _msg("function", "f2"),
        _msg("user", "u3"),
        _msg("model", "m3"),
    ]

    out = compact_messages(msgs, model="gemini-3.1-pro-preview")

    assert out[0]["role"] == "user"
    # The raw backslash sequences survive into the summary body verbatim.
    assert r"\d+" in out[0]["parts"][0]["text"]
    assert r"C:\Users\x" in out[0]["parts"][0]["text"]
    assert r"\dfrac" in out[0]["parts"][0]["text"]


def test_compact_messages_summarises_even_when_kept_starts_on_non_user(monkeypatch):
    """MainAgent shape: only one user (index 0); kept may start on any role."""
    called = []
    monkeypatch.setattr(
        compact,
        "call_llm",
        lambda **kw: called.append(kw) or SimpleNamespace(text="<summary>ok</summary>"),
    )
    msgs = [
        _msg("user", "u1"),
        _msg("model", "m1"),
        _msg("function", "f1"),
        _msg("model", "m2"),
        _msg("function", "f2"),
    ]
    out = compact_messages(msgs, model="gemini-3.1-pro-preview")

    # cut = 5 - 4 = 1; kept = msgs[1:] (starts with model — no walk-back).
    assert len(called) == 1
    assert len(out) == 5
    assert out[0]["role"] == "user"
    assert out[1:] == msgs[1:]
