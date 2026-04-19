"""Unit tests for the standalone developer agent."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents import developer
from agents.developer import HistoryEntry, run_developer


def _fake_llm_response(text: str) -> SimpleNamespace:
    """Mimic the shape of a genai text response.

    The tool-loop in `_generate_code` reads `response.candidates[0].content.parts`
    to check for function calls before falling back to `response.text`. We model the
    response as having a single text-only part (no `function_call` attribute) so the
    loop's `has_function_calls` check correctly evaluates to False and the loop returns
    the extracted code.
    """
    text_part = SimpleNamespace(text=text)  # intentionally has no `function_call` attr
    content = SimpleNamespace(parts=[text_part])
    candidate = SimpleNamespace(content=content)
    return SimpleNamespace(text=text, candidates=[candidate])


def _hello_script(version_dir: Path) -> str:
    return (
        "import logging\n"
        "logging.basicConfig(level=logging.INFO)\n"
        "from pathlib import Path\n"
        f"Path({str(version_dir)!r}).mkdir(parents=True, exist_ok=True)\n"
        f"Path({str(version_dir)!r}, 'out.txt').write_text('hello')\n"
        "logging.info('wrote out.txt')\n"
    )


@pytest.fixture
def fake_pipeline(monkeypatch, tmp_path):
    """Patch the LLM, log monitor, and guardrails so the loop can run offline."""
    captured = {"codegen_calls": 0, "executed": []}

    def fake_call_llm(*, model, system_instruction, messages, text_format=None, **kwargs):
        captured["codegen_calls"] += 1
        version = captured["codegen_calls"]
        version_dir = tmp_path / "run" / f"v{version}"
        return _fake_llm_response(
            "Here is the script:\n```python\n" + _hello_script(version_dir) + "```\n"
        )

    monkeypatch.setattr(developer, "call_llm", fake_call_llm)

    def fake_execute_with_monitor(code_path, *, timeout_seconds, log_monitor_interval, logger, conda_env=None):
        captured["executed"].append(Path(code_path))
        code_text = Path(code_path).read_text()
        exec_globals: dict = {"__name__": "__main__", "__file__": str(code_path)}
        try:
            exec(compile(code_text, str(code_path), "exec"), exec_globals)
        except Exception as exc:  # pragma: no cover - surfaced via train.txt
            return f"[exec failed: {exc}]"
        return "wrote out.txt\n"

    monkeypatch.setattr(developer, "execute_with_monitor", fake_execute_with_monitor)

    # Disable guardrails (the dummy script trivially passes them, but we want determinism).
    def fake_evaluate_guardrails(**kwargs):
        return {"decision": "proceed", "logging_check": {}, "leakage_check": {}, "safety_check": {}}

    monkeypatch.setattr(developer, "evaluate_guardrails", fake_evaluate_guardrails)

    return captured


def test_iterates_until_max_versions(fake_pipeline, tmp_path):
    """Loop should run exactly ``max_versions`` iterations (review is noop, no early exit)."""
    history = run_developer(
        goal_text="Write `hello` to out.txt",
        run_dir=tmp_path / "run",
        max_versions=3,
    )
    assert len(history) == 3
    for entry in history:
        assert isinstance(entry, HistoryEntry)
        assert entry.review is not None  # noop returns empty Review(), not None


def test_guardrails_block_skips_execution(monkeypatch, fake_pipeline, tmp_path):
    """A guardrails block should skip execution and append a fix-it message."""
    blocks = {"count": 0}

    def block_then_proceed(**kwargs):
        blocks["count"] += 1
        if blocks["count"] == 1:
            return {
                "decision": "block",
                "logging_check": {"status": "fail", "violations": [{"line": 1, "reason": "bad order"}]},
                "leakage_check": {},
                "safety_check": {"decision": "proceed"},
            }
        return {"decision": "proceed", "logging_check": {}, "leakage_check": {}, "safety_check": {}}

    monkeypatch.setattr(developer, "evaluate_guardrails", block_then_proceed)

    history = run_developer(
        goal_text="Write `hello` to out.txt",
        run_dir=tmp_path / "run",
        max_versions=2,
    )
    assert len(history) == 2
    assert history[0].review is None  # blocked, no review
    assert (tmp_path / "run" / "v1" / "guardrails_block.txt").exists()
    assert (tmp_path / "run" / "v1" / "train.txt").exists() is False
    assert history[1].review is not None  # proceeded, noop review
    assert fake_pipeline["executed"] == [tmp_path / "run" / "v2" / "train.py"]


def test_history_json_persisted(fake_pipeline, tmp_path):
    """history.json should round-trip the recorded entries."""
    run_developer(
        goal_text="Write `hello` to out.txt",
        run_dir=tmp_path / "run",
        max_versions=2,
    )
    history_path = tmp_path / "run" / "history.json"
    assert history_path.exists()
    payload = json.loads(history_path.read_text())
    assert isinstance(payload, list) and len(payload) == 2
    assert payload[0]["version"] == 1
    assert payload[0]["review"] == {}  # empty Review() serialises to empty dict
