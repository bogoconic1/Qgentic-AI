"""Unit tests for the standalone goal-mode developer agent."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents import goal_developer
from agents.goal_developer import HistoryEntry, run_goal_mode
from schemas.goal_developer import GoalReview


def _fake_llm_response(text: str) -> SimpleNamespace:
    """Mimic the shape of a genai text response.

    The new tool-loop in `_generate_code` reads `response.candidates[0].content.parts`
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
    captured = {"codegen_calls": 0, "review_calls": 0, "executed": []}

    def fake_call_llm(*, model, system_instruction, messages, text_format=None, **kwargs):
        # Codegen path uses call_llm; the review path now bypasses it and
        # calls the Anthropic SDK directly, so this mock only handles codegen.
        captured["codegen_calls"] += 1
        version = captured["codegen_calls"]
        version_dir = tmp_path / "run" / f"v{version}"
        return _fake_llm_response(
            "Here is the script:\n```python\n" + _hello_script(version_dir) + "```\n"
        )

    def fake_review(goal_text, code, output):
        captured["review_calls"] += 1
        return captured["review_factory"](captured["review_calls"])

    monkeypatch.setattr(goal_developer, "call_llm", fake_call_llm)
    monkeypatch.setattr(goal_developer, "_review", fake_review)

    def fake_execute_with_monitor(code_path, *, timeout_seconds, log_monitor_interval, logger, conda_env=None):
        # Actually run the script so out.txt is materialised — exercises the real subprocess too slowly.
        # We just simulate it: invoke the script via exec() inside this process.
        captured["executed"].append(Path(code_path))
        code_text = Path(code_path).read_text()
        exec_globals: dict = {"__name__": "__main__", "__file__": str(code_path)}
        try:
            exec(compile(code_text, str(code_path), "exec"), exec_globals)
        except Exception as exc:  # pragma: no cover - surfaced via train.txt
            return f"[exec failed: {exc}]"
        return "wrote out.txt\n"

    monkeypatch.setattr(goal_developer, "execute_with_monitor", fake_execute_with_monitor)

    # Disable guardrails (the dummy script trivially passes them, but we want determinism).
    def fake_evaluate_guardrails(**kwargs):
        return {"decision": "proceed", "logging_check": {}, "leakage_check": {}, "safety_check": {}}

    monkeypatch.setattr(goal_developer, "evaluate_guardrails", fake_evaluate_guardrails)

    captured["review_factory"] = lambda n: GoalReview(
        reasoning=f"v{n}: looks good",
        is_valid=True,
        violations=[],
        score=0.0,
        done=True,
        next_step="",
    )
    return captured


def test_terminates_on_done(fake_pipeline, tmp_path):
    """Loop should exit after one iteration when the reviewer says done."""
    history = run_goal_mode(
        goal_text="Write `hello` to out.txt",
        run_dir=tmp_path / "run",
        max_versions=10,
    )
    assert len(history) == 1
    assert history[0].review is not None and history[0].review.done is True
    assert (tmp_path / "run" / "v1" / "train.py").exists()
    assert (tmp_path / "run" / "v1" / "review.json").exists()
    assert (tmp_path / "run" / "v1" / "out.txt").read_text() == "hello"


def test_iterates_until_max_versions(fake_pipeline, tmp_path):
    """Loop should run max_versions iterations when the reviewer never says done."""
    fake_pipeline["review_factory"] = lambda n: GoalReview(
        reasoning=f"v{n}: not yet",
        is_valid=True,
        violations=[],
        score=1.0,
        done=False,
        next_step="keep going",
    )
    history = run_goal_mode(
        goal_text="Write `hello` to out.txt",
        run_dir=tmp_path / "run",
        max_versions=3,
    )
    assert len(history) == 3
    assert all(entry.review and entry.review.done is False for entry in history)


def test_review_failure_yields_degraded_entry(fake_pipeline, tmp_path):
    """When the review LLM raises, the loop synthesises a degraded GoalReview and continues."""
    calls = {"n": 0}

    def flaky_factory(n):
        calls["n"] = n
        if n == 1:
            raise RuntimeError("review service unreachable")
        return GoalReview(
            reasoning="recovered",
            is_valid=True,
            violations=[],
            score=0.0,
            done=True,
            next_step="",
        )

    fake_pipeline["review_factory"] = flaky_factory

    history = run_goal_mode(
        goal_text="Write `hello` to out.txt",
        run_dir=tmp_path / "run",
        max_versions=3,
    )
    assert len(history) == 2
    assert history[0].review is not None
    assert history[0].review.violations == ["review_call_error"]
    assert history[0].review.is_valid is False
    assert history[1].review is not None and history[1].review.done is True


def test_guardrails_block_skips_execution(monkeypatch, fake_pipeline, tmp_path):
    """A guardrails block should skip execution and review, and append a fix-it message."""
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

    monkeypatch.setattr(goal_developer, "evaluate_guardrails", block_then_proceed)

    history = run_goal_mode(
        goal_text="Write `hello` to out.txt",
        run_dir=tmp_path / "run",
        max_versions=3,
    )
    # v1 was blocked, v2 ran and the reviewer marked it done.
    assert len(history) == 2
    assert history[0].review is None  # blocked, never reviewed
    assert (tmp_path / "run" / "v1" / "guardrails_block.txt").exists()
    assert (tmp_path / "run" / "v1" / "train.txt").exists() is False
    assert history[1].review is not None and history[1].review.done is True
    assert fake_pipeline["executed"] == [tmp_path / "run" / "v2" / "train.py"]


def test_history_json_persisted(fake_pipeline, tmp_path):
    """history.json should round-trip the recorded entries."""
    run_goal_mode(
        goal_text="Write `hello` to out.txt",
        run_dir=tmp_path / "run",
        max_versions=10,
    )
    history_path = tmp_path / "run" / "history.json"
    assert history_path.exists()
    payload = json.loads(history_path.read_text())
    assert isinstance(payload, list) and len(payload) == 1
    assert payload[0]["version"] == 1
    assert payload[0]["review"]["done"] is True
