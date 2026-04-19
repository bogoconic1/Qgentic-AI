"""Unit tests for the DeveloperAgent class."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents import developer
from agents.developer import DeveloperAgent


def _fake_llm_response(text: str) -> SimpleNamespace:
    """Mimic the shape of a genai text response with a single text-only part."""
    text_part = SimpleNamespace(text=text)
    content = SimpleNamespace(parts=[text_part])
    candidate = SimpleNamespace(content=content)
    return SimpleNamespace(text=text, candidates=[candidate])


def _valid_script() -> str:
    """A script that writes train_stats.json with a finite score next to itself."""
    return (
        "import logging\n"
        "logging.basicConfig(level=logging.INFO)\n"
        "import json\n"
        "from pathlib import Path\n"
        "out_dir = Path(__file__).parent\n"
        "out_dir.mkdir(parents=True, exist_ok=True)\n"
        "(out_dir / 'train_stats.json').write_text(json.dumps({'score': 0.5}))\n"
        "logging.info('wrote train_stats.json')\n"
    )


def _stats_less_script() -> str:
    """A script that runs cleanly but does NOT write train_stats.json."""
    return (
        "import logging\n"
        "logging.basicConfig(level=logging.INFO)\n"
        "logging.info('intentionally skipped train_stats.json')\n"
    )


@pytest.fixture
def fake_pipeline(monkeypatch, tmp_path):
    """Patch task_root, the LLM, executor, guardrails, and stack-trace enrichment."""
    monkeypatch.setattr(developer, "_TASK_ROOT", tmp_path / "task")

    captured: dict = {
        "codegen_calls": 0,
        "executed": [],
        "system_prompts": [],
        "script_factory": lambda k: _valid_script(),
    }

    def fake_call_llm(*, model, system_instruction, messages, text_format=None, **kwargs):
        captured["codegen_calls"] += 1
        captured["system_prompts"].append(system_instruction)
        k = captured["codegen_calls"]
        code = captured["script_factory"](k)
        response = _fake_llm_response(f"Here is the script:\n```python\n{code}\n```\n")
        return (response, 0) if kwargs.get("include_usage") else response

    monkeypatch.setattr(developer, "call_llm", fake_call_llm)

    def fake_execute_with_monitor(
        code_path, *, timeout_seconds, log_monitor_interval, logger, conda_env=None
    ):
        captured["executed"].append(Path(code_path))
        code_text = Path(code_path).read_text()
        exec_globals: dict = {"__name__": "__main__", "__file__": str(code_path)}
        try:
            exec(compile(code_text, str(code_path), "exec"), exec_globals)
        except Exception as exc:  # pragma: no cover
            return f"Traceback (most recent call last):\n  ...\n{type(exc).__name__}: {exc}"
        return "Script completed\n"

    monkeypatch.setattr(developer, "execute_with_monitor", fake_execute_with_monitor)

    def fake_evaluate_guardrails(**kwargs):
        return {
            "decision": "proceed",
            "logging_check": {},
            "leakage_check": {},
            "safety_check": {},
        }

    monkeypatch.setattr(developer, "evaluate_guardrails", fake_evaluate_guardrails)

    monkeypatch.setattr(
        developer,
        "web_search_stack_trace",
        lambda query: query + "\nThis is how you can fix the error: \nmocked hint",
    )

    return captured


def test_success_on_first_attempt(fake_pipeline, tmp_path):
    """Happy path — first attempt writes train_stats.json, DeveloperAgent returns success."""
    dev = DeveloperAgent(
        slug="test-slug",
        run_id="r1",
        dev_iter=1,
        goal_text="Produce a finite score.",
    )
    payload = dev.run(idea=None)

    assert payload["status"] == "success"
    assert payload["summary"]["attempts_made"] == 1
    assert payload["summary"]["score"] == 0.5
    assert payload["summary"]["final_error"] is None

    base = tmp_path / "task" / "test-slug" / "r1" / "developer_1"
    assert (base / "1" / "train.py").exists()
    assert (base / "1" / "train_stats.json").exists()
    assert (base / "1" / "train.txt").exists()
    assert payload["code_path"] == str(base / "1" / "train.py")


def test_retries_until_stats_written(fake_pipeline, tmp_path):
    """First attempt is valid-but-silent; second writes train_stats.json; loop succeeds at k=2."""

    def factory(k: int) -> str:
        return _stats_less_script() if k == 1 else _valid_script()

    fake_pipeline["script_factory"] = factory

    dev = DeveloperAgent(
        slug="test-slug",
        run_id="r1",
        dev_iter=1,
        goal_text="Produce a finite score.",
    )
    payload = dev.run(idea=None)

    assert payload["status"] == "success"
    assert payload["summary"]["attempts_made"] == 2

    base = tmp_path / "task" / "test-slug" / "r1" / "developer_1"
    assert (base / "1" / "train.py").exists()  # failed attempt kept
    assert not (base / "1" / "train_stats.json").exists()
    assert (base / "2" / "train.py").exists()
    assert (base / "2" / "train_stats.json").exists()
    assert payload["code_path"].endswith("developer_1/2/train.py")


def test_guardrails_block_counts_as_failed_attempt(monkeypatch, fake_pipeline, tmp_path):
    """A guardrails block → attempt counted, retried on next <k>."""
    state = {"count": 0}

    def block_then_proceed(**kwargs):
        state["count"] += 1
        if state["count"] == 1:
            return {
                "decision": "block",
                "logging_check": {
                    "status": "fail",
                    "violations": [{"line": 1, "reason": "bad order"}],
                },
                "leakage_check": {},
                "safety_check": {"decision": "proceed"},
            }
        return {
            "decision": "proceed",
            "logging_check": {},
            "leakage_check": {},
            "safety_check": {},
        }

    monkeypatch.setattr(developer, "evaluate_guardrails", block_then_proceed)

    dev = DeveloperAgent(
        slug="test-slug",
        run_id="r1",
        dev_iter=1,
        goal_text="Produce a finite score.",
    )
    payload = dev.run(idea=None)

    assert payload["status"] == "success"
    assert payload["summary"]["attempts_made"] == 2

    base = tmp_path / "task" / "test-slug" / "r1" / "developer_1"
    assert (base / "1" / "guardrails_block.txt").exists()
    assert not (base / "1" / "train.txt").exists()  # never executed
    assert (base / "2" / "train_stats.json").exists()


def test_previous_code_threaded_into_system_prompt(fake_pipeline, tmp_path):
    """dev_iter=2 with a prior successful developer_1/<k>/ should inline previous_code."""
    # Seed a prior successful run
    prev_attempt = tmp_path / "task" / "test-slug" / "r1" / "developer_1" / "1"
    prev_attempt.mkdir(parents=True)
    (prev_attempt / "train.py").write_text("# marker-prior-code\nprint('prev')\n")
    (prev_attempt / "train_stats.json").write_text(json.dumps({"score": 0.3}))

    dev = DeveloperAgent(
        slug="test-slug",
        run_id="r1",
        dev_iter=2,
        goal_text="Improve the prior run.",
    )
    payload = dev.run(idea="try a bigger model")

    assert payload["status"] == "success"
    # The system prompt from the first codegen call should contain the previous train.py
    assert len(fake_pipeline["system_prompts"]) >= 1
    system_prompt = fake_pipeline["system_prompts"][0]
    assert "<previous_code>" in system_prompt
    assert "marker-prior-code" in system_prompt
    assert "<idea>" in system_prompt
    assert "try a bigger model" in system_prompt
