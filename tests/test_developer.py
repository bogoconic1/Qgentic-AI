"""Unit tests for the rewritten DeveloperAgent."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agents import developer
from agents.developer import DeveloperAgent, _SOLUTION_PY_SCAFFOLD


def _fake_text(text: str):
    """Build a text-only response (no function_call attribute on the part)."""
    part = SimpleNamespace(text=text)
    content = SimpleNamespace(parts=[part])
    content.model_dump = lambda **_: {"role": "model", "parts": [{"text": text}]}
    candidate = SimpleNamespace(content=content)
    return SimpleNamespace(candidates=[candidate])


def _fake_fc(name: str, **args):
    """Build a single-function-call response."""
    fc = SimpleNamespace(name=name, args=args)
    part = SimpleNamespace(function_call=fc)
    content = SimpleNamespace(parts=[part])
    content.model_dump = lambda **_: {
        "role": "model",
        "parts": [{"function_call": {"name": name, "args": dict(args)}}],
    }
    candidate = SimpleNamespace(content=content)
    return SimpleNamespace(candidates=[candidate])


@pytest.fixture
def patched_developer(monkeypatch, tmp_path):
    monkeypatch.setattr(developer, "_TASK_ROOT", tmp_path / "task")
    return tmp_path


def test_init_creates_solution_py_scaffold(patched_developer):
    """Constructing DeveloperAgent scaffolds SOLUTION.py with the logging stanza."""
    dev = DeveloperAgent(slug="test", run_id="r1", dev_iter=0)

    assert dev.solution_py_path == dev.base_dir / "SOLUTION.py"
    assert dev.solution_py_path.exists()
    assert dev.solution_py_path.read_text(encoding="utf-8") == _SOLUTION_PY_SCAFFOLD


def test_init_does_not_clobber_existing_solution_py(patched_developer):
    """Re-instantiating with the same dev_iter leaves a populated SOLUTION.py untouched."""
    base_dir = developer._TASK_ROOT / "test" / "r1" / "developer_v0"
    base_dir.mkdir(parents=True, exist_ok=True)
    populated = "import logging\nlogging.basicConfig()\nimport torch\n"
    (base_dir / "SOLUTION.py").write_text(populated, encoding="utf-8")

    dev = DeveloperAgent(slug="test", run_id="r1", dev_iter=0)

    assert dev.solution_py_path.read_text(encoding="utf-8") == populated


def test_run_dispatches_run_solution_success(patched_developer, monkeypatch):
    """run_solution success threads score/stats/elapsed into the return shape."""
    dev = DeveloperAgent(slug="test", run_id="r1", dev_iter=0)

    responses = iter([_fake_fc("run_solution"), _fake_text("done")])
    monkeypatch.setattr(
        developer, "call_llm", lambda **kwargs: (next(responses), 0)
    )
    fake_result = json.dumps(
        {
            "success": True,
            "score": 0.42,
            "stats": {"score": 0.42, "n": 100},
            "elapsed_seconds": 12.3,
            "output_tail": "log tail...",
        }
    )
    monkeypatch.setattr(
        developer, "tool_run_solution", lambda version_dir: fake_result
    )

    result = dev.run(idea="produce a baseline")

    assert set(result.keys()) == {"status", "version_dir", "summary", "report"}
    assert result["status"] == "success"
    assert result["version_dir"] == str(dev.base_dir)
    assert result["summary"] == {
        "score": 0.42,
        "stats": {"score": 0.42, "n": 100},
        "elapsed_seconds": pytest.approx(12.3),
        "runs_made": 1,
        "final_error": None,
    }
    assert result["report"] == "# produce a baseline\n"


def test_run_failed_run_solution_records_error_kind(patched_developer, monkeypatch):
    """A failed run_solution call records final_error and yields status=failed."""
    dev = DeveloperAgent(slug="test", run_id="r1", dev_iter=0)

    responses = iter([_fake_fc("run_solution"), _fake_text("done")])
    monkeypatch.setattr(
        developer, "call_llm", lambda **kwargs: (next(responses), 0)
    )
    monkeypatch.setattr(
        developer,
        "tool_run_solution",
        lambda version_dir: json.dumps(
            {
                "success": False,
                "error_kind": "guardrail_basicconfig",
                "violations": [{"line": 1, "reason": "..."}],
            }
        ),
    )

    result = dev.run(idea="test")

    assert result["status"] == "failed"
    assert result["summary"]["final_error"] == "guardrail_basicconfig"
    assert result["summary"]["runs_made"] == 1
    assert result["summary"]["score"] is None


def test_run_no_run_status_when_no_run_solution_called(patched_developer, monkeypatch):
    """Agent terminating without calling run_solution returns status='no_run'."""
    dev = DeveloperAgent(slug="test", run_id="r1", dev_iter=0)

    monkeypatch.setattr(
        developer, "call_llm", lambda **kwargs: (_fake_text("nothing to do"), 0)
    )

    result = dev.run(idea="explore-only")

    assert result["status"] == "no_run"
    assert result["summary"]["runs_made"] == 0
    assert result["summary"]["score"] is None
    assert result["summary"]["final_error"] is None


def test_run_loads_developer_instructions_when_present(patched_developer, monkeypatch):
    """DEVELOPER_INSTRUCTIONS.md at task/<slug>/ is threaded into build_system."""
    task_root = developer._TASK_ROOT
    (task_root / "test").mkdir(parents=True, exist_ok=True)
    (task_root / "test" / "DEVELOPER_INSTRUCTIONS.md").write_text(
        "Always use mixed precision.\n", encoding="utf-8"
    )

    captured = {}

    def fake_build_system(custom_instructions=None):
        captured["custom_instructions"] = custom_instructions
        return "system prompt"

    monkeypatch.setattr(developer, "build_system", fake_build_system)
    monkeypatch.setattr(
        developer, "call_llm", lambda **kwargs: (_fake_text("done"), 0)
    )

    dev = DeveloperAgent(slug="test", run_id="r1", dev_iter=0)
    dev.run(idea="test")

    assert "mixed precision" in captured["custom_instructions"]
