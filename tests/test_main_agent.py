"""Unit tests for the MainAgent dispatch loop."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents import main_agent
from agents.main_agent import MainAgent


def _fake_fc(name: str, **args):
    """Build a response whose single part is a function_call matching the given name/args."""
    fc = SimpleNamespace(name=name, args=args)
    part = SimpleNamespace(function_call=fc)
    content = SimpleNamespace(parts=[part])
    content.model_dump = lambda **_: {
        "role": "model",
        "parts": [{"function_call": {"name": name, "args": dict(args)}}],
    }
    candidate = SimpleNamespace(content=content)
    return SimpleNamespace(candidates=[candidate])


def _fake_text(text: str):
    """Build a text-only response (no function_call attribute on the part)."""
    part = SimpleNamespace(text=text)  # intentionally has no function_call
    content = SimpleNamespace(parts=[part])
    content.model_dump = lambda **_: {"role": "model", "parts": [{"text": text}]}
    candidate = SimpleNamespace(content=content)
    return SimpleNamespace(candidates=[candidate])


@pytest.fixture
def patched_main_agent(monkeypatch, tmp_path):
    monkeypatch.setattr(main_agent, "_TASK_ROOT", tmp_path / "task")

    monkeypatch.setattr(
        main_agent,
        "execute_code",
        lambda filepath, timeout_seconds=None: SimpleNamespace(
            result=lambda: f"ran {Path(filepath).name}"
        ),
    )

    dev_calls: list = []

    class FakeDeveloperAgent:
        def __init__(self, slug, run_id, dev_iter, goal_text):
            dev_calls.append(
                {
                    "slug": slug,
                    "run_id": run_id,
                    "dev_iter": dev_iter,
                    "goal_text": goal_text,
                }
            )
            self.dev_iter = dev_iter

        def run(self, idea):
            dev_calls[-1]["idea"] = idea
            return {
                "status": "success",
                "code": "# fake",
                "code_path": f"dev_{self.dev_iter}",
                "summary": {
                    "score": 0.5,
                    "stats": {},
                    "stdout_tail": "",
                    "attempts_made": 1,
                    "final_error": None,
                },
            }

    monkeypatch.setattr(main_agent, "DeveloperAgent", FakeDeveloperAgent)

    research_calls: list = []

    class FakeResearcherAgent:
        def __init__(self, slug, run_id, research_iter, goal_text):
            research_calls.append(
                {
                    "slug": slug,
                    "run_id": run_id,
                    "research_iter": research_iter,
                    "goal_text": goal_text,
                }
            )

        def run(self, instruction):
            return f"# report\n\nResearched: {instruction}"

    monkeypatch.setattr(main_agent, "ResearcherAgent", FakeResearcherAgent)

    return {"dev_calls": dev_calls, "research_calls": research_calls}


def test_dispatches_each_tool(patched_main_agent, monkeypatch):
    agent = MainAgent(slug="test", run_id="r1", goal_text="do the thing")

    responses = iter(
        [
            _fake_fc("add_idea", title="try it", description="body"),
            _fake_fc("update_idea", idea_id=1, description="body v2"),
            _fake_fc("develop", idea_id=1),
            _fake_fc("research", instruction="look up X"),
            _fake_fc("analyze", code="print('hi')"),
            _fake_fc("remove_idea", idea_id=1),
        ]
    )
    monkeypatch.setattr(
        main_agent, "call_llm", lambda **kwargs: (next(responses), 0)
    )

    for _ in range(6):
        agent._step([])

    # developer + researcher subagents invoked once each with threaded goal_text
    assert len(patched_main_agent["dev_calls"]) == 1
    assert patched_main_agent["dev_calls"][0]["goal_text"] == "do the thing"
    assert patched_main_agent["dev_calls"][0]["dev_iter"] == 1
    # develop resolved idea_id=1 → full file body (includes the updated description "body v2")
    assert "body v2" in patched_main_agent["dev_calls"][0]["idea"]
    assert len(patched_main_agent["research_calls"]) == 1
    assert patched_main_agent["research_calls"][0]["research_iter"] == 1
    assert patched_main_agent["research_calls"][0]["goal_text"] == "do the thing"

    # idea pool mutated: add → update → remove leaves pool empty
    assert "try it" not in (agent.ideas_dir / "INDEX.md").read_text()

    # execute_python wrote and ran a snippet
    assert (agent.snippets_dir / "001.py").exists()
    assert "print('hi')" in (agent.snippets_dir / "001.py").read_text()

    # chat log captured every step (6 assistant turns + 6 tool results = 12 records)
    records = [json.loads(line) for line in agent.chat_log.read_text().splitlines()]
    assert len(records) == 12
    assert [r["name"] for r in records[1::2]] == [
        "add_idea",
        "update_idea",
        "develop",
        "research",
        "analyze",
        "remove_idea",
    ]


def test_text_only_response_is_logged_and_ignored(patched_main_agent, monkeypatch, caplog):
    agent = MainAgent(slug="test", run_id="r1", goal_text="do the thing")
    monkeypatch.setattr(
        main_agent, "call_llm", lambda **kwargs: (_fake_text("hello"), 0)
    )

    with caplog.at_level("WARNING"):
        agent._step([])

    assert any("text-only" in rec.message for rec in caplog.records)
    records = [json.loads(line) for line in agent.chat_log.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["role"] == "assistant"
