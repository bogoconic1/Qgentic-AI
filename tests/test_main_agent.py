"""Unit tests for the MainAgent dispatch loop."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from openrouter.components.chatassistantmessage import ChatAssistantMessage
from openrouter.components.chatchoice import ChatChoice
from openrouter.components.chatresult import ChatResult
from openrouter.components.chattoolcall import ChatToolCall, ChatToolCallFunction

from agents import main_agent
from agents.main_agent import MainAgent


def _chat_response(message: ChatAssistantMessage, finish_reason: str) -> ChatResult:
    return ChatResult(
        choices=[ChatChoice(finish_reason=finish_reason, index=0, message=message)],
        created=0,
        id="chatcmpl_test",
        model="deepseek/deepseek-v4-pro",
        object="chat.completion",
        system_fingerprint=None,
    )


def _tool_call(name: str, args: dict, call_id: str) -> ChatToolCall:
    return ChatToolCall(
        function=ChatToolCallFunction(name=name, arguments=json.dumps(args)),
        id=call_id,
        type="function",
    )


def _fake_fc(name: str, **args) -> ChatResult:
    """Build a response whose single tool call matches the given name/args."""
    message = ChatAssistantMessage(
        role="assistant",
        tool_calls=[_tool_call(name, dict(args), "call_0")],
    )
    return _chat_response(message, "tool_calls")


def _fake_text(text: str) -> ChatResult:
    """Build a text-only response."""
    return _chat_response(
        ChatAssistantMessage(role="assistant", content=text),
        "stop",
    )


def _fake_multi(*calls: tuple[str, dict]) -> ChatResult:
    """Build a response whose tool calls are emitted in the given order."""
    tool_calls = [
        _tool_call(name, dict(args), f"call_{idx}")
        for idx, (name, args) in enumerate(calls)
    ]
    return _chat_response(
        ChatAssistantMessage(role="assistant", tool_calls=tool_calls),
        "tool_calls",
    )


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
        def __init__(self, slug, run_id, dev_iter):
            dev_calls.append(
                {
                    "slug": slug,
                    "run_id": run_id,
                    "dev_iter": dev_iter,
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
        def __init__(self, slug, run_id, research_iter):
            research_calls.append(
                {
                    "slug": slug,
                    "run_id": run_id,
                    "research_iter": research_iter,
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

    # developer + researcher subagents invoked once each
    assert len(patched_main_agent["dev_calls"]) == 1
    assert patched_main_agent["dev_calls"][0]["dev_iter"] == 1
    # develop resolved idea_id=1 → full file body (includes the updated description "body v2")
    assert "body v2" in patched_main_agent["dev_calls"][0]["idea"]
    assert len(patched_main_agent["research_calls"]) == 1
    assert patched_main_agent["research_calls"][0]["research_iter"] == 1

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


def test_baseline_develop_passes_goal_as_idea(patched_main_agent, monkeypatch):
    """`develop()` with no idea_id falls through to idea=<goal_text>."""
    agent = MainAgent(slug="test", run_id="r1", goal_text="the session goal body")
    responses = iter([_fake_fc("develop")])
    monkeypatch.setattr(
        main_agent, "call_llm", lambda **kwargs: (next(responses), 0)
    )

    agent._step([])

    assert len(patched_main_agent["dev_calls"]) == 1
    assert patched_main_agent["dev_calls"][0]["idea"] == "the session goal body"


def test_parallel_dispatch_preserves_order(patched_main_agent, monkeypatch):
    """Multiple tool calls in one turn execute and keep original ordering."""
    agent = MainAgent(slug="test", run_id="r1", goal_text="do the thing")

    # Seed the idea pool with two ideas the LLM can develop.
    agent._dispatch("add_idea", {"title": "alpha", "description": "idea-alpha"})
    agent._dispatch("add_idea", {"title": "beta", "description": "idea-beta"})

    # Single LLM turn returning three parallel calls in a specific order.
    response = _fake_multi(
        ("analyze", {"code": "print('first')"}),
        ("develop", {"idea_id": 1}),
        ("develop", {"idea_id": 2}),
    )
    monkeypatch.setattr(main_agent, "call_llm", lambda **kwargs: (response, 0))

    agent._step([])

    # Both develop calls ran; each got its own dev_iter.
    assert len(patched_main_agent["dev_calls"]) == 2
    assert sorted(c["dev_iter"] for c in patched_main_agent["dev_calls"]) == [1, 2]

    # Tool result messages are appended in the same order as the model tool calls.
    emitted_tool_ids = [m["tool_call_id"] for m in agent.input_list[-3:]]
    assert emitted_tool_ids == ["call_0", "call_1", "call_2"]
    assert [m["role"] for m in agent.input_list[-3:]] == ["tool", "tool", "tool"]

    # Chat log: assistant turn + 3 tool records in the original order.
    records = [json.loads(line) for line in agent.chat_log.read_text().splitlines()]
    tool_records = [r for r in records if r["role"] == "tool"]
    assert [r["name"] for r in tool_records] == ["analyze", "develop", "develop"]

    # Snippet counter advanced exactly once for the single `analyze` call.
    assert (agent.snippets_dir / "001.py").exists()
    assert not (agent.snippets_dir / "002.py").exists()


def test_parallel_analyze_calls_get_distinct_snippet_numbers(
    patched_main_agent, monkeypatch
):
    """Two analyze calls in the same turn must not collide on the snippet filename."""
    agent = MainAgent(slug="test", run_id="r1", goal_text="do the thing")

    response = _fake_multi(
        ("analyze", {"code": "print('a')"}),
        ("analyze", {"code": "print('b')"}),
        ("analyze", {"code": "print('c')"}),
    )
    monkeypatch.setattr(main_agent, "call_llm", lambda **kwargs: (response, 0))

    agent._step([])

    for n in (1, 2, 3):
        assert (agent.snippets_dir / f"{n:03d}.py").exists()
    contents = {
        (agent.snippets_dir / f"{n:03d}.py").read_text() for n in (1, 2, 3)
    }
    # All three snippets kept their distinct bodies (no overwrite).
    assert any("print('a')" in c for c in contents)
    assert any("print('b')" in c for c in contents)
    assert any("print('c')" in c for c in contents)


def test_stuck_nudge_fires_after_repeated_identical_calls(
    patched_main_agent, monkeypatch
):
    """If the same single-call turn repeats N times, MainAgent appends a static
    "push on / never stop" nudge to input_list and clears its history.

    Regression for #257 — the live failure showed `analyze(time.sleep(2))` repeated
    623 times in a row.
    """
    agent = MainAgent(slug="test", run_id="r1", goal_text="do the thing")

    response = _fake_fc("analyze", code="import time\ntime.sleep(2)\nprint('Waiting...')")
    monkeypatch.setattr(main_agent, "call_llm", lambda **kwargs: (response, 0))

    threshold = main_agent._STUCK_REPEAT_THRESHOLD
    nudge_text = main_agent._STUCK_NUDGE

    def _has_nudge(messages: list[dict]) -> bool:
        for msg in messages:
            if msg.get("role") != "user":
                continue
            if nudge_text in msg["content"]:
                return True
        return False

    # First (threshold-1) repeats: no nudge yet — pattern not confirmed.
    for _ in range(threshold - 1):
        agent._step([])
    assert not _has_nudge(agent.input_list)

    # The threshold'th identical turn triggers the nudge.
    agent._step([])
    last = agent.input_list[-1]
    assert last["role"] == "user"
    assert nudge_text in last["content"]

    # History reset, so the next identical turn does NOT immediately re-nudge.
    agent._step([])
    assert agent.input_list[-1]["role"] == "tool"


def test_stuck_nudge_does_not_fire_for_varied_calls(patched_main_agent, monkeypatch):
    """A turn whose tool args differ from the previous resets the stuck window."""
    agent = MainAgent(slug="test", run_id="r1", goal_text="do the thing")

    threshold = main_agent._STUCK_REPEAT_THRESHOLD
    nudge_text = main_agent._STUCK_NUDGE

    responses = iter(
        [_fake_fc("analyze", code=f"print({i})") for i in range(threshold + 2)]
    )
    monkeypatch.setattr(
        main_agent, "call_llm", lambda **kwargs: (next(responses), 0)
    )

    for _ in range(threshold + 2):
        agent._step([])

    for msg in agent.input_list:
        if msg.get("role") != "user":
            continue
        assert nudge_text not in msg["content"]


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
