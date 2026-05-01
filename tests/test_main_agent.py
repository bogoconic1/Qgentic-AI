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


def _fake_multi(*calls: tuple[str, dict]):
    """Build a response whose parts are several function_calls in the given order."""
    parts = []
    dumped_parts = []
    for name, args in calls:
        fc = SimpleNamespace(name=name, args=dict(args))
        parts.append(SimpleNamespace(function_call=fc))
        dumped_parts.append({"function_call": {"name": name, "args": dict(args)}})
    content = SimpleNamespace(parts=parts)
    content.model_dump = lambda **_: {"role": "model", "parts": dumped_parts}
    candidate = SimpleNamespace(content=content)
    return SimpleNamespace(candidates=[candidate])


@pytest.fixture
def patched_main_agent(monkeypatch, tmp_path):
    monkeypatch.setattr(main_agent, "_TASK_ROOT", tmp_path / "task")

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
                "version_dir": f"task/test/r1/developer_v{self.dev_iter}",
                "summary": {
                    "score": 0.5,
                    "stats": {},
                    "elapsed_seconds": 0.0,
                    "runs_made": 1,
                    "final_error": None,
                },
                "report": f"# fake idea\n\nDev iter {self.dev_iter} fake report.\n",
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
            _fake_fc("remove_idea", idea_id=1),
        ]
    )
    monkeypatch.setattr(
        main_agent, "call_llm", lambda **kwargs: (next(responses), 0)
    )

    for _ in range(5):
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

    # chat log captured every step (5 assistant turns + 5 tool results = 10 records)
    records = [json.loads(line) for line in agent.chat_log.read_text().splitlines()]
    assert len(records) == 10
    assert [r["name"] for r in records[1::2]] == [
        "add_idea",
        "update_idea",
        "develop",
        "research",
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
    """Multiple function_calls in one turn execute and keep original ordering."""
    agent = MainAgent(slug="test", run_id="r1", goal_text="do the thing")

    # Seed the idea pool with two ideas the LLM can develop.
    agent._dispatch("add_idea", {"title": "alpha", "description": "idea-alpha"})
    agent._dispatch("add_idea", {"title": "beta", "description": "idea-beta"})

    # Filesystem tool calls are routed through execute_filesystem_tool — stub it.
    monkeypatch.setattr(
        main_agent,
        "execute_filesystem_tool",
        lambda name, args: json.dumps({"output": f"ran {name}", "returncode": 0}),
    )

    # Single LLM turn returning three parallel calls in a specific order.
    response = _fake_multi(
        ("read_file", {"path": "/tmp/seed.txt"}),
        ("develop", {"idea_id": 1}),
        ("develop", {"idea_id": 2}),
    )
    monkeypatch.setattr(main_agent, "call_llm", lambda **kwargs: (response, 0))

    agent._step([])

    # Both develop calls ran; each got its own dev_iter.
    assert len(patched_main_agent["dev_calls"]) == 2
    assert sorted(c["dev_iter"] for c in patched_main_agent["dev_calls"]) == [1, 2]

    # function_response Content appended to input_list matches function_call order.
    func_content = agent.input_list[-1]
    assert func_content["role"] == "function"
    emitted_names = [p["function_response"]["name"] for p in func_content["parts"]]
    assert emitted_names == ["read_file", "develop", "develop"]

    # Chat log: assistant turn + 3 tool records in the original order.
    records = [json.loads(line) for line in agent.chat_log.read_text().splitlines()]
    tool_records = [r for r in records if r["role"] == "tool"]
    assert [r["name"] for r in tool_records] == ["read_file", "develop", "develop"]


def test_stuck_nudge_fires_after_repeated_identical_calls(
    patched_main_agent, monkeypatch
):
    """If the same single-call turn repeats N times, MainAgent appends a static
    "push on / never stop" nudge to input_list and clears its history.

    Regression for #257 — a live failure showed the same no-op tool call
    repeated 623 times in a row.
    """
    agent = MainAgent(slug="test", run_id="r1", goal_text="do the thing")

    monkeypatch.setattr(
        main_agent,
        "execute_filesystem_tool",
        lambda name, args: json.dumps({"output": "ok", "returncode": 0}),
    )
    response = _fake_fc("read_file", path="/tmp/loop.txt")
    monkeypatch.setattr(main_agent, "call_llm", lambda **kwargs: (response, 0))

    threshold = main_agent._STUCK_REPEAT_THRESHOLD
    nudge_text = main_agent._STUCK_NUDGE

    def _has_nudge(messages: list[dict]) -> bool:
        for msg in messages:
            if msg.get("role") != "user":
                continue
            for part in msg.get("parts", []):
                if nudge_text in part.get("text", ""):
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
    assert nudge_text in last["parts"][0]["text"]

    # History reset, so the next identical turn does NOT immediately re-nudge.
    agent._step([])
    assert agent.input_list[-1]["role"] == "function"


def test_stuck_nudge_does_not_fire_for_varied_calls(patched_main_agent, monkeypatch):
    """A turn whose tool args differ from the previous resets the stuck window."""
    agent = MainAgent(slug="test", run_id="r1", goal_text="do the thing")

    threshold = main_agent._STUCK_REPEAT_THRESHOLD
    nudge_text = main_agent._STUCK_NUDGE

    monkeypatch.setattr(
        main_agent,
        "execute_filesystem_tool",
        lambda name, args: json.dumps({"output": "ok", "returncode": 0}),
    )
    responses = iter(
        [_fake_fc("read_file", path=f"/tmp/{i}.txt") for i in range(threshold + 2)]
    )
    monkeypatch.setattr(
        main_agent, "call_llm", lambda **kwargs: (next(responses), 0)
    )

    for _ in range(threshold + 2):
        agent._step([])

    for msg in agent.input_list:
        if msg.get("role") != "user":
            continue
        for part in msg.get("parts", []):
            assert nudge_text not in part.get("text", "")


def test_filesystem_tool_calls_route_to_filesystem_helpers(
    patched_main_agent, monkeypatch
):
    """A `list_dir` tool call from MainAgent is dispatched to tools.filesystem."""
    captured = {}

    def fake_execute_filesystem_tool(name, args):
        captured["name"] = name
        captured["args"] = args
        return json.dumps({"entries": ["fake/"], "total": 1})

    monkeypatch.setattr(
        main_agent, "execute_filesystem_tool", fake_execute_filesystem_tool
    )

    agent = MainAgent(slug="test", run_id="r1", goal_text="do the thing")
    response = _fake_fc("list_dir", path="/workspace")
    monkeypatch.setattr(main_agent, "call_llm", lambda **kwargs: (response, 0))

    agent._step([])

    assert captured == {"name": "list_dir", "args": {"path": "/workspace"}}
    records = [json.loads(line) for line in agent.chat_log.read_text().splitlines()]
    tool_records = [r for r in records if r["role"] == "tool"]
    assert len(tool_records) == 1
    assert tool_records[0]["name"] == "list_dir"
    assert json.loads(tool_records[0]["result"])["entries"] == ["fake/"]


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


# ---------------------------------------------------------------------------
# MAIN.md scaffold (mirrors the RESEARCH.md pattern from #275)
# ---------------------------------------------------------------------------


def test_init_creates_main_md_scaffold(patched_main_agent):
    """Constructing MainAgent scaffolds MAIN.md with `# {goal_text}\\n`."""
    goal = "win the competition by Friday"
    agent = MainAgent(slug="test", run_id="r1", goal_text=goal)

    assert agent.main_md_path == agent.base_dir / "MAIN.md"
    assert agent.main_md_path.exists()
    assert agent.main_md_path.read_text(encoding="utf-8") == f"# {goal}\n"


def test_init_does_not_clobber_existing_main_md(patched_main_agent):
    """Re-instantiating with the same (slug, run_id) leaves a populated MAIN.md untouched."""
    base_dir = main_agent._TASK_ROOT / "test" / "r1"
    base_dir.mkdir(parents=True, exist_ok=True)
    populated = "# done\n\n## What I tried\n\n- Idea 1: failed.\n"
    (base_dir / "MAIN.md").write_text(populated, encoding="utf-8")

    agent = MainAgent(slug="test", run_id="r1", goal_text="ignored on second construct")

    assert agent.main_md_path.read_text(encoding="utf-8") == populated
