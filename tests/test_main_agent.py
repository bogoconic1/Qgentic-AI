"""Unit tests for the MainAgent dispatch loop."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents import main_agent
from agents.main_agent import MainAgent


_resp_counter = 0


def _next_resp_id():
    global _resp_counter
    _resp_counter += 1
    return f"resp_{_resp_counter}"


def _fake_fc(name: str, **args):
    fc = SimpleNamespace(
        type="function_call",
        name=name,
        arguments=json.dumps(args),
        call_id=f"call_{name}",
    )
    resp = SimpleNamespace(id=_next_resp_id(), output=[fc], output_text=None)
    resp.usage = SimpleNamespace(input_tokens=0)
    return resp


def _fake_text(text: str):
    msg = SimpleNamespace(
        type="message", content=[SimpleNamespace(type="output_text", text=text)]
    )
    resp = SimpleNamespace(id=_next_resp_id(), output=[msg], output_text=text)
    resp.usage = SimpleNamespace(input_tokens=0)
    return resp


def _fake_multi(*calls: tuple[str, dict]):
    items = []
    for name, args in calls:
        items.append(
            SimpleNamespace(
                type="function_call",
                name=name,
                arguments=json.dumps(args),
                call_id=f"call_{name}",
            )
        )
    resp = SimpleNamespace(id=_next_resp_id(), output=items, output_text=None)
    resp.usage = SimpleNamespace(input_tokens=0)
    return resp


@pytest.fixture
def patched_main_agent(monkeypatch, tmp_path):
    monkeypatch.setattr(main_agent, "_TASK_ROOT", tmp_path / "task")
    monkeypatch.setattr(main_agent, "load_agent_skills", lambda: [])

    run_solution_calls: list = []

    def fake_run_solution(version_dir):
        run_solution_calls.append({"version_dir": str(version_dir)})
        return json.dumps(
            {
                "success": True,
                "score": 0.5,
                "stats": {},
                "elapsed_seconds": 0.0,
                "output_tail": "",
            }
        )

    monkeypatch.setattr(main_agent, "tool_run_solution", fake_run_solution)

    web_search_calls: list = []

    def fake_web_search_stack_trace(query):
        web_search_calls.append({"query": query})
        return f"trace + fix for: {query[:40]}"

    monkeypatch.setattr(
        main_agent, "tool_web_search_stack_trace", fake_web_search_stack_trace
    )

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

    return {
        "run_solution_calls": run_solution_calls,
        "web_search_calls": web_search_calls,
        "research_calls": research_calls,
    }


def test_dispatches_each_tool(patched_main_agent, monkeypatch):
    agent = MainAgent(slug="test", run_id="r1", goal_text="do the thing")

    responses = iter(
        [
            _fake_fc("add_idea", title="try it", description="body"),
            _fake_fc("update_idea", idea_id=1, description="body v2"),
            _fake_fc("start_dev_session", idea_id=1),
            _fake_fc(
                "run_solution",
                version_dir=str(agent.base_dir / "developer_v1"),
            ),
            _fake_fc("web_search_stack_trace", query="Traceback ..."),
            _fake_fc("research", instruction="look up X"),
            _fake_fc("remove_idea", idea_id=1),
        ]
    )
    monkeypatch.setattr(main_agent, "call_llm", lambda **kwargs: (next(responses), 0))

    for _ in range(7):
        agent._step([])

    version_dir = agent.base_dir / "developer_v1"
    assert version_dir.is_dir()
    assert (version_dir / "SOLUTION.py").exists()
    assert (version_dir / "SOLUTION.md").read_text(encoding="utf-8").startswith("# ")
    assert agent.dev_iter == 1

    assert len(patched_main_agent["run_solution_calls"]) == 1
    assert patched_main_agent["run_solution_calls"][0]["version_dir"] == str(
        version_dir
    )

    assert len(patched_main_agent["web_search_calls"]) == 1
    assert patched_main_agent["web_search_calls"][0]["query"].startswith("Traceback")

    assert len(patched_main_agent["research_calls"]) == 1
    assert patched_main_agent["research_calls"][0]["research_iter"] == 1

    assert "try it" not in (agent.ideas_dir / "INDEX.md").read_text()

    records = [json.loads(line) for line in agent.chat_log.read_text().splitlines()]
    assert len(records) == 14
    assert [r["name"] for r in records[1::2]] == [
        "add_idea",
        "update_idea",
        "start_dev_session",
        "run_solution",
        "web_search_stack_trace",
        "research",
        "remove_idea",
    ]


def test_run_solution_without_version_dir_returns_error(
    patched_main_agent, monkeypatch
):
    agent = MainAgent(slug="test", run_id="r1", goal_text="the session goal body")

    result = agent._dispatch("run_solution", {})

    assert "error" in json.loads(result)
    assert patched_main_agent["run_solution_calls"] == []


def test_start_dev_session_uses_idea_title_for_solution_md(
    patched_main_agent, monkeypatch
):
    agent = MainAgent(slug="test", run_id="r1", goal_text="goal")
    agent._dispatch(
        "add_idea", {"title": "fancy refactor", "description": "do something"}
    )

    result = json.loads(agent._dispatch("start_dev_session", {"idea_id": 1}))
    version_dir = Path(result["version_dir"])

    assert version_dir.name == "developer_v1"
    solution_md = (version_dir / "SOLUTION.md").read_text(encoding="utf-8")
    assert solution_md.startswith("# fancy refactor")


def test_parallel_dispatch_preserves_order(patched_main_agent, monkeypatch):
    agent = MainAgent(slug="test", run_id="r1", goal_text="do the thing")

    monkeypatch.setattr(
        main_agent,
        "execute_filesystem_tool",
        lambda name, args, *, writable_root: json.dumps(
            {"output": f"ran {name}", "returncode": 0}
        ),
    )

    response = _fake_multi(
        ("read_file", {"path": "/tmp/seed.txt"}),
        ("start_dev_session", {}),
        ("research", {"instruction": "explore Y"}),
    )
    monkeypatch.setattr(main_agent, "call_llm", lambda **kwargs: (response, 0))

    agent._step([])

    assert (agent.base_dir / "developer_v1").is_dir()
    assert len(patched_main_agent["research_calls"]) == 1

    tool_outputs = [
        item
        for item in agent.input_list
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    ]
    assert len(tool_outputs) == 3

    records = [json.loads(line) for line in agent.chat_log.read_text().splitlines()]
    tool_records = [r for r in records if r["role"] == "tool"]
    assert [r["name"] for r in tool_records] == [
        "read_file",
        "start_dev_session",
        "research",
    ]


def test_stuck_nudge_fires_after_repeated_identical_calls(
    patched_main_agent, monkeypatch
):
    agent = MainAgent(slug="test", run_id="r1", goal_text="do the thing")

    monkeypatch.setattr(
        main_agent,
        "execute_filesystem_tool",
        lambda name, args, *, writable_root: json.dumps(
            {"output": "ok", "returncode": 0}
        ),
    )
    response = _fake_fc("read_file", path="/tmp/loop.txt")
    monkeypatch.setattr(main_agent, "call_llm", lambda **kwargs: (response, 0))

    threshold = main_agent._STUCK_REPEAT_THRESHOLD
    nudge_text = main_agent._STUCK_NUDGE

    def _has_nudge(messages: list[dict]) -> bool:
        for msg in messages:
            if msg.get("role") != "user":
                continue
            if nudge_text in msg.get("content", ""):
                return True
        return False

    for _ in range(threshold - 1):
        agent._step([])
    assert not _has_nudge(agent.input_list)

    agent._step([])
    last = agent.input_list[-1]
    assert last["role"] == "user"
    assert nudge_text in last["content"]

    records = [json.loads(line) for line in agent.chat_log.read_text().splitlines()]
    user_records = [r for r in records if r.get("role") == "user"]
    assert any(
        nudge_text in r.get("content", {}).get("content", "") for r in user_records
    ), "stuck nudge must be written to JSONL chat log, not just input_list"

    agent._step([])
    assert agent.input_list[-1].get("type") == "function_call_output"


def test_stuck_nudge_does_not_fire_for_varied_calls(patched_main_agent, monkeypatch):
    agent = MainAgent(slug="test", run_id="r1", goal_text="do the thing")

    threshold = main_agent._STUCK_REPEAT_THRESHOLD
    nudge_text = main_agent._STUCK_NUDGE

    monkeypatch.setattr(
        main_agent,
        "execute_filesystem_tool",
        lambda name, args, *, writable_root: json.dumps(
            {"output": "ok", "returncode": 0}
        ),
    )
    responses = iter(
        [_fake_fc("read_file", path=f"/tmp/{i}.txt") for i in range(threshold + 2)]
    )
    monkeypatch.setattr(main_agent, "call_llm", lambda **kwargs: (next(responses), 0))

    for _ in range(threshold + 2):
        agent._step([])

    for msg in agent.input_list:
        if msg.get("role") != "user":
            continue
        assert nudge_text not in msg.get("content", "")


def test_filesystem_tool_calls_route_to_filesystem_helpers(
    patched_main_agent, monkeypatch
):
    captured = {}

    def fake_execute_filesystem_tool(name, args, *, writable_root):
        captured["name"] = name
        captured["args"] = args
        captured["writable_root"] = writable_root
        return json.dumps({"entries": ["fake/"], "total": 1})

    monkeypatch.setattr(
        main_agent, "execute_filesystem_tool", fake_execute_filesystem_tool
    )

    agent = MainAgent(slug="test", run_id="r1", goal_text="do the thing")
    response = _fake_fc("list_dir", path="/workspace")
    monkeypatch.setattr(main_agent, "call_llm", lambda **kwargs: (response, 0))

    agent._step([])

    assert captured["writable_root"] == agent.base_dir
    records = [json.loads(line) for line in agent.chat_log.read_text().splitlines()]
    tool_records = [r for r in records if r["role"] == "tool"]
    assert len(tool_records) == 1
    assert tool_records[0]["name"] == "list_dir"
    assert json.loads(tool_records[0]["result"])["entries"] == ["fake/"]


def test_single_text_only_passes_through_silently(patched_main_agent, monkeypatch):
    agent = MainAgent(slug="test", run_id="r1", goal_text="do the thing")
    monkeypatch.setattr(
        main_agent, "call_llm", lambda **kwargs: (_fake_text("hello"), 0)
    )

    agent._step([])

    assert agent._consecutive_text_only == 1
    assert agent._done is False
    assert agent.input_list[-1]["role"] == "assistant"
    records = [json.loads(line) for line in agent.chat_log.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["role"] == "assistant"


def test_three_consecutive_text_only_sets_done_flag(patched_main_agent, monkeypatch):
    agent = MainAgent(slug="test", run_id="r1", goal_text="do the thing")
    monkeypatch.setattr(
        main_agent, "call_llm", lambda **kwargs: (_fake_text("hello"), 0)
    )

    threshold = main_agent._TEXT_ONLY_TERMINATE_THRESHOLD
    for _ in range(threshold):
        agent._step([])

    assert agent._consecutive_text_only == threshold
    assert agent._done is True


def test_function_call_resets_text_only_counter(patched_main_agent, monkeypatch):
    agent = MainAgent(slug="test", run_id="r1", goal_text="do the thing")

    monkeypatch.setattr(
        main_agent,
        "execute_filesystem_tool",
        lambda name, args, *, writable_root: json.dumps(
            {"output": "ok", "returncode": 0}
        ),
    )

    responses = iter(
        [
            _fake_text("hello"),
            _fake_text("still here"),
            _fake_fc("read_file", path="/tmp/x.txt"),
            _fake_text("hmm"),
        ]
    )
    monkeypatch.setattr(main_agent, "call_llm", lambda **kwargs: (next(responses), 0))

    for _ in range(4):
        agent._step([])

    assert agent._consecutive_text_only == 1
    assert agent._done is False


def test_init_creates_main_md_scaffold(patched_main_agent):
    goal = "win the competition by Friday"
    agent = MainAgent(slug="test", run_id="r1", goal_text=goal)

    assert agent.main_md_path == agent.base_dir / "MAIN.md"
    assert agent.main_md_path.exists()
    assert agent.main_md_path.read_text(encoding="utf-8") == f"# {goal}\n"


def test_init_does_not_clobber_existing_main_md(patched_main_agent):
    base_dir = main_agent._TASK_ROOT / "test" / "r1"
    base_dir.mkdir(parents=True, exist_ok=True)
    populated = "# done\n\n## What I tried\n\n- Idea 1: failed.\n"
    (base_dir / "MAIN.md").write_text(populated, encoding="utf-8")

    agent = MainAgent(slug="test", run_id="r1", goal_text="ignored on second construct")

    assert agent.main_md_path.read_text(encoding="utf-8") == populated
