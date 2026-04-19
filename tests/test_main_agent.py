"""Unit tests for the flat MainAgent dispatcher."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agents import main_agent
from agents.main_agent import MainAgent


def _fake_fc(name: str, **args):
    """A Gemini-like response whose single part is a function_call."""
    fc = SimpleNamespace(name=name, args=args)
    part = SimpleNamespace(function_call=fc)
    content = SimpleNamespace(parts=[part])
    content.model_dump = lambda **_: {
        "role": "model",
        "parts": [{"function_call": {"name": name, "args": dict(args)}}],
    }
    return SimpleNamespace(candidates=[SimpleNamespace(content=content)])


def _fake_text(text: str):
    """A text-only response (no function_call attribute)."""
    part = SimpleNamespace(text=text)
    content = SimpleNamespace(parts=[part])
    content.model_dump = lambda **_: {"role": "model", "parts": [{"text": text}]}
    return SimpleNamespace(candidates=[SimpleNamespace(content=content)])


@pytest.fixture
def patched_main_agent(monkeypatch, tmp_path):
    monkeypatch.setattr(main_agent, "_TASK_ROOT", tmp_path / "task")

    # Stub execute_code — return a job that echoes "ran {name}" without spawning a subprocess.
    class _FakeJob:
        def __init__(self, filepath):
            self._name = filepath

        def result(self):
            return f"ran {self._name.rsplit('/', 1)[-1]}"

    monkeypatch.setattr(
        main_agent, "execute_code", lambda filepath, timeout_seconds=300: _FakeJob(filepath)
    )

    # Stub the two web helpers so no network hits happen.
    research_calls: list = []
    fetch_calls: list = []
    monkeypatch.setattr(
        main_agent,
        "tool_web_research",
        lambda query, num_results=None: research_calls.append((query, num_results))
        or json.dumps({"results": [{"url": "https://example.com", "title": "t", "text": "body", "published_date": None}]}),
    )
    monkeypatch.setattr(
        main_agent,
        "tool_web_fetch",
        lambda url: fetch_calls.append(url)
        or json.dumps({"url": url, "title": "t", "markdown": "# body"}),
    )

    return {"research_calls": research_calls, "fetch_calls": fetch_calls}


def test_dispatches_every_tool(patched_main_agent, monkeypatch):
    agent = MainAgent(slug="test", run_id="r1", goal_text="do the thing")

    responses = iter(
        [
            _fake_fc("add_idea", title="first", description="body"),
            _fake_fc("update_idea", idea_id=1, description="body v2"),
            _fake_fc("execute_python", filename="probe.py", code="print('hi')"),
            _fake_fc("read_file", path=str(agent.ideas_dir / "INDEX.md")),
            _fake_fc("list_dir", path=str(agent.base_dir)),
            _fake_fc("glob_files", root=str(agent.ideas_dir), pattern="*.md"),
            _fake_fc(
                "grep_code",
                root=str(agent.ideas_dir),
                pattern="first",
                file_glob="*.md",
            ),
            _fake_fc("bash_readonly", command="ls " + str(agent.ideas_dir)),
            _fake_fc("web_research", query="foo"),
            _fake_fc("web_fetch", url="https://example.com"),
            _fake_fc("remove_idea", idea_id=1),
        ]
    )
    monkeypatch.setattr(main_agent, "call_llm", lambda **kwargs: (next(responses), 0))

    for _ in range(11):
        agent._step([])

    # execute_python wrote the LLM-chosen filename
    assert (agent.snippets_dir / "probe.py").exists()
    assert "print('hi')" in (agent.snippets_dir / "probe.py").read_text()

    # Idea pool lifecycle: add → update → remove → empty
    index_md = (agent.ideas_dir / "INDEX.md").read_text()
    assert "first" not in index_md

    # Web helpers each fired exactly once; audit records persisted.
    assert patched_main_agent["research_calls"] == [("foo", None)]
    assert patched_main_agent["fetch_calls"] == ["https://example.com"]
    assert (agent.web_research_dir / "1.md").exists()
    assert (agent.web_fetch_dir / "1.md").exists()

    # Chat log has 11 assistant turns + 11 tool results = 22 records.
    records = [json.loads(line) for line in agent.chat_log.read_text().splitlines()]
    assert len(records) == 22
    assert [r["name"] for r in records[1::2]] == [
        "add_idea",
        "update_idea",
        "execute_python",
        "read_file",
        "list_dir",
        "glob_files",
        "grep_code",
        "bash_readonly",
        "web_research",
        "web_fetch",
        "remove_idea",
    ]


def test_execute_python_rejects_bad_filename(patched_main_agent, monkeypatch):
    agent = MainAgent(slug="test", run_id="r1", goal_text="demo")

    for bad in ("../evil.py", "foo/bar.py", "no_extension", ".hidden.py"):
        result = agent._dispatch("execute_python", {"filename": bad, "code": "pass"})
        payload = json.loads(result)
        assert "error" in payload, f"expected rejection for {bad!r}, got {payload}"


def test_text_only_response_is_logged_and_ignored(patched_main_agent, monkeypatch, caplog):
    agent = MainAgent(slug="test", run_id="r1", goal_text="demo")
    monkeypatch.setattr(
        main_agent, "call_llm", lambda **kwargs: (_fake_text("hello"), 0)
    )

    with caplog.at_level("WARNING"):
        agent._step([])

    assert any("text-only" in rec.message for rec in caplog.records)
    records = [json.loads(line) for line in agent.chat_log.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["role"] == "assistant"
