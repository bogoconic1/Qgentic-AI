"""Tests for the Deep Research sub-agent tool helpers."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agents import researcher as research_module


class _StubExa:
    def __init__(self, *, api_key=None):
        self.calls = []
        self._next_results: list[SimpleNamespace] = []

    def search_and_contents(self, query, **kwargs):
        self.calls.append({"query": query, **kwargs})
        return SimpleNamespace(results=self._next_results)


class _StubFirecrawl:
    def __init__(self, *, api_key=None):
        self.calls = []
        self._next_doc: SimpleNamespace | None = None

    def scrape(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return self._next_doc


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    exa = _StubExa()
    fc = _StubFirecrawl()

    monkeypatch.setenv("EXA_API_KEY", "test-exa")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "test-fc")
    monkeypatch.setattr(research_module, "Exa", lambda api_key: exa)
    monkeypatch.setattr(research_module, "Firecrawl", lambda api_key: fc)

    research_dir = tmp_path / "research_1"
    (research_dir / "web_research").mkdir(parents=True)
    (research_dir / "web_fetch").mkdir(parents=True)

    return SimpleNamespace(
        exa=exa,
        fc=fc,
        research_dir=research_dir,
    )


def test_web_research_success_and_no_truncation(stubbed):
    long_text = "x" * 50_000
    stubbed.exa._next_results = [
        SimpleNamespace(
            url="https://a.example/1",
            title="A",
            text=long_text,
            published_date="2026-01-01",
        ),
    ]
    result = json.loads(research_module._tool_web_research("q", num_results=3))
    assert len(result["results"]) == 1
    assert result["results"][0]["text"] == long_text
    assert stubbed.exa.calls[0]["num_results"] == 3


def test_web_research_empty_returns_error(stubbed):
    stubbed.exa._next_results = []
    result = json.loads(research_module._tool_web_research("nothing", num_results=None))
    assert "error" in result
    assert "num_results" not in stubbed.exa.calls[0]


def test_web_fetch_success_and_no_truncation(stubbed):
    long_md = "# heading\n" + ("para\n" * 10_000)
    stubbed.fc._next_doc = SimpleNamespace(
        markdown=long_md, metadata=SimpleNamespace(title="Title")
    )
    result = json.loads(research_module._tool_web_fetch("https://e.example"))
    assert result["markdown"] == long_md


def _make_fc(name, args_dict):
    return SimpleNamespace(
        name=name,
        arguments=json.dumps(args_dict),
        call_id=f"call_{name}",
    )


def test_execute_tool_call_dispatches_and_writes_markdown_records(stubbed):
    stubbed.exa._next_results = [
        SimpleNamespace(
            url="https://a.example",
            title="A",
            text="body-a",
            published_date=None,
        ),
    ]
    stubbed.fc._next_doc = SimpleNamespace(
        markdown="# hello\nworld", metadata=SimpleNamespace(title="H")
    )

    state = {
        "research_dir": stubbed.research_dir,
        "tool_seq": {},
    }

    research_module._execute_tool_call(
        _make_fc("web_research", {"query": "foo", "num_results": 2}), state
    )
    wr_record = (stubbed.research_dir / "web_research" / "1.md").read_text()
    assert "# web_research #1" in wr_record
    assert "https://a.example" in wr_record
    assert "body-a" in wr_record

    research_module._execute_tool_call(
        _make_fc("web_fetch", {"url": "https://e.example"}), state
    )
    wf_record = (stubbed.research_dir / "web_fetch" / "1.md").read_text()
    assert "# web_fetch #1" in wf_record
    assert "# hello\nworld" in wf_record

    with pytest.raises(ValueError, match="Unknown tool"):
        research_module._execute_tool_call(_make_fc("nope", {}), state)


def test_build_system_inlines_custom_instructions():
    from prompts.research import build_system

    body = "Cite at least three peer-reviewed sources per claim."
    out = build_system(writable_root="/tmp/research_1", custom_instructions=body)
    assert body in out
    assert "/tmp/research_1" in out


def test_render_tool_record_markdown_error_path():
    rendered = research_module._render_tool_record_markdown(
        "web_research", 2, {"query": "q"}, json.dumps({"error": "exa down"})
    )
    assert "# web_research #2" in rendered
    assert "**ERROR:** exa down" in rendered


def _terminating_response():
    msg = SimpleNamespace(
        type="message",
        content=[SimpleNamespace(type="output_text", text="done")],
    )
    resp = SimpleNamespace(id="resp_term", output=[msg], output_text="done")
    resp.usage = SimpleNamespace(input_tokens=100)
    return resp


def test_run_creates_research_md_scaffold(monkeypatch, tmp_path):
    monkeypatch.setattr(research_module, "_TASK_ROOT", tmp_path)
    monkeypatch.setattr(research_module, "should_compact", lambda _: False)
    monkeypatch.setattr(
        research_module,
        "call_llm",
        lambda **_: (_terminating_response(), 100),
    )

    agent = research_module.ResearcherAgent("slug", "run-1", 1)
    instruction = "Research how to fine-tune Llama on Modal"
    result = agent.run(instruction)

    research_md = tmp_path / "slug" / "run-1" / "research_1" / "RESEARCH.md"
    assert research_md.exists()
    assert research_md.read_text(encoding="utf-8") == f"# {instruction}\n"
    assert result == f"# {instruction}\n"


def test_run_returns_research_md_contents_when_populated(monkeypatch, tmp_path):
    monkeypatch.setattr(research_module, "_TASK_ROOT", tmp_path)
    monkeypatch.setattr(research_module, "should_compact", lambda _: False)

    research_md_path = tmp_path / "slug" / "run-1" / "research_1" / "RESEARCH.md"
    populated = "# done\n\nFinding: foo is bar (https://example.com).\n"

    call_count = [0]

    def fake_call_llm(**_):
        call_count[0] += 1
        if call_count[0] == 1:
            fc = SimpleNamespace(
                type="function_call",
                name="write_file",
                arguments=json.dumps(
                    {"path": str(research_md_path), "content": populated}
                ),
                call_id="call_write",
            )
            resp = SimpleNamespace(id="resp_fc", output=[fc], output_text=None)
            resp.usage = SimpleNamespace(input_tokens=100)
            return resp, 100
        return _terminating_response(), 100

    monkeypatch.setattr(research_module, "call_llm", fake_call_llm)

    monkeypatch.setattr(
        research_module,
        "execute_filesystem_tool",
        lambda name, args, *, writable_root: json.dumps(
            {
                "type": "create",
                "path": str(research_md_path),
                "bytes_written": len(populated),
            }
        ),
    )

    def fake_execute_tool_call(fc, state):
        args = json.loads(fc.arguments)
        if fc.name == "write_file":
            research_md_path.parent.mkdir(parents=True, exist_ok=True)
            research_md_path.write_text(args["content"], encoding="utf-8")
            return json.dumps({"type": "create", "path": args["path"]})
        raise ValueError(f"Unknown: {fc.name}")

    monkeypatch.setattr(research_module, "_execute_tool_call", fake_execute_tool_call)

    agent = research_module.ResearcherAgent("slug", "run-1", 1)
    result = agent.run("any instruction")

    assert research_md_path.read_text(encoding="utf-8") == populated
    assert result == populated
    assert call_count[0] == 2
