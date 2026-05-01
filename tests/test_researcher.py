"""Tests for the Deep Research sub-agent tool helpers."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agents import researcher as research_module


class _StubExa:
    """Minimal Exa client stub: records kwargs and returns canned results."""

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
        SimpleNamespace(url="https://a.example/1", title="A", text=long_text, published_date="2026-01-01"),
    ]
    result = json.loads(research_module._tool_web_research("q", num_results=3))
    assert len(result["results"]) == 1
    assert result["results"][0]["text"] == long_text          # no truncation
    assert stubbed.exa.calls[0]["num_results"] == 3           # passed through, no clamp
    assert stubbed.exa.calls[0]["type"] == "auto"
    assert stubbed.exa.calls[0]["text"] is True


def test_web_research_empty_returns_error(stubbed):
    stubbed.exa._next_results = []
    result = json.loads(research_module._tool_web_research("nothing", num_results=None))
    assert "error" in result
    assert "num_results" not in stubbed.exa.calls[0]          # omitted when None


def test_web_fetch_success_and_no_truncation(stubbed):
    long_md = "# heading\n" + ("para\n" * 10_000)
    stubbed.fc._next_doc = SimpleNamespace(
        markdown=long_md, metadata=SimpleNamespace(title="Title")
    )
    result = json.loads(research_module._tool_web_fetch("https://e.example"))
    assert result["markdown"] == long_md                      # no truncation
    assert stubbed.fc.calls[0]["only_main_content"] is True


def test_execute_tool_call_dispatches_and_writes_markdown_records(stubbed):
    stubbed.exa._next_results = [
        SimpleNamespace(url="https://a.example", title="A", text="body-a", published_date=None),
    ]
    stubbed.fc._next_doc = SimpleNamespace(
        markdown="# hello\nworld", metadata=SimpleNamespace(title="H")
    )

    state = {
        "research_dir": stubbed.research_dir,
        "tool_seq": {},
    }

    # web_research
    item = SimpleNamespace(name="web_research", args={"query": "foo", "num_results": 2})
    research_module._execute_tool_call(item, state)
    wr_record = (stubbed.research_dir / "web_research" / "1.md").read_text()
    assert "# web_research #1" in wr_record
    assert "https://a.example" in wr_record
    assert "body-a" in wr_record

    # web_fetch
    item = SimpleNamespace(name="web_fetch", args={"url": "https://e.example"})
    research_module._execute_tool_call(item, state)
    wf_record = (stubbed.research_dir / "web_fetch" / "1.md").read_text()
    assert "# web_fetch #1" in wf_record
    assert "# hello\nworld" in wf_record

    # unknown tool raises
    with pytest.raises(ValueError, match="Unknown tool"):
        research_module._execute_tool_call(
            SimpleNamespace(name="nope", args={}), state
        )


def test_filesystem_tool_calls_route_to_filesystem_helpers(stubbed, monkeypatch):
    """A `bash` tool call routes through tools.filesystem.execute_filesystem_tool."""
    captured = {}

    def fake_execute_filesystem_tool(name, args):
        captured["name"] = name
        captured["args"] = args
        return json.dumps({"output": "ok", "returncode": 0, "truncated": False})

    monkeypatch.setattr(
        research_module, "execute_filesystem_tool", fake_execute_filesystem_tool
    )

    state = {
        "research_dir": stubbed.research_dir,
        "tool_seq": {},
    }

    item = SimpleNamespace(name="bash", args={"command": "ls -la"})
    out = research_module._execute_tool_call(item, state)

    assert captured == {"name": "bash", "args": {"command": "ls -la"}}
    assert json.loads(out)["returncode"] == 0


def test_build_system_inlines_custom_instructions():
    """The researcher system prompt inlines RESEARCHER_INSTRUCTIONS.md when present."""
    from prompts.research import build_system

    body = "Cite at least three peer-reviewed sources per claim."
    out = build_system(custom_instructions=body)
    assert "<custom_instructions>" in out
    assert body in out


def test_build_system_omits_section_when_no_instructions():
    """Absent / empty custom_instructions → no <custom_instructions> wrapper."""
    from prompts.research import build_system

    assert "<custom_instructions>" not in build_system()
    assert "<custom_instructions>" not in build_system(custom_instructions="")
    assert "<custom_instructions>" not in build_system(custom_instructions="   \n")


def test_render_tool_record_markdown_error_path():
    rendered = research_module._render_tool_record_markdown(
        "web_research", 2, {"query": "q"}, json.dumps({"error": "exa down"})
    )
    assert "# web_research #2" in rendered
    assert "**ERROR:** exa down" in rendered
