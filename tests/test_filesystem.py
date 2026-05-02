"""Unit tests for the shared filesystem tools (tools/filesystem.py)."""

from __future__ import annotations

import json

import pytest

from schemas.bash_safety import BashSafetyVerdict
from tools import filesystem as fs


# ---------------------------------------------------------------------------
# Sandbox: every test runs against a tmp_path that is the writable_root.
# ---------------------------------------------------------------------------


@pytest.fixture
def sandbox(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("VERSION = '1.0'\n")
    (tmp_path / "pkg" / "core.py").write_text(
        "def add(a, b):\n"
        "    \"\"\"Add two numbers.\"\"\"\n"
        "    return a + b\n"
    )
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "config.json").write_text('{"k": 1}\n')
    return tmp_path


# ---------------------------------------------------------------------------
# Read-only filesystem tools — run wide, no allow-list
# ---------------------------------------------------------------------------


def test_read_file(sandbox):
    """read_file returns numbered content; line range slices; missing → error."""
    full = json.loads(fs._tool_read_file(str(sandbox / "pkg" / "core.py")))
    assert "1: def add(a, b):" in full["content"]
    assert full["total_lines"] == 3

    sliced = json.loads(
        fs._tool_read_file(
            str(sandbox / "pkg" / "core.py"), start_line=1, end_line=1
        )
    )
    assert "1: def add(a, b):" in sliced["content"]
    assert "return" not in sliced["content"]

    missing = json.loads(fs._tool_read_file(str(sandbox / "nope.py")))
    assert "error" in missing


def test_reads_run_wide_no_allowlist(sandbox, tmp_path_factory):
    """Reads outside the sandbox now succeed — there is no global allow-list."""
    outside = tmp_path_factory.mktemp("outside") / "data.py"
    outside.write_text("x = 1\n")

    out = json.loads(fs._tool_read_file(str(outside)))
    assert "x = 1" in out["content"]

    glob_out = json.loads(fs._tool_glob_files(str(outside.parent), "*.py"))
    assert "data.py" in glob_out["matches"]

    grep_out = json.loads(fs._tool_grep_code(str(outside.parent), "x"))
    assert grep_out["total_matches"] == 1

    list_out = json.loads(fs._tool_list_dir(str(outside.parent)))
    assert "data.py" in list_out["entries"]


def test_glob_grep_list(sandbox):
    """Happy paths for glob_files, grep_code (with non-default file_glob), list_dir."""
    glob_out = json.loads(fs._tool_glob_files(str(sandbox / "pkg"), "*.py"))
    assert set(glob_out["matches"]) == {"__init__.py", "core.py"}

    grep_out = json.loads(fs._tool_grep_code(str(sandbox / "pkg"), r"def add"))
    assert grep_out["total_matches"] == 1

    grep_json = json.loads(
        fs._tool_grep_code(str(sandbox), r"\"k\"", file_glob="*.json")
    )
    assert grep_json["total_matches"] == 1

    list_out = json.loads(fs._tool_list_dir(str(sandbox)))
    assert "pkg/" in list_out["entries"]
    assert "data/" in list_out["entries"]


# ---------------------------------------------------------------------------
# bash — judge-gated, runs with cwd=writable_root
# ---------------------------------------------------------------------------


def _patch_judge(monkeypatch, verdict: str, reason: str = "ok"):
    """Replace judge_bash_command with a function returning a fixed verdict."""
    monkeypatch.setattr(
        fs,
        "judge_bash_command",
        lambda command, writable_root: BashSafetyVerdict(
            verdict=verdict, reason=reason
        ),
    )


def test_bash_runs_when_judge_allows(sandbox, monkeypatch):
    """An allow verdict → command actually runs and stdout is returned."""
    _patch_judge(monkeypatch, "allow")
    out = json.loads(fs._tool_bash("ls pkg", writable_root=sandbox))
    assert out["returncode"] == 0
    assert "core.py" in out["output"]


def test_bash_runs_with_cwd_set_to_writable_root(sandbox, monkeypatch):
    """`pwd` inside bash returns the writable_root, since cwd is pinned to it."""
    _patch_judge(monkeypatch, "allow")
    out = json.loads(fs._tool_bash("pwd", writable_root=sandbox))
    assert out["returncode"] == 0
    assert str(sandbox.resolve()) in out["output"]


def test_bash_judge_receives_writable_root(sandbox, monkeypatch):
    """The bash judge stub is called with (command, writable_root)."""
    seen: list[tuple[str, str]] = []

    def fake_judge(command, writable_root):
        seen.append((command, writable_root))
        return BashSafetyVerdict(verdict="allow", reason="ok")

    monkeypatch.setattr(fs, "judge_bash_command", fake_judge)
    fs._tool_bash("echo hi", writable_root=sandbox)

    assert seen == [("echo hi", str(sandbox))]


def test_bash_blocks_when_judge_blocks(sandbox, monkeypatch):
    """A block verdict → no execution, error message includes the reason."""
    _patch_judge(monkeypatch, "block", reason="rm -rf / would delete the host")
    out = json.loads(fs._tool_bash("rm -rf /", writable_root=sandbox))
    assert "error" in out
    assert "rm -rf /" in out["error"]


def test_bash_supports_pipes_and_redirection(sandbox, monkeypatch):
    """bash -c lets the judge'd command use shell composition."""
    _patch_judge(monkeypatch, "allow")
    out = json.loads(
        fs._tool_bash(
            "ls pkg | head -n 1 > out.txt && cat out.txt",
            writable_root=sandbox,
        )
    )
    assert out["returncode"] == 0
    assert out["output"].strip() in {"__init__.py", "core.py"}


def test_bash_can_mutate_project_files(sandbox, monkeypatch):
    """End-to-end: an allow verdict really lets the agent write inside the sandbox."""
    _patch_judge(monkeypatch, "allow")
    out = json.loads(
        fs._tool_bash("mkdir -p build && touch build/marker", writable_root=sandbox)
    )
    assert out["returncode"] == 0
    assert (sandbox / "build" / "marker").exists()


# ---------------------------------------------------------------------------
# write_file + edit_file — pinned to writable_root
# ---------------------------------------------------------------------------


def test_write_file_create_then_overwrite(sandbox):
    """First write returns type=create; second write to the same path returns type=update."""
    target = sandbox / "out" / "hello.py"

    first = json.loads(
        fs._tool_write_file(str(target), "print('v1')\n", writable_root=sandbox)
    )
    assert first["type"] == "create"
    assert target.read_text() == "print('v1')\n"

    second = json.loads(
        fs._tool_write_file(str(target), "print('v2')\n", writable_root=sandbox)
    )
    assert second["type"] == "update"
    assert target.read_text() == "print('v2')\n"


def test_edit_file_basic_replacement(sandbox):
    """A unique old_string is replaced once."""
    target = sandbox / "edit_basic.py"
    target.write_text("x = 1\ny = 2\n")

    out = json.loads(
        fs._tool_edit_file(str(target), "y = 2", "y = 99", writable_root=sandbox)
    )
    assert out["type"] == "update"
    assert out["matches_replaced"] == 1
    assert target.read_text() == "x = 1\ny = 99\n"


def test_edit_file_unique_match_required_or_replace_all(sandbox):
    """Two matches without replace_all → error; with replace_all=True both replaced."""
    target = sandbox / "edit_multi.py"
    target.write_text("a = 1\na = 1\n")

    err = json.loads(
        fs._tool_edit_file(str(target), "a = 1", "a = 2", writable_root=sandbox)
    )
    assert "error" in err
    assert "2 matches" in err["error"]
    assert target.read_text() == "a = 1\na = 1\n"

    out = json.loads(
        fs._tool_edit_file(
            str(target), "a = 1", "a = 2", replace_all=True, writable_root=sandbox
        )
    )
    assert out["matches_replaced"] == 2
    assert target.read_text() == "a = 2\na = 2\n"


def test_edit_file_no_match_returns_error(sandbox):
    """old_string that doesn't appear → error, file untouched."""
    target = sandbox / "edit_nope.py"
    target.write_text("x = 1\n")

    out = json.loads(
        fs._tool_edit_file(str(target), "absent", "anything", writable_root=sandbox)
    )
    assert "error" in out
    assert "not found" in out["error"]
    assert target.read_text() == "x = 1\n"


def test_edit_file_curly_quote_normalization(sandbox):
    """File has curly quotes; model passes straight quotes; replacement still succeeds."""
    target = sandbox / "edit_curly.py"
    target.write_text("msg = “hello”\n")

    out = json.loads(
        fs._tool_edit_file(
            str(target), 'msg = "hello"', 'msg = "world"', writable_root=sandbox
        )
    )
    assert out["type"] == "update"
    assert "world" in target.read_text()
    assert "hello" not in target.read_text()


def test_write_file_rejects_path_outside_writable_root(sandbox, tmp_path_factory):
    """write_file refuses to write a file outside writable_root."""
    outside = tmp_path_factory.mktemp("outside") / "evil.py"

    err = json.loads(
        fs._tool_write_file(str(outside), "data\n", writable_root=sandbox)
    )
    assert "error" in err
    assert "outside writable_root" in err["error"]
    assert not outside.exists()


def test_edit_file_rejects_path_outside_writable_root(sandbox, tmp_path_factory):
    """edit_file refuses to modify a file outside writable_root."""
    outside = tmp_path_factory.mktemp("outside") / "evil.py"
    outside.write_text("# pre-existing\n")

    err = json.loads(
        fs._tool_edit_file(
            str(outside), "pre-existing", "patched", writable_root=sandbox
        )
    )
    assert "error" in err
    assert "outside writable_root" in err["error"]
    assert outside.read_text() == "# pre-existing\n"


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def test_execute_filesystem_tool_routes_by_name(sandbox, monkeypatch):
    """Each known name reaches the right helper; an unknown name returns None."""
    _patch_judge(monkeypatch, "allow")

    out = fs.execute_filesystem_tool(
        "read_file", {"path": str(sandbox / "pkg" / "core.py")},
        writable_root=sandbox,
    )
    assert "def add" in json.loads(out)["content"]

    out = fs.execute_filesystem_tool(
        "glob_files", {"root": str(sandbox / "pkg"), "pattern": "*.py"},
        writable_root=sandbox,
    )
    assert json.loads(out)["total"] == 2

    out = fs.execute_filesystem_tool(
        "grep_code", {"root": str(sandbox / "pkg"), "pattern": "def add"},
        writable_root=sandbox,
    )
    assert json.loads(out)["total_matches"] == 1

    out = fs.execute_filesystem_tool(
        "list_dir", {"path": str(sandbox)}, writable_root=sandbox
    )
    assert "pkg/" in json.loads(out)["entries"]

    out = fs.execute_filesystem_tool(
        "bash", {"command": "ls pkg"}, writable_root=sandbox
    )
    assert json.loads(out)["returncode"] == 0

    assert fs.execute_filesystem_tool("nope", {}, writable_root=sandbox) is None


# ---------------------------------------------------------------------------
# LLM-side palette: get_filesystem_tools() must stay in sync with the dispatcher
# ---------------------------------------------------------------------------


def test_get_filesystem_tools_palette_matches_dispatcher():
    """The FunctionDeclaration palette and the dispatcher's tool-name set must
    list the same tools. If they drift, agents will either advertise tools the
    dispatcher cannot run, or the dispatcher will silently support tools the
    LLM is never told about."""
    from utils.llm_utils import get_filesystem_tools

    palette = {decl.name for decl in get_filesystem_tools()}
    assert palette == fs.FILESYSTEM_TOOL_NAMES
