"""Unit tests for the shared filesystem tools (tools/filesystem.py)."""

from __future__ import annotations

import json

import pytest

from schemas.bash_safety import BashSafetyVerdict
from tools import filesystem as fs


# ---------------------------------------------------------------------------
# Sandbox: every test runs against a tmp_path that is the only allowed root.
# ---------------------------------------------------------------------------


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("VERSION = '1.0'\n")
    (tmp_path / "pkg" / "core.py").write_text(
        "def add(a, b):\n"
        "    \"\"\"Add two numbers.\"\"\"\n"
        "    return a + b\n"
    )
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "config.json").write_text('{"k": 1}\n')

    monkeypatch.setattr(fs, "_ALLOWED_ROOTS", [tmp_path.resolve()])
    return tmp_path


# ---------------------------------------------------------------------------
# Read-only filesystem tools
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


def test_path_validation_blocks_outside_roots(sandbox, tmp_path_factory):
    """Every read tool must reject paths outside the configured allowlist."""
    outside = tmp_path_factory.mktemp("outside") / "evil.py"
    outside.write_text("x = 1\n")

    for blob in [
        fs._tool_read_file(str(outside)),
        fs._tool_glob_files(str(outside.parent), "*.py"),
        fs._tool_grep_code(str(outside.parent), "x"),
        fs._tool_list_dir(str(outside.parent)),
    ]:
        result = json.loads(blob)
        assert "error" in result and "outside the allowed roots" in result["error"]


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
# bash — judge-gated execution
# ---------------------------------------------------------------------------


def _patch_judge(monkeypatch, verdict: str, reason: str = "ok"):
    """Replace judge_bash_command with a function returning a fixed verdict."""
    monkeypatch.setattr(
        fs,
        "judge_bash_command",
        lambda command: BashSafetyVerdict(verdict=verdict, reason=reason),
    )


def test_bash_runs_when_judge_allows(sandbox, monkeypatch):
    """An allow verdict → command actually runs and stdout is returned."""
    _patch_judge(monkeypatch, "allow")
    out = json.loads(fs._tool_bash(f"ls {sandbox / 'pkg'}"))
    assert out["returncode"] == 0
    assert "core.py" in out["output"]


def test_bash_blocks_when_judge_blocks(monkeypatch):
    """A block verdict → no execution, error message includes the reason."""
    _patch_judge(monkeypatch, "block", reason="rm -rf / would delete the host")
    out = json.loads(fs._tool_bash("rm -rf /"))
    assert "error" in out
    assert "rm -rf /" in out["error"]


def test_bash_supports_pipes_and_redirection(sandbox, monkeypatch):
    """bash -c lets the judge'd command use shell composition."""
    _patch_judge(monkeypatch, "allow")
    out_path = sandbox / "out.txt"
    out = json.loads(
        fs._tool_bash(
            f"ls {sandbox / 'pkg'} | head -n 1 > {out_path} && cat {out_path}"
        )
    )
    assert out["returncode"] == 0
    assert out["output"].strip() in {"__init__.py", "core.py"}


def test_bash_can_mutate_project_files(sandbox, monkeypatch):
    """End-to-end: an allow verdict really lets the agent write inside the sandbox."""
    _patch_judge(monkeypatch, "allow")
    new_dir = sandbox / "build"
    out = json.loads(fs._tool_bash(f"mkdir -p {new_dir} && touch {new_dir}/marker"))
    assert out["returncode"] == 0
    assert (new_dir / "marker").exists()


# ---------------------------------------------------------------------------
# write_file + edit_file (modeled on claude-code FileWriteTool / FileEditTool)
# ---------------------------------------------------------------------------


def test_write_file_create_then_overwrite(sandbox):
    """First write returns type=create; second write to the same path returns type=update."""
    target = sandbox / "out" / "hello.py"

    first = json.loads(fs._tool_write_file(str(target), "print('v1')\n"))
    assert first["type"] == "create"
    assert target.read_text() == "print('v1')\n"

    second = json.loads(fs._tool_write_file(str(target), "print('v2')\n"))
    assert second["type"] == "update"
    assert target.read_text() == "print('v2')\n"


def test_edit_file_basic_replacement(sandbox):
    """A unique old_string is replaced once."""
    target = sandbox / "edit_basic.py"
    target.write_text("x = 1\ny = 2\n")

    out = json.loads(fs._tool_edit_file(str(target), "y = 2", "y = 99"))
    assert out["type"] == "update"
    assert out["matches_replaced"] == 1
    assert target.read_text() == "x = 1\ny = 99\n"


def test_edit_file_unique_match_required_or_replace_all(sandbox):
    """Two matches without replace_all → error; with replace_all=True both replaced."""
    target = sandbox / "edit_multi.py"
    target.write_text("a = 1\na = 1\n")

    err = json.loads(fs._tool_edit_file(str(target), "a = 1", "a = 2"))
    assert "error" in err
    assert "2 matches" in err["error"]
    # File untouched.
    assert target.read_text() == "a = 1\na = 1\n"

    out = json.loads(
        fs._tool_edit_file(str(target), "a = 1", "a = 2", replace_all=True)
    )
    assert out["matches_replaced"] == 2
    assert target.read_text() == "a = 2\na = 2\n"


def test_edit_file_no_match_returns_error(sandbox):
    """old_string that doesn't appear → error, file untouched."""
    target = sandbox / "edit_nope.py"
    target.write_text("x = 1\n")

    out = json.loads(fs._tool_edit_file(str(target), "absent", "anything"))
    assert "error" in out
    assert "not found" in out["error"]
    assert target.read_text() == "x = 1\n"


def test_edit_file_curly_quote_normalization(sandbox):
    """File has curly quotes; model passes straight quotes; replacement still succeeds."""
    target = sandbox / "edit_curly.py"
    target.write_text("msg = “hello”\n")  # curly double quotes

    out = json.loads(fs._tool_edit_file(str(target), 'msg = "hello"', 'msg = "world"'))
    assert out["type"] == "update"
    # The replacement uses the actual (curly) substring sliced from the file,
    # so the new content reflects the edit cleanly.
    assert "world" in target.read_text()
    assert "hello" not in target.read_text()


def test_write_and_edit_reject_path_outside_allowed_roots(sandbox, tmp_path_factory):
    """Both new tools refuse paths outside the configured allowlist."""
    outside = tmp_path_factory.mktemp("outside") / "evil.py"

    write_err = json.loads(fs._tool_write_file(str(outside), "data\n"))
    assert "error" in write_err
    assert "outside the allowed roots" in write_err["error"]
    assert not outside.exists()

    outside.write_text("# pre-existing\n")
    edit_err = json.loads(fs._tool_edit_file(str(outside), "pre-existing", "patched"))
    assert "error" in edit_err
    assert "outside the allowed roots" in edit_err["error"]
    assert outside.read_text() == "# pre-existing\n"


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def test_execute_filesystem_tool_routes_by_name(sandbox, monkeypatch):
    """Each known name reaches the right helper; an unknown name returns None."""
    _patch_judge(monkeypatch, "allow")

    out = fs.execute_filesystem_tool(
        "read_file", {"path": str(sandbox / "pkg" / "core.py")}
    )
    assert "def add" in json.loads(out)["content"]

    out = fs.execute_filesystem_tool(
        "glob_files", {"root": str(sandbox / "pkg"), "pattern": "*.py"}
    )
    assert json.loads(out)["total"] == 2

    out = fs.execute_filesystem_tool(
        "grep_code", {"root": str(sandbox / "pkg"), "pattern": "def add"}
    )
    assert json.loads(out)["total_matches"] == 1

    out = fs.execute_filesystem_tool("list_dir", {"path": str(sandbox)})
    assert "pkg/" in json.loads(out)["entries"]

    out = fs.execute_filesystem_tool("bash", {"command": f"ls {sandbox / 'pkg'}"})
    assert json.loads(out)["returncode"] == 0

    assert fs.execute_filesystem_tool("nope", {}) is None
