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
