"""Unit tests for the codebase exploration sub-agent's tool helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents import explorer as explore_module


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Create a temp directory tree and point _ALLOWED_ROOTS at it."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("VERSION = '1.0'\n")
    (tmp_path / "pkg" / "core.py").write_text(
        "def add(a, b):\n"
        "    \"\"\"Add two numbers.\"\"\"\n"
        "    return a + b\n"
    )
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "config.json").write_text('{"k": 1}\n')

    monkeypatch.setattr(explore_module, "_ALLOWED_ROOTS", [tmp_path.resolve()])
    return tmp_path


def test_read_file(sandbox):
    """read_file returns numbered content; line range slices; missing → error."""
    full = json.loads(explore_module._tool_read_file(str(sandbox / "pkg" / "core.py")))
    assert "1: def add(a, b):" in full["content"]
    assert full["total_lines"] == 3

    sliced = json.loads(
        explore_module._tool_read_file(
            str(sandbox / "pkg" / "core.py"), start_line=1, end_line=1
        )
    )
    assert "1: def add(a, b):" in sliced["content"]
    assert "return" not in sliced["content"]

    missing = json.loads(explore_module._tool_read_file(str(sandbox / "nope.py")))
    assert "error" in missing


def test_path_validation_blocks_outside_roots(sandbox, tmp_path_factory):
    """Every read tool must reject paths outside the configured allowlist."""
    outside = tmp_path_factory.mktemp("outside") / "evil.py"
    outside.write_text("x = 1\n")

    for blob in [
        explore_module._tool_read_file(str(outside)),
        explore_module._tool_glob_files(str(outside.parent), "*.py"),
        explore_module._tool_grep_code(str(outside.parent), "x"),
        explore_module._tool_list_dir(str(outside.parent)),
    ]:
        result = json.loads(blob)
        assert "error" in result and "outside the allowed roots" in result["error"]


def test_glob_grep_list(sandbox):
    """Happy paths for glob_files, grep_code (with non-default file_glob), list_dir."""
    glob_out = json.loads(
        explore_module._tool_glob_files(str(sandbox / "pkg"), "*.py")
    )
    assert set(glob_out["matches"]) == {"__init__.py", "core.py"}

    grep_out = json.loads(
        explore_module._tool_grep_code(str(sandbox / "pkg"), r"def add")
    )
    assert grep_out["total_matches"] == 1

    grep_json = json.loads(
        explore_module._tool_grep_code(
            str(sandbox), r"\"k\"", file_glob="*.json"
        )
    )
    assert grep_json["total_matches"] == 1

    list_out = json.loads(explore_module._tool_list_dir(str(sandbox)))
    assert "pkg/" in list_out["entries"]
    assert "data/" in list_out["entries"]


@pytest.mark.parametrize(
    "command",
    [
        "ls -la /tmp",
        "cat /etc/hosts",
        "head -n 5 file.txt",
        "git status",
        "git log --oneline -5",
        "git diff HEAD",
    ],
)
def test_validate_bash_command_allows(command):
    assert explore_module._validate_bash_command(command) is None


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf /",          # unallowed command
        "python script.py",  # unallowed command
        "git add .",         # unallowed git subcommand
        "git commit -m foo",
        "ls | head",         # pipe
        "ls > out.txt",      # redirect
        "ls; rm -rf /",      # chain
        "ls && rm a",
        "echo `whoami`",     # backtick
        "echo $(whoami)",    # command substitution
        "",                  # empty
    ],
)
def test_validate_bash_command_rejects(command):
    assert explore_module._validate_bash_command(command) is not None


def test_bash_readonly_executes(sandbox):
    """End-to-end: bash_readonly runs an allowlisted command and returns output."""
    out = json.loads(explore_module._tool_bash_readonly(f"ls {sandbox / 'pkg'}"))
    assert out["returncode"] == 0
    assert "core.py" in out["output"]
