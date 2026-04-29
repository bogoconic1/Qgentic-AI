"""Shared filesystem tools (read + grep + glob + list + bash).

Lifted out of ``agents/explorer.py`` so the same five tools can be exposed
to the developer and researcher subagents. The four read-only helpers
(``_tool_read_file``, ``_tool_glob_files``, ``_tool_grep_code``,
``_tool_list_dir``) carry the same semantics as before — paths must
resolve under one of the configured allowed roots.

The fifth tool, ``_tool_bash``, is the widened replacement for the old
``bash_readonly``: it runs through ``bash -c`` so pipes, redirection,
chaining, and any command name are allowed, but every command is run
past ``judge_bash_command`` first to block destructive operations.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sysconfig
from pathlib import Path

from project_config import get_config
from tools.bash_judge import judge_bash_command


logger = logging.getLogger(__name__)


_RUNTIME_CFG = get_config()["runtime"]

_USER_ROOTS = [Path(r).resolve() for r in _RUNTIME_CFG["explore_allowed_roots"]]
_SITE_PACKAGES = Path(sysconfig.get_path("purelib")).resolve()
_ALLOWED_ROOTS: list[Path] = _USER_ROOTS + [_SITE_PACKAGES]


_BASH_OUTPUT_CAP_BYTES = 8 * 1024
_BASH_TIMEOUT_SECONDS = int(_RUNTIME_CFG.get("bash_timeout_seconds", 600))


# ---------------------------------------------------------------------------
# Path validation
# ---------------------------------------------------------------------------


def _validate_path(path: Path) -> str | None:
    """Return an error string if the resolved path is outside every allowed root."""
    try:
        resolved = path.resolve()
    except Exception as exc:
        return f"Failed to resolve path: {exc}"
    for root in _ALLOWED_ROOTS:
        try:
            resolved.relative_to(root)
            return None
        except ValueError:
            continue
    return (
        f"Path {resolved} is outside the allowed roots "
        f"({', '.join(str(r) for r in _ALLOWED_ROOTS)})"
    )


# ---------------------------------------------------------------------------
# Read-only filesystem tools
# ---------------------------------------------------------------------------


def _tool_read_file(
    path: str, start_line: int | None = None, end_line: int | None = None
) -> str:
    full_path = Path(path)
    error = _validate_path(full_path)
    if error:
        return json.dumps({"error": error})

    resolved = full_path.resolve()
    if not resolved.exists():
        return json.dumps({"error": f"File not found: {path}"})
    if not resolved.is_file():
        return json.dumps({"error": f"Not a file: {path}"})

    try:
        lines = resolved.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        return json.dumps({"error": f"Failed to read file: {exc}"})

    total_lines = len(lines)

    if start_line is not None or end_line is not None:
        s = (start_line or 1) - 1
        e = end_line or total_lines
        lines = lines[s:e]
        line_offset = s + 1
    else:
        line_offset = 1
        if total_lines > 300:
            lines = lines[:300]
            numbered = "\n".join(
                f"{i + line_offset}: {line}" for i, line in enumerate(lines)
            )
            return json.dumps(
                {
                    "content": numbered,
                    "total_lines": total_lines,
                    "truncated": True,
                    "showing": "1-300",
                }
            )

    numbered = "\n".join(f"{i + line_offset}: {line}" for i, line in enumerate(lines))
    return json.dumps({"content": numbered, "total_lines": total_lines})


def _tool_glob_files(root: str, pattern: str) -> str:
    root_path = Path(root)
    error = _validate_path(root_path)
    if error:
        return json.dumps({"error": error})

    resolved = root_path.resolve()
    if not resolved.exists():
        return json.dumps({"error": f"Root not found: {root}"})
    if not resolved.is_dir():
        return json.dumps({"error": f"Root is not a directory: {root}"})

    try:
        matches = sorted(str(p.relative_to(resolved)) for p in resolved.glob(pattern))
    except Exception as exc:
        return json.dumps({"error": f"Glob failed: {exc}"})

    total = len(matches)
    if total > 50:
        matches = matches[:50]

    return json.dumps({"matches": matches, "total": total, "truncated": total > 50})


def _tool_grep_code(
    root: str,
    pattern: str,
    file_glob: str = "*.py",
    max_results: int = 20,
) -> str:
    root_path = Path(root)
    error = _validate_path(root_path)
    if error:
        return json.dumps({"error": error})

    resolved = root_path.resolve()
    if not resolved.exists():
        return json.dumps({"error": f"Root not found: {root}"})
    if not resolved.is_dir():
        return json.dumps({"error": f"Root is not a directory: {root}"})

    try:
        result = subprocess.run(
            ["grep", "-rn", f"--include={file_glob}", "-E", pattern, str(resolved)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        lines = result.stdout.strip().splitlines()
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "Search timed out"})
    except Exception as exc:
        return json.dumps({"error": f"Search failed: {exc}"})

    cleaned = lines[:max_results]
    return json.dumps(
        {
            "matches": cleaned,
            "total_matches": len(lines),
            "showing": min(max_results, len(lines)),
        }
    )


def _tool_list_dir(path: str, max_entries: int = 100) -> str:
    full_path = Path(path)
    error = _validate_path(full_path)
    if error:
        return json.dumps({"error": error})

    resolved = full_path.resolve()
    if not resolved.exists():
        return json.dumps({"error": f"Path not found: {path}"})
    if not resolved.is_dir():
        return json.dumps({"error": f"Not a directory: {path}"})

    try:
        children = sorted(resolved.iterdir(), key=lambda p: p.name)
    except Exception as exc:
        return json.dumps({"error": f"Failed to list directory: {exc}"})

    total = len(children)
    truncated = total > max_entries
    if truncated:
        children = children[:max_entries]

    entries = [f"{p.name}/" if p.is_dir() else p.name for p in children]
    return json.dumps(
        {
            "entries": entries,
            "total": total,
            "showing": len(entries),
            "truncated": truncated,
        }
    )


# ---------------------------------------------------------------------------
# Bash — widened, safety-judged
# ---------------------------------------------------------------------------


def _tool_bash(command: str) -> str:
    """Run a shell command after the LLM safety judge has signed off.

    Goes through ``bash -c`` so the agent can use pipes, redirection,
    command chaining, environment variables, and any installed binary.
    Each unique command is judged by ``judge_bash_command`` first; the
    judge's verdict is cached.
    """
    verdict = judge_bash_command(command)
    if verdict.verdict != "allow":
        return json.dumps(
            {"error": f"Blocked by safety judge: {verdict.reason}"}
        )

    logger.info("bash exec: %s", command[:200])
    try:
        result = subprocess.run(
            ["bash", "-c", command],
            capture_output=True,
            text=True,
            timeout=_BASH_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return json.dumps({"error": f"Command timed out after {_BASH_TIMEOUT_SECONDS}s"})
    except Exception as exc:
        return json.dumps({"error": f"Command failed: {exc}"})

    output = (result.stdout or "") + (result.stderr or "")
    truncated = len(output) > _BASH_OUTPUT_CAP_BYTES
    if truncated:
        output = output[:_BASH_OUTPUT_CAP_BYTES] + "\n[output truncated]"

    return json.dumps(
        {
            "output": output,
            "returncode": result.returncode,
            "truncated": truncated,
        }
    )


# ---------------------------------------------------------------------------
# Shared dispatcher
# ---------------------------------------------------------------------------


FILESYSTEM_TOOL_NAMES = frozenset(
    {"read_file", "glob_files", "grep_code", "list_dir", "bash"}
)


def execute_filesystem_tool(name: str, args: dict) -> str | None:
    """Dispatch one filesystem tool call by name. Returns ``None`` if name is unknown.

    The five tools are:
      - ``read_file(path, start_line?, end_line?)`` — read file lines.
      - ``glob_files(root, pattern)`` — glob under ``root``.
      - ``grep_code(root, pattern, file_glob?, max_results?)`` — recursive regex search.
      - ``list_dir(path, max_entries?)`` — list a directory.
      - ``bash(command)`` — judge-gated shell command.
    """
    if name == "read_file":
        return _tool_read_file(
            args["path"],
            args.get("start_line"),
            args.get("end_line"),
        )
    if name == "glob_files":
        return _tool_glob_files(args["root"], args["pattern"])
    if name == "grep_code":
        return _tool_grep_code(
            args["root"],
            args["pattern"],
            args.get("file_glob", "*.py"),
            args.get("max_results", 20),
        )
    if name == "list_dir":
        return _tool_list_dir(args["path"], args.get("max_entries", 100))
    if name == "bash":
        return _tool_bash(args["command"])
    return None
