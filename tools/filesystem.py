"""Shared filesystem tools (read + grep + glob + list + bash + write + edit).

A single palette of file/system tools usable by every sub-agent.

**Reads** (``_tool_read_file``, ``_tool_glob_files``, ``_tool_grep_code``,
``_tool_list_dir``) run wide — any path on the workspace is fair game.
The agent legitimately needs to inspect baselines, library source, and
sibling artifacts.

**Writes** (``_tool_write_file``, ``_tool_edit_file``) are pinned to the
caller's ``writable_root`` (per-agent: ``developer_v{N}/`` for the
DeveloperAgent, ``research_<N>/`` for the ResearcherAgent, the run dir
for the MainAgent). Targets that resolve outside ``writable_root`` are
rejected before any I/O happens.

**Bash** (``_tool_bash``) runs through ``bash -c`` with
``cwd=writable_root``, so relative-path writes naturally land inside
the agent's directory. Each command is judged by ``judge_bash_command``
(LLM-as-judge), which rejects ``cd``/``pushd``/``chdir`` (no escape from
cwd) and writes whose targets resolve outside ``writable_root``, on
top of the existing destructive-op block list.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from project_config import get_config
from tools.bash_judge import judge_bash_command


logger = logging.getLogger(__name__)


_RUNTIME_CFG = get_config()["runtime"]


_BASH_OUTPUT_CAP_BYTES = 8 * 1024
_BASH_TIMEOUT_SECONDS = int(_RUNTIME_CFG.get("bash_timeout_seconds", 600))


# ---------------------------------------------------------------------------
# writable_root containment check (used by write_file / edit_file only)
# ---------------------------------------------------------------------------


def _inside(target: Path, writable_root: Path) -> bool:
    """Return True iff ``target`` resolves to a path under ``writable_root``."""
    try:
        target.resolve().relative_to(writable_root.resolve())
        return True
    except ValueError:
        return False


def _outside_writable_root_error(target: Path, writable_root: Path) -> str:
    return json.dumps(
        {
            "error": (
                f"Path {target.resolve()} is outside writable_root "
                f"{writable_root.resolve()}. Writes must land inside the "
                "agent's working directory."
            )
        }
    )


# ---------------------------------------------------------------------------
# Read-only filesystem tools
# ---------------------------------------------------------------------------


def _tool_read_file(
    path: str, start_line: int | None = None, end_line: int | None = None
) -> str:
    full_path = Path(path)
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


def _tool_bash(command: str, *, writable_root: Path) -> str:
    """Run a shell command after the LLM safety judge has signed off.

    Goes through ``bash -c`` with ``cwd=writable_root`` so the agent can
    use pipes, redirection, command chaining, environment variables, and
    any installed binary, and so relative-path writes naturally land
    inside the agent's directory. Each unique command is judged by
    ``judge_bash_command(command, writable_root)`` first; the judge's
    verdict is cached.
    """
    verdict = judge_bash_command(command, str(writable_root))
    if verdict.verdict != "allow":
        return json.dumps(
            {"error": f"Blocked by safety judge: {verdict.reason}"}
        )

    logger.info("bash exec (cwd=%s): %s", writable_root, command[:200])
    try:
        result = subprocess.run(
            ["bash", "-c", command],
            capture_output=True,
            text=True,
            timeout=_BASH_TIMEOUT_SECONDS,
            cwd=str(writable_root),
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
# Write + Edit (modeled on claude-code FileWriteTool / FileEditTool)
# ---------------------------------------------------------------------------

# When a model emits straight quotes but the file uses typographic curly
# quotes (or vice versa), the matcher should still locate the substring.
# Mirrors FileEditTool/utils.ts.
_LEFT_SINGLE_CURLY = "‘"
_RIGHT_SINGLE_CURLY = "’"
_LEFT_DOUBLE_CURLY = "“"
_RIGHT_DOUBLE_CURLY = "”"

_WRITE_FILE_MAX_BYTES = 5 * 1024 * 1024


def _normalize_quotes(s: str) -> str:
    return (
        s.replace(_LEFT_SINGLE_CURLY, "'")
        .replace(_RIGHT_SINGLE_CURLY, "'")
        .replace(_LEFT_DOUBLE_CURLY, '"')
        .replace(_RIGHT_DOUBLE_CURLY, '"')
    )


def _find_actual_string(file_content: str, search: str) -> str | None:
    if search in file_content:
        return search
    normalized_search = _normalize_quotes(search)
    normalized_file = _normalize_quotes(file_content)
    idx = normalized_file.find(normalized_search)
    if idx == -1:
        return None
    return file_content[idx : idx + len(search)]


def _apply_edit_to_file(content: str, old: str, new: str, replace_all: bool) -> str:
    if new != "":
        return content.replace(old, new) if replace_all else content.replace(old, new, 1)
    if not old.endswith("\n") and (old + "\n") in content:
        old_eff = old + "\n"
    else:
        old_eff = old
    return content.replace(old_eff, "") if replace_all else content.replace(old_eff, "", 1)


def _tool_write_file(path: str, content: str, *, writable_root: Path) -> str:
    full_path = Path(path)
    if not _inside(full_path, writable_root):
        return _outside_writable_root_error(full_path, writable_root)
    resolved = full_path.resolve()
    if resolved.exists() and resolved.is_dir():
        return json.dumps({"error": f"Path is a directory: {path}"})
    if len(content.encode("utf-8")) > _WRITE_FILE_MAX_BYTES:
        return json.dumps({"error": f"Content exceeds {_WRITE_FILE_MAX_BYTES} byte cap"})
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        existed = resolved.exists()
        resolved.write_text(content, encoding="utf-8")
    except Exception as exc:
        return json.dumps({"error": f"Write failed: {exc}"})
    return json.dumps(
        {
            "type": "update" if existed else "create",
            "path": str(resolved),
            "bytes_written": len(content.encode("utf-8")),
        }
    )


def _tool_edit_file(
    path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
    *,
    writable_root: Path,
) -> str:
    full_path = Path(path)
    if not _inside(full_path, writable_root):
        return _outside_writable_root_error(full_path, writable_root)
    resolved = full_path.resolve()
    if old_string == new_string:
        return json.dumps(
            {"error": "No changes: old_string and new_string are identical"}
        )

    if not resolved.exists():
        if old_string == "":
            try:
                resolved.parent.mkdir(parents=True, exist_ok=True)
                resolved.write_text(new_string, encoding="utf-8")
            except Exception as exc:
                return json.dumps({"error": f"Write failed: {exc}"})
            return json.dumps(
                {
                    "type": "create",
                    "path": str(resolved),
                    "bytes_written": len(new_string.encode("utf-8")),
                }
            )
        return json.dumps({"error": f"File not found: {path}"})

    if not resolved.is_file():
        return json.dumps({"error": f"Not a regular file: {path}"})

    try:
        content = resolved.read_text(encoding="utf-8")
    except Exception as exc:
        return json.dumps({"error": f"Failed to read file: {exc}"})

    actual_old = _find_actual_string(content, old_string)
    if actual_old is None:
        return json.dumps({"error": "String to replace not found in file."})

    matches = content.count(actual_old)
    if matches > 1 and not replace_all:
        return json.dumps(
            {
                "error": (
                    f"Found {matches} matches of the old_string, but replace_all is false. "
                    "Either set replace_all=true or provide more context to make old_string unique."
                )
            }
        )

    updated = _apply_edit_to_file(content, actual_old, new_string, replace_all)
    if updated == content:
        return json.dumps({"error": "Edit produced no changes."})

    try:
        resolved.write_text(updated, encoding="utf-8")
    except Exception as exc:
        return json.dumps({"error": f"Write failed: {exc}"})

    return json.dumps(
        {
            "type": "update",
            "path": str(resolved),
            "matches_replaced": matches if replace_all else 1,
        }
    )


# ---------------------------------------------------------------------------
# Shared dispatcher
# ---------------------------------------------------------------------------


FILESYSTEM_TOOL_NAMES = frozenset(
    {
        "read_file",
        "glob_files",
        "grep_code",
        "list_dir",
        "bash",
        "write_file",
        "edit_file",
    }
)


def execute_filesystem_tool(
    name: str, args: dict, *, writable_root: Path
) -> str | None:
    """Dispatch one filesystem tool call by name. Returns ``None`` if name is unknown.

    Reads (``read_file`` / ``glob_files`` / ``grep_code`` / ``list_dir``)
    run wide and ignore ``writable_root``. Writes (``write_file`` /
    ``edit_file``) and ``bash`` are pinned to ``writable_root``: the
    write helpers reject paths that resolve outside it; bash runs with
    ``cwd=writable_root`` and is gated by ``judge_bash_command``.

    The seven tools are:
      - ``read_file(path, start_line?, end_line?)`` — read file lines.
      - ``glob_files(root, pattern)`` — glob under ``root``.
      - ``grep_code(root, pattern, file_glob?, max_results?)`` — recursive regex search.
      - ``list_dir(path, max_entries?)`` — list a directory.
      - ``bash(command)`` — judge-gated shell command.
      - ``write_file(path, content)`` — write/overwrite a file.
      - ``edit_file(path, old_string, new_string, replace_all?)`` — exact-string replacement.
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
        return _tool_bash(args["command"], writable_root=writable_root)
    if name == "write_file":
        return _tool_write_file(
            args["path"], args["content"], writable_root=writable_root
        )
    if name == "edit_file":
        return _tool_edit_file(
            args["path"],
            args["old_string"],
            args["new_string"],
            args.get("replace_all", False),
            writable_root=writable_root,
        )
    return None
