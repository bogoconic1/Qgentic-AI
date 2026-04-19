"""Filesystem tools for MainAgent.

Plain helper functions (no agent wrapper). Read tools scope their paths to the
module-level allowed-roots list (from ``runtime.explore_allowed_roots`` plus the
active interpreter's ``site-packages``). ``tool_bash`` runs arbitrary shell
commands with no allowlist — MainAgent owns its workspace.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sysconfig
from pathlib import Path

from project_config import get_config


logger = logging.getLogger(__name__)


_CONFIG = get_config()
_RUNTIME_CFG = _CONFIG["runtime"]
_USER_ROOTS = [Path(r).resolve() for r in _RUNTIME_CFG["explore_allowed_roots"]]
_SITE_PACKAGES = Path(sysconfig.get_path("purelib")).resolve()
_ALLOWED_ROOTS: list[Path] = _USER_ROOTS + [_SITE_PACKAGES]


_BASH_OUTPUT_CAP_BYTES = 8 * 1024
_BASH_TIMEOUT_SECONDS = 600


def _validate_path(path: Path) -> str | None:
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


def tool_read_file(
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


def tool_glob_files(root: str, pattern: str) -> str:
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


def tool_grep_code(
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


def tool_list_dir(path: str, max_entries: int = 100) -> str:
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


def tool_bash(command: str) -> str:
    """Run an arbitrary shell command via ``bash -c``. No allowlist.

    Returns combined stdout+stderr (capped at 8KB), the exit code, and a
    truncation flag. Timeouts after ``_BASH_TIMEOUT_SECONDS`` seconds.
    """
    if not command or not command.strip():
        return json.dumps({"error": "Empty command"})

    try:
        result = subprocess.run(
            command,
            shell=True,
            executable="/bin/bash",
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
