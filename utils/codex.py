"""Shared `codex exec` invocation.

Subagent work is delegated to the `codex` CLI so it runs on the user's own
Codex subscription rather than a metered API key. Two consumers today: the
Deep Research subagent (``agents.researcher``) and the training log monitor
(``tools.developer``). They differ only in sandbox posture and whether they
want a typed verdict back:

- Research writes files, so it runs ``workspace-write`` with the research
  directory as its only writable root, and its deliverable is a file on disk.
- The monitor writes nothing — it reads a log tail and inspects processes —
  so it runs ``read-only`` and returns a schema-validated object.

Under both sandboxes shell commands have **no network**. Codex's built-in web
search still works, because it is executed by codex rather than by the shell.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ValidationError


logger = logging.getLogger(__name__)


class CodexUnavailableError(RuntimeError):
    """The `codex` CLI is missing, so subagent work cannot run."""


@dataclass(frozen=True)
class CodexResult:
    returncode: int
    stderr: str
    parsed: BaseModel | None = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def preflight() -> None:
    """Raise if the `codex` CLI is not on PATH.

    Called at agent construction so a misconfigured machine fails before the
    agent has burned a turn, rather than mid-run.
    """
    if shutil.which("codex") is None:
        raise CodexUnavailableError(
            "The `codex` CLI is not on PATH. Subagents run on codex so they "
            "draw on your subscription instead of API credit — install it and "
            "sign in before launching the agent."
        )


def run_codex(
    prompt: str,
    *,
    cwd: Path,
    sandbox: str,
    model: str,
    output_schema: type[BaseModel] | None = None,
    events_path: Path | None = None,
    timeout: float | None = None,
) -> CodexResult:
    """Run one `codex exec` to completion and return its outcome.

    Args:
        prompt: Full instruction text. Delivered on stdin rather than argv —
            prompts run to several KB once custom instructions are inlined.
        cwd: Passed as ``--cd``. Under ``workspace-write`` this is also the
            only writable root.
        sandbox: One of ``read-only``, ``workspace-write``,
            ``danger-full-access``.
        model: Passed as ``-m``. Explicit; codex's own default is not inherited.
        output_schema: When given, codex is constrained to emit JSON matching
            it and the validated model is returned on ``CodexResult.parsed``.
        events_path: When given, the raw ``--json`` event stream is written
            here for auditing.
        timeout: Seconds before the codex process is killed and the call
            reports failure. ``None`` — the researcher's setting — means run
            to completion, however long that takes: a cap would kill real deep
            work. Bounded callers (the log monitor) pass one so a wedged codex
            cannot block their caller's own watchdog loop.

    Returns:
        A ``CodexResult``. Non-zero exits, schema-validation failures, and
        timeouts are reported through it rather than raised — callers decide
        how much a failed subagent call matters.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        prompt_file = tmp_path / "prompt.md"
        prompt_file.write_text(prompt, encoding="utf-8")

        command = [
            "codex", "exec",
            "--cd", str(cwd),
            "--sandbox", sandbox,
            "--skip-git-repo-check",
            "-m", model,
        ]

        last_message = tmp_path / "last_message.json"
        if output_schema is not None:
            schema_file = tmp_path / "schema.json"
            schema = output_schema.model_json_schema()
            schema["additionalProperties"] = False
            schema_file.write_text(json.dumps(schema), encoding="utf-8")
            command += [
                "--output-schema", str(schema_file),
                "-o", str(last_message),
            ]
        if events_path is not None:
            command.append("--json")
        command.append("-")

        stdout = (
            events_path.open("w", encoding="utf-8")
            if events_path is not None
            else subprocess.DEVNULL
        )
        try:
            with prompt_file.open("r", encoding="utf-8") as stdin:
                proc = subprocess.run(
                    command,
                    stdin=stdin,
                    stdout=stdout,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                    timeout=timeout,
                )
        except subprocess.TimeoutExpired:
            # subprocess.run has already killed the child before re-raising.
            logger.error("codex exec exceeded %.0fs timeout — killed", timeout)
            return CodexResult(
                returncode=-1,
                stderr=f"codex exec killed after {timeout:.0f}s timeout",
            )
        finally:
            if events_path is not None:
                stdout.close()

        stderr = proc.stderr or ""
        if proc.returncode != 0:
            logger.error("codex exec exited %d: %s", proc.returncode, stderr)
            return CodexResult(returncode=proc.returncode, stderr=stderr)

        parsed = None
        if output_schema is not None:
            try:
                parsed = output_schema.model_validate_json(
                    last_message.read_text(encoding="utf-8")
                )
            except (OSError, ValidationError, json.JSONDecodeError) as exc:
                logger.error("codex output did not match %s: %s",
                             output_schema.__name__, exc)
                return CodexResult(returncode=proc.returncode, stderr=str(exc))

        return CodexResult(returncode=proc.returncode, stderr=stderr, parsed=parsed)
