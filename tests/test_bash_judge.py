"""Test for the codex bash-safety judge."""

from __future__ import annotations

from schemas.bash_safety import BashSafetyVerdict
from tools import bash_judge
from utils.codex import CodexResult


def test_judge_failure_blocks_but_is_not_cached(monkeypatch, tmp_path):
    """A transient codex outage must block the command now and NOT poison it.

    If a failure verdict were cached, one rate-limit blip would permanently
    block that command for the rest of a 12-hour session — MainAgent would
    retry forever against a cache entry that never re-consults the judge.
    """
    monkeypatch.setattr(bash_judge, "_VERDICTS", {})
    calls = {"n": 0}

    def flaky_run_codex(prompt, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return CodexResult(returncode=1, stderr="rate limited")
        return CodexResult(
            returncode=0,
            stderr="",
            parsed=BashSafetyVerdict(verdict="allow", reason="read-only"),
        )

    monkeypatch.setattr(bash_judge, "run_codex", flaky_run_codex)

    first = bash_judge.judge_bash_command("ls -la", str(tmp_path))
    assert first.verdict == "block"

    second = bash_judge.judge_bash_command("ls -la", str(tmp_path))
    assert second.verdict == "allow", "failure verdict must not have been cached"
    assert calls["n"] == 2

    # The real verdict IS cached: a third call must not re-consult codex.
    third = bash_judge.judge_bash_command("ls -la", str(tmp_path))
    assert third.verdict == "allow"
    assert calls["n"] == 2
