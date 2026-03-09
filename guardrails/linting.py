"""Static lint guardrail checks for generated Python code."""

from __future__ import annotations

import io
import logging

logger = logging.getLogger(__name__)


def check_python_lint(code: str, filename: str = "generated_train.py") -> dict:
    """Run a fast lint check to catch undefined/unassigned variables before execution.

    Uses pyflakes when available; if pyflakes is unavailable, the check is skipped.
    """
    try:
        from pyflakes.api import check as pyflakes_check
        from pyflakes.reporter import Reporter
    except Exception:
        logger.warning("pyflakes is not available; skipping lint guardrail")
        return {
            "decision": "proceed",
            "status": "skipped",
            "reason": "pyflakes not installed",
            "errors": [],
        }

    stdout = io.StringIO()
    stderr = io.StringIO()
    reporter = Reporter(stdout, stderr)

    issue_count = pyflakes_check(code, filename=filename, reporter=reporter)
    output = "\n".join(x for x in (stdout.getvalue(), stderr.getvalue()) if x).strip()

    if issue_count > 0:
        errors = [line.strip() for line in output.splitlines() if line.strip()]
        logger.warning("Lint guardrail blocked execution: %d issue(s)", issue_count)
        return {
            "decision": "block",
            "status": "fail",
            "reason": f"Detected {issue_count} lint issue(s)",
            "errors": errors,
        }

    return {
        "decision": "proceed",
        "status": "pass",
        "reason": "No lint issues detected",
        "errors": [],
    }
