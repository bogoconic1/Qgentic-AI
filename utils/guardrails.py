import logging
from typing import Any, Dict

from guardrails.developer import check_logging_basicconfig_order


logger = logging.getLogger(__name__)

# this is a "nice to have" utility - not a "must have" for a successful pipeline
def evaluate_guardrails(
    *,
    code_text: str,
    enable_logging_guard: bool,
) -> Dict[str, Any]:
    """Run logging guardrail and return a report with decision."""
    guard_report: Dict[str, Any] = {
        "logging_check": {},
        "decision": "proceed",
    }

    # Static logging.basicConfig order check
    log_check: Dict[str, Any] = {"status": "skipped", "reason": "disabled in config"}
    if enable_logging_guard:
        try:
            log_check = check_logging_basicconfig_order(code_text)
        except Exception:
            logger.exception("Logging AST check failed inside guardrails util")
            log_check = {"status": "error", "error": "exception in logging check"}
    guard_report["logging_check"] = log_check

    return guard_report


def build_block_summary(guard_report: Dict[str, Any]) -> str:
    """Compose a human-readable summary for a blocking guardrail decision."""
    lines: list[str] = ["Guardrail check failed:"]

    # Logging order
    try:
        log_check = guard_report.get("logging_check", {}) or {}
        if isinstance(log_check, dict) and log_check.get("status") == "fail":
            lines.append("- logging.basicConfig must be called before any top-level logging usage.")
    except Exception:
        pass

    return "\n".join(lines)


