import logging
from typing import Any, Dict

from guardrails.developer import check_logging_basicconfig_order
from guardrails.code_safety import check_code_safety, format_safety_feedback


logger = logging.getLogger(__name__)


def evaluate_guardrails(
    *,
    code_text: str,
    enable_logging_guard: bool,
    enable_code_safety: bool = True,
) -> Dict[str, Any]:
    """
    Run guardrails and return a report with decision.

    Args:
        code_text: Code to analyze
        enable_logging_guard: Check logging.basicConfig order (nice-to-have)
        enable_code_safety: LLM-based critical security check (recommended)

    Returns:
        Dict with:
        - decision: "proceed" or "block"
        - logging_check: Results from logging check
        - safety_check: Results from LLM safety check
    """
    guard_report: Dict[str, Any] = {
        "logging_check": {},
        "safety_check": {},
        "decision": "proceed",
    }

    # 1. Static logging.basicConfig order check (nice-to-have)
    log_check: Dict[str, Any] = {"status": "skipped", "reason": "disabled in config"}
    if enable_logging_guard:
        try:
            log_check = check_logging_basicconfig_order(code_text)
        except Exception:
            logger.exception("Logging AST check failed inside guardrails util")
            log_check = {"status": "error", "error": "exception in logging check"}
    guard_report["logging_check"] = log_check

    # 2. LLM-based critical security check (recommended)
    safety_check: Dict[str, Any] = {"decision": "proceed", "reason": "disabled in config"}
    if enable_code_safety:
        try:
            safety_check = check_code_safety(code_text)
            logger.info(
                "Code safety check completed: %s (confidence: %.2f)",
                safety_check["decision"],
                safety_check.get("confidence", 0.0)
            )
        except Exception:
            logger.exception("Code safety check failed inside guardrails util")
            # Fail open (lenient) - allow on error
            safety_check = {
                "decision": "proceed",
                "reasoning": "Exception during safety check, defaulting to allow",
                "violations": [],
                "suggested_fix": "",
                "confidence": 0.0,
            }
    guard_report["safety_check"] = safety_check

    # Overall decision: block if safety check blocks
    if safety_check.get("decision") == "block":
        guard_report["decision"] = "block"

    return guard_report


def build_block_summary(guard_report: Dict[str, Any]) -> str:
    """Compose a human-readable summary for a blocking guardrail decision."""
    lines: list[str] = []

    # Safety check (critical issues)
    safety_check = guard_report.get("safety_check", {})
    if safety_check.get("decision") == "block":
        # Use format_safety_feedback for consistent formatting
        safety_summary = format_safety_feedback(safety_check)
        lines.append(safety_summary)

    # Logging order (nice-to-have)
    try:
        log_check = guard_report.get("logging_check", {}) or {}
        if isinstance(log_check, dict) and log_check.get("status") == "fail":
            if lines:  # Add separator if safety check already added content
                lines.append("\n---\n")
            lines.append("Additional issue (non-blocking):")
            lines.append("- logging.basicConfig should be called before any top-level logging usage.")
    except Exception:
        pass

    return "\n".join(lines) if lines else "Guardrail check failed."


