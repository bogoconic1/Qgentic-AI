import logging
from typing import Any, Dict

from guardrails.developer import check_logging_basicconfig_order, llm_leakage_review
from guardrails.code_safety import check_code_safety, format_safety_feedback
from schemas.guardrails import LeakageReviewResponse


logger = logging.getLogger(__name__)

def evaluate_guardrails(
    *,
    code_text: str,
    enable_logging_guard: bool,
    enable_leakage_guard: bool,
    enable_code_safety: bool,
) -> Dict[str, Any]:
    """Run all configured guardrails and return a unified report with decision."""
    guard_report: Dict[str, Any] = {
        "logging_check": {},
        "leakage_check": {},
        "safety_check": {},
        "decision": "proceed",
    }

    log_check: Dict[str, Any] = {"status": "skipped", "reason": "disabled in config"}
    if enable_logging_guard:
        try:
            log_check = check_logging_basicconfig_order(code_text)
        except Exception:
            logger.exception("Logging AST check failed inside guardrails util")
            log_check = {"status": "error", "error": "exception in logging check"}
    guard_report["logging_check"] = log_check

    leakage_result: LeakageReviewResponse | Dict[str, Any] = {"status": "skipped", "reason": "disabled in config"}
    if enable_leakage_guard:
        try:
            leakage_result = llm_leakage_review(code_text)
            if leakage_result.severity == "block":
                guard_report["decision"] = "block"
        except Exception:
            logger.exception("LLM leakage review call failed")
            leakage_result = {"status": "error", "error": "LLM call failed"}
    guard_report["leakage_check"] = leakage_result

    safety_check: Dict[str, Any] = {"decision": "proceed", "reason": "disabled in config"}
    if enable_code_safety:
        try:
            safety_check = check_code_safety(code_text)
            if safety_check.get("decision") == "block":
                guard_report["decision"] = "block"
        except Exception:
            logger.exception("Code safety check failed inside guardrails util")
            safety_check = {
                "decision": "proceed",
                "reasoning": "Exception during safety check, defaulting to allow",
                "violations": [],
                "suggested_fix": "",
                "confidence": 0.0,
            }
    guard_report["safety_check"] = safety_check

    return guard_report


def build_block_summary(guard_report: Dict[str, Any]) -> str:
    """Compose a human-readable summary for a blocking guardrail decision."""
    lines: list[str] = []

    safety_check = guard_report.get("safety_check", {})
    if safety_check.get("decision") == "block":
        safety_summary = format_safety_feedback(safety_check)
        lines.append(safety_summary)

    try:
        leakage_check = guard_report.get("leakage_check")
        if isinstance(leakage_check, LeakageReviewResponse):
            if leakage_check.severity == "block":
                if lines:
                    lines.append("\n---\n")
                lines.append("Potential data leakage risks detected. Please fix as suggested.")
                if leakage_check.findings:
                    lines.append("\nLeakage reviewer findings:")
                    for idx, f in enumerate(leakage_check.findings, start=1):
                        lines.append(
                            f"{idx}. rule_id={f.rule_id}\n   - snippet: {f.snippet}\n   - rationale: {f.rationale}\n   - suggestion: {f.suggestion}"
                        )
        elif isinstance(leakage_check, dict) and leakage_check.get("status") == "error":
            lines.append("- Data leakage review failed: " + leakage_check.get("error", "unknown error"))
    except Exception:
        pass

    # Logging order (non-blocking, informational)
    try:
        log_check = guard_report.get("logging_check", {}) or {}
        if isinstance(log_check, dict) and log_check.get("status") == "fail":
            if lines:
                lines.append("\n---\n")
            lines.append("Additional issue (non-blocking):")
            lines.append("- logging.basicConfig should be called before any top-level logging usage.")
    except Exception:
        pass

    return "\n".join(lines) if lines else "Guardrail check failed."


