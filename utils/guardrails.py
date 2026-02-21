import logging

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
) -> dict:
    """Run all configured guardrails and return a unified report with decision."""
    guard_report: dict = {
        "logging_check": {},
        "leakage_check": {},
        "safety_check": {},
        "decision": "proceed",
    }

    log_check: dict = {"status": "skipped", "reason": "disabled in config"}
    if enable_logging_guard:
        log_check = check_logging_basicconfig_order(code_text)
    guard_report["logging_check"] = log_check

    leakage_result: LeakageReviewResponse | dict = {
        "status": "skipped",
        "reason": "disabled in config",
    }
    if enable_leakage_guard:
        leakage_result = llm_leakage_review(code_text)
        if leakage_result.severity == "block":
            guard_report["decision"] = "block"
    guard_report["leakage_check"] = leakage_result

    safety_check: dict = {"decision": "proceed", "reason": "disabled in config"}
    if enable_code_safety:
        safety_check = check_code_safety(code_text)
        if safety_check["decision"] == "block":
            guard_report["decision"] = "block"
    guard_report["safety_check"] = safety_check

    return guard_report


def build_block_summary(guard_report: dict) -> str:
    """Compose a human-readable summary for a blocking guardrail decision."""
    lines: list[str] = []

    safety_check = guard_report["safety_check"]
    if safety_check["decision"] == "block":
        safety_summary = format_safety_feedback(safety_check)
        lines.append(safety_summary)

    leakage_check = guard_report["leakage_check"]
    if (
        isinstance(leakage_check, LeakageReviewResponse)
        and leakage_check.severity == "block"
    ):
        if lines:
            lines.append("\n---\n")
        lines.append("Potential data leakage risks detected. Please fix as suggested.")
        if leakage_check.findings:
            lines.append("\nLeakage reviewer findings:")
            for idx, f in enumerate(leakage_check.findings, start=1):
                lines.append(
                    f"{idx}. rule_id={f.rule_id}\n   - snippet: {f.snippet}\n   - rationale: {f.rationale}\n   - suggestion: {f.suggestion}"
                )

    log_check = guard_report["logging_check"]
    if log_check["status"] == "fail":
        if lines:
            lines.append("\n---\n")
        lines.append("Additional issue (non-blocking):")
        lines.append(
            "- logging.basicConfig should be called before any top-level logging usage."
        )

    return "\n".join(lines) if lines else "Guardrail check failed."
