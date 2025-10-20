import json
import logging
from typing import Any, Dict

from guardrails.developer import (
    check_logging_basicconfig_order,
    llm_debug_sequence_review,
    llm_leakage_review,
)


logger = logging.getLogger(__name__)

# this is a "nice to have" utility - not a "must have" for a successful pipeline
def evaluate_guardrails(
    *,
    description: str,
    code_text: str,
    enable_logging_guard: bool,
    enable_nan_guard: bool,
    enable_leakage_guard: bool,
) -> Dict[str, Any]:
    """Run all configured guardrails and return a unified report with decision."""
    guard_report: Dict[str, Any] = {
        "logging_check": {},
        "debug_sequence_check": {},
        "leakage_check": {},
        "decision": "proceed",
    }

    # 1) Static logging.basicConfig order
    log_check: Dict[str, Any] = {"status": "skipped", "reason": "disabled in config"}
    if enable_logging_guard:
        try:
            log_check = check_logging_basicconfig_order(code_text)
        except Exception:
            logger.exception("Logging AST check failed inside guardrails util")
            log_check = {"status": "error", "error": "exception in logging check"}
    guard_report["logging_check"] = log_check

    # 2) DEBUG sequencing + NaN guard (LLM)
    debug_json_text: str | Dict[str, Any] = {"status": "skipped", "reason": "disabled in config"}
    if enable_nan_guard:
        try:
            debug_json_text = llm_debug_sequence_review(code_text)
        except Exception:
            logger.exception("DEBUG sequencing guardrail call failed")
            debug_json_text = '{"severity":"warn","findings":[{"rule_id":"llm_error","snippet":"N/A","rationale":"LLM call failed","suggestion":"Manually ensure DEBUG runs before FULL and loss/metric NaN or zero values raise exceptions."}]}'
        try:
            parsed = json.loads(debug_json_text) if isinstance(debug_json_text, str) else debug_json_text
            if parsed.get("severity") == "block":
                guard_report["decision"] = "block"
        except Exception:
            # ignore malformed JSON for decision
            pass
    guard_report["debug_sequence_check"] = debug_json_text

    # 3) Leakage review (LLM)
    leakage_json_text: str | Dict[str, Any] = {"status": "skipped", "reason": "disabled in config"}
    if enable_leakage_guard:
        try:
            leakage_json_text = llm_leakage_review(description, code_text)
        except Exception:
            logger.exception("LLM leakage review call failed")
            leakage_json_text = '{"severity":"warn","findings":[{"rule_id":"llm_error","snippet":"N/A","rationale":"LLM call failed","suggestion":"Proceed with caution"}]}'
        try:
            parsed = json.loads(leakage_json_text) if isinstance(leakage_json_text, str) else leakage_json_text
            if parsed.get("severity") == "block":
                guard_report["decision"] = "block"
        except Exception:
            # ignore malformed JSON for decision
            pass
    guard_report["leakage_check"] = leakage_json_text

    return guard_report


def build_block_summary(guard_report: Dict[str, Any]) -> str:
    """Compose a human-readable summary for a blocking guardrail decision."""
    lines: list[str] = ["Guardrail checks failed:"]

    # Logging order
    try:
        log_check = guard_report.get("logging_check", {}) or {}
        if isinstance(log_check, dict) and log_check.get("status") == "fail":
            lines.append("- logging.basicConfig must be called before any top-level logging usage.")
    except Exception:
        pass

    # DEBUG sequencing
    try:
        raw_debug = guard_report.get("debug_sequence_check", "{}")
        parsed_debug = json.loads(raw_debug.strip()) if isinstance(raw_debug, str) else (raw_debug or {})
        if parsed_debug.get("severity") == "block":
            lines.append("- Ensure the script runs DEBUG mode before FULL mode and raises an Exception when loss/metric is NaN or 0.")
            findings = parsed_debug.get("findings", [])
            if findings:
                lines.append("\nDEBUG sequencing findings:")
                for idx, finding in enumerate(findings, start=1):
                    lines.append(
                        f"{idx}. rule_id={finding.get('rule_id', 'unknown')}\n   - snippet: {finding.get('snippet', '')}\n   - rationale: {finding.get('rationale', '')}\n   - suggestion: {finding.get('suggestion', '')}"
                    )
    except Exception:
        try:
            lines.append("- DEBUG sequencing reviewer returned non-JSON content:")
            lines.append(str(guard_report.get("debug_sequence_check")))
        except Exception:
            pass

    # Leakage
    try:
        raw_leak = guard_report.get("leakage_check", "{}")
        parsed_leak = json.loads(raw_leak.strip()) if isinstance(raw_leak, str) else (raw_leak or {})
        sev = parsed_leak.get("severity")
        if sev == "block":
            lines.append("- Potential data leakage risks detected. Please fix as suggested.")
            findings = parsed_leak.get("findings", [])
            if findings:
                lines.append("\nLeakage reviewer findings:")
                for idx, f in enumerate(findings, start=1):
                    rule_id = f.get("rule_id", "unknown")
                    snippet = f.get("snippet", "")
                    rationale = f.get("rationale", "")
                    suggestion = f.get("suggestion", "")
                    lines.append(
                        f"{idx}. rule_id={rule_id}\n   - snippet: {snippet}\n   - rationale: {rationale}\n   - suggestion: {suggestion}"
                    )
    except Exception:
        try:
            lines.append("- Data leakage reviewer returned non-JSON content:")
            lines.append(str(guard_report.get("leakage_check")))
        except Exception:
            pass

    return "\n".join(lines)


