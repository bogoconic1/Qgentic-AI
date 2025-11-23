"""
LLM-based code safety checker for generated code.
Uses Gemini 2.5 Flash for fast, intelligent security analysis.

Focuses on critical security issues only:
- Code execution (eval, exec, compile)
- Command injection (os.system, subprocess with shell=True)
- Credential leakage

Does NOT restrict file system access - trusts LLM code generation.
"""

import logging
from typing import Dict, Any

from tools.helpers import call_llm_with_retry_google
from schemas.guardrails import CodeSafetyCheck
from prompts.guardrails import code_safety_system, code_safety_user
from project_config import get_config

logger = logging.getLogger(__name__)

_CONFIG = get_config()
_LLM_CFG = _CONFIG.get("llm")
_SAFETY_MODEL = "gemini-2.5-flash"  # Latest Flash model (2025)


def check_code_safety(code: str) -> Dict[str, Any]:
    """
    Check if generated code is safe to execute using LLM analysis.

    Focuses on critical security issues only (eval, exec, command injection, secrets).
    Does NOT restrict file system access.

    Args:
        code: Python code to analyze

    Returns:
        Dict with:
        - decision: "proceed" or "block"
        - reasoning: Explanation
        - violations: List of critical issues
        - suggested_fix: How to fix (if blocked)
        - confidence: 0.0-1.0
    """
    logger.info("Running LLM-based code safety check (critical issues only)...")

    try:
        system_prompt = code_safety_system()
        user_prompt = code_safety_user(code)

        response = call_llm_with_retry_google(
            model=_SAFETY_MODEL,
            system_instruction=system_prompt,
            messages=user_prompt,
            text_format=CodeSafetyCheck,
            temperature=0.0,  # Deterministic for security
            max_retries=3,
            enable_google_search=False,
            top_p=0.95,
            thinking_budget=None,
        )

        # Parse structured output
        if response and hasattr(response, 'decision'):
            decision = "proceed" if response.decision == "allow" else "block"

            logger.info(
                "Code safety check: %s (confidence: %.2f)",
                decision,
                response.confidence
            )

            if response.violations:
                logger.warning(
                    "CRITICAL security violations found: %s",
                    ", ".join(response.violations)
                )

            return {
                "decision": decision,
                "reasoning": response.reasoning,
                "violations": response.violations,
                "suggested_fix": response.suggested_fix,
                "confidence": response.confidence,
            }

        # Fallback if structured output fails
        logger.warning("Structured output parsing failed, defaulting to ALLOW (lenient)")
        return {
            "decision": "proceed",
            "reasoning": "Failed to parse LLM response, defaulting to allow",
            "violations": [],
            "suggested_fix": "",
            "confidence": 0.5,
        }

    except Exception as e:
        logger.exception("Code safety check failed with exception")
        # Fail open (lenient) - allow execution on errors
        logger.warning("Allowing code execution despite safety check failure")
        return {
            "decision": "proceed",
            "reasoning": f"Safety check error: {str(e)}, defaulting to allow",
            "violations": [],
            "suggested_fix": "",
            "confidence": 0.0,
        }


def format_safety_feedback(safety_result: Dict[str, Any]) -> str:
    """
    Format safety check results as feedback for the LLM to fix issues.

    Args:
        safety_result: Result from check_code_safety()

    Returns:
        Formatted message for LLM
    """
    if safety_result["decision"] == "proceed":
        return "Code passed security checks."

    lines = ["Code BLOCKED by security guardrails (CRITICAL ISSUES):\n"]

    # Add reasoning
    lines.append(f"**Reasoning:** {safety_result['reasoning']}\n")

    # List violations
    if safety_result["violations"]:
        lines.append("**Critical Violations:**")
        for violation in safety_result["violations"]:
            lines.append(f"  - {violation}")
        lines.append("")

    # Add fix suggestion
    if safety_result["suggested_fix"]:
        lines.append(f"**How to fix:**\n{safety_result['suggested_fix']}\n")

    lines.append("Please regenerate the code addressing these CRITICAL security issues.")

    return "\n".join(lines)
