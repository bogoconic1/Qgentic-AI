import logging
import os
import ast
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from project_config import get_config

from tools.helpers import call_llm_with_retry
from prompts.guardrails import (
    leakage_review as prompt_leakage_review,
    debug_sequence_review as prompt_debug_sequence_review,
)


load_dotenv()
logger = logging.getLogger(__name__)

_CONFIG = get_config()
_LLM_CFG = _CONFIG.get("llm")
_LEAKAGE_REVIEW_MODEL = _LLM_CFG.get("leakage_review_model")
_LEAKAGE_FOLLOWUP_MODEL = _LLM_CFG.get("leakage_followup_model")

# -----------------------------
# Guardrails: Static logging AST
# -----------------------------
LOG_LEVEL_METHODS = {"debug", "info", "warning", "error", "critical"}


# caution: I did not check logging guardrails - I assume its correct
def _is_logging_basicconfig_call(call: ast.Call) -> bool:
    """Return True if call node is logging.basicConfig(...)."""
    try:
        return (
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "logging"
            and call.func.attr == "basicConfig"
        )
    except Exception:
        return False


def _is_logging_direct_call(call: ast.Call) -> bool:
    """Detect logging.<level>(...) or logging.getLogger(...).<level>(...)."""
    # logging.<level>(...)
    if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
        if call.func.value.id == "logging" and call.func.attr in LOG_LEVEL_METHODS:
            return True
    # logging.getLogger(...).<level>(...)
    if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Call):
        inner = call.func.value
        if isinstance(inner.func, ast.Attribute) and isinstance(inner.func.value, ast.Name):
            if inner.func.value.id == "logging" and inner.func.attr == "getLogger":
                if call.func.attr in LOG_LEVEL_METHODS:
                    return True
    return False


def _collect_logger_aliases(nodes: List[ast.stmt]) -> List[str]:
    """Collect top-level variable names assigned from logging.getLogger(...)."""
    aliases: List[str] = []
    for node in nodes:
        try:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                val = node.value
                if isinstance(val.func, ast.Attribute) and isinstance(val.func.value, ast.Name):
                    if val.func.value.id == "logging" and val.func.attr == "getLogger":
                        for tgt in node.targets:
                            if isinstance(tgt, ast.Name):
                                aliases.append(tgt.id)
        except Exception:
            continue
    return aliases


def check_logging_basicconfig_order(code: str) -> Dict[str, Any]:
    """
    Ensure logging.basicConfig is executed before any top-level logging statements.

    Policy:
    - If any top-level logging statement (logging.<level> or logger.<level> where logger
      is assigned from logging.getLogger at top-level) appears before a top-level
      logging.basicConfig call, flag as FAIL.
    - If no top-level logging statements are present, PASS (cannot assert runtime order).
    - If basicConfig appears only under __main__ guard and there are top-level logging
      statements, FAIL.
    Returns a dict report with status, basicConfig_line, and violations.
    """

    try:
        module = ast.parse(code)
    except SyntaxError as exc:
        return {
            "status": "error",
            "error": f"SyntaxError while parsing: {exc}",
        }

    lines = code.splitlines()
    aliases = _collect_logger_aliases(module.body)

    basic_line: Optional[int] = None
    violations: List[Dict[str, Any]] = []

    for node in module.body:
        # Detect a top-level basicConfig call
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            if _is_logging_basicconfig_call(node.value) and basic_line is None:
                basic_line = getattr(node, "lineno", None)
                continue

        # Detect top-level logging statements prior to basicConfig
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            is_log_call = _is_logging_direct_call(call)

            # logger_alias.<level>(...)
            if not is_log_call and isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
                if call.func.attr in LOG_LEVEL_METHODS and call.func.value.id in aliases:
                    is_log_call = True

            if is_log_call and basic_line is None:
                lineno = getattr(node, "lineno", -1)
                code = lines[lineno - 1] if 0 < lineno <= len(lines) else ""
                violations.append({
                    "line": lineno,
                    "code": code.strip(),
                    "reason": "Logging call appears before logging.basicConfig at top-level.",
                })

    status = "pass"
    if violations:
        status = "fail"

    return {
        "status": status,
        "basicConfig_line": basic_line,
        "violations": violations,
    }


# ----------------------------------------------
# Guardrails: LLM-based data leakage risk review
# ----------------------------------------------
def llm_leakage_review(code: str) -> str:
    """
    Ask an LLM to review potential data leakage risks in the generated code.
    Uses the configured leakage review model via OpenRouter. Returns raw model text.
    """
    system_prompt = prompt_leakage_review()

    input_list = [
        {"role": "user", "content": "Python Training Script: \n\n" + code}
    ]

    response = call_llm_with_retry(
        model=_LEAKAGE_REVIEW_MODEL,
        instructions=system_prompt,
        tools=[],
        messages=input_list,
    )

    return response.output_text or ""



def llm_debug_sequence_review(code: str) -> str:
    """Ensure generated code runs DEBUG then FULL modes and guards against NaNs/zeros."""
    content_payload = prompt_debug_sequence_review()

    input_list = [
        {"role": "user", "content": "Python Training Pipeline: \n\n" + code}
    ]

    response = call_llm_with_retry(
        model=_LEAKAGE_FOLLOWUP_MODEL,
        instructions=content_payload,
        tools=[],
        messages=input_list,
    )

    return response.output_text or ""