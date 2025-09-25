import logging
import os
import ast
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from tools.helpers import call_llm_with_retry


load_dotenv()
logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.environ.get("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")


# -----------------------------
# Guardrails: Static logging AST
# -----------------------------
LOG_LEVEL_METHODS = {"debug", "info", "warning", "error", "critical"}


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


def check_logging_basicconfig_order(filepath: str) -> Dict[str, Any]:
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
        with open(filepath, "r") as f:
            source = f.read()
    except Exception as exc:
        return {
            "status": "error",
            "error": f"Failed to read file: {exc}",
        }

    try:
        module = ast.parse(source)
    except SyntaxError as exc:
        return {
            "status": "error",
            "error": f"SyntaxError while parsing: {exc}",
        }

    lines = source.splitlines()
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
def llm_leakage_review(task_description: str, code: str) -> str:
    """
    Ask an LLM to review potential data leakage risks in the generated code.
    Uses qwen/qwen3-next-80b-a3b-thinking via OpenRouter. Returns raw model text.
    """
    PROMPT = """
You are a senior ML engineer auditing a training script for data leakage.

Goals:
- Identify any train/test contamination risks (explicit or implicit), including:
  - Fitting transforms on combined train+test (scalers/encoders/PCA/imputers/etc.)
  - Using test labels or label-derived features anywhere
  - Feature selection or target encoding trained outside CV/OOF
  - Leaks via merges/aggregations that use future/test information
  - Wrong split strategy (e.g., random splits for time series)
  - Using KFold for classification when label is imbalanced (should prefer StratifiedKFold)
- Point to specific snippets and explain the rationale succinctly.

Output strictly as JSON with this schema:
{
  "severity": "block" | "warn" | "none",
  "findings": [
    {"rule_id": "<short_id>", "snippet": "<inline snippet or description>", "rationale": "<why this is risky>", "suggestion": "<how to fix>"}
  ]
}

Be concise and pragmatic; do not include prose outside JSON.
    """

    content_payload = (
        f"{PROMPT}\n\n"
        f"Task Description:\n" + "\"\"\"" + f"\n{task_description}\n" + "\"\"\"" + "\n\n"
        f"Code:\n" + "\"\"\"" + f"\n{code}\n" + "\"\"\""
    )

    messages = [
        {"role": "user", "content": content_payload}
    ]

    try:
        content = ""
        while content == "":
            completion = call_llm_with_retry(
                client,
                model="qwen/qwen3-next-80b-a3b-thinking",
                messages=messages,
            )
            msg = completion.choices[0].message
            content = msg.content or ""
        return content
    except Exception:
        logger.exception("Leakage LLM review failed")
        return '{"severity": "warn", "findings": [{"rule_id": "llm_error", "snippet": "N/A", "rationale": "LLM call failed", "suggestion": "Proceed with caution; add manual review."}]}'


def llm_debug_sequence_review(code: str) -> str:
    """Ensure generated code runs DEBUG then FULL modes and raises on NaNs."""

    PROMPT = """
You are reviewing a Python training pipeline for compliance with two runtime rules:
1. The script must execute with DEBUG=True (using a tiny subset/config) before it executes with DEBUG=False (full run). Both executions should happen sequentially in the same process.
2. For deep learning pipelines, if at the end of the 1st epoch of fold 0, the loss or metric is NaN, raise an Exception to stop the run immediately.

Examine the code and determine whether both requirements are satisfied.

Output strictly as JSON in this schema:
{
  "severity": "block" | "warn" | "none",
  "findings": [
    {"rule_id": "debug_sequence" | "nan_guard", "snippet": "<excerpt>", "rationale": "<why non-compliant>", "suggestion": "<how to fix>"}
  ]
}

Set severity="block" if either requirement is missing or incorrect. Use severity="warn" if unsure. Use severity="none" only when both requirements are clearly implemented.
Be concise; no extra prose outside JSON.
    """

    content_payload = (
        f"{PROMPT}\n\nCode:\n" + "\"\"\"" + f"\n{code}\n" + "\"\"\""
    )

    messages = [
        {"role": "user", "content": content_payload}
    ]

    try:
        content = ""
        while content == "":
            completion = call_llm_with_retry(
                client,
                model="qwen/qwen3-next-80b-a3b-instruct",
                messages=messages,
            )
            msg = completion.choices[0].message
            content = msg.content or ""
        return content
    except Exception:
        logger.exception("DEBUG sequence LLM review failed")
        return '{"severity": "warn", "findings": [{"rule_id": "llm_error", "snippet": "N/A", "rationale": "LLM call failed", "suggestion": "Manually verify DEBUG sequencing and NaN safeguards."}]}'

