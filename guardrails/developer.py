import logging
import ast

from dotenv import load_dotenv
from project_config import get_config

from tools.helpers import call_llm
from prompts.guardrails import leakage_review as prompt_leakage_review
from utils.llm_utils import append_message
from schemas.guardrails import LeakageReviewResponse


load_dotenv()
logger = logging.getLogger(__name__)

_CONFIG = get_config()
_LLM_CFG = _CONFIG["llm"]
_LEAKAGE_REVIEW_MODEL = _LLM_CFG["leakage_review_model"]

# -----------------------------
# Guardrails: Static logging AST
# -----------------------------
LOG_LEVEL_METHODS = {"debug", "info", "warning", "error", "critical"}


# caution: I did not check logging guardrails - I assume its correct
def _is_logging_basicconfig_call(call: ast.Call) -> bool:
    """Return True if call node is logging.basicConfig(...)."""
    return (
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "logging"
        and call.func.attr == "basicConfig"
    )


def _is_logging_direct_call(call: ast.Call) -> bool:
    """Detect logging.<level>(...) or logging.getLogger(...).<level>(...)."""
    # logging.<level>(...)
    if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
        if call.func.value.id == "logging" and call.func.attr in LOG_LEVEL_METHODS:
            return True
    # logging.getLogger(...).<level>(...)
    if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Call):
        inner = call.func.value
        if isinstance(inner.func, ast.Attribute) and isinstance(
            inner.func.value, ast.Name
        ):
            if inner.func.value.id == "logging" and inner.func.attr == "getLogger":
                if call.func.attr in LOG_LEVEL_METHODS:
                    return True
    return False


def _collect_logger_aliases(nodes: list[ast.stmt]) -> list[str]:
    """Collect top-level variable names assigned from logging.getLogger(...)."""
    aliases: list[str] = []
    for node in nodes:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            val = node.value
            if isinstance(val.func, ast.Attribute) and isinstance(
                val.func.value, ast.Name
            ):
                if val.func.value.id == "logging" and val.func.attr == "getLogger":
                    for tgt in node.targets:
                        if isinstance(tgt, ast.Name):
                            aliases.append(tgt.id)
    return aliases


def check_logging_basicconfig_order(code: str) -> dict:
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
    module = ast.parse(code)
    lines = code.splitlines()
    aliases = _collect_logger_aliases(module.body)

    basic_line: int | None = None
    violations: list[dict] = []

    for node in module.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            if _is_logging_basicconfig_call(node.value) and basic_line is None:
                basic_line = node.lineno
                continue

        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            is_log_call = _is_logging_direct_call(call)

            # logger_alias.<level>(...)
            if (
                not is_log_call
                and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
            ):
                if (
                    call.func.attr in LOG_LEVEL_METHODS
                    and call.func.value.id in aliases
                ):
                    is_log_call = True

            if is_log_call and basic_line is None:
                lineno = node.lineno
                code = lines[lineno - 1] if 0 < lineno <= len(lines) else ""
                violations.append(
                    {
                        "line": lineno,
                        "code": code.strip(),
                        "reason": "Logging call appears before logging.basicConfig at top-level.",
                    }
                )

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
def llm_leakage_review(code: str) -> LeakageReviewResponse:
    """
    Ask an LLM to review potential data leakage risks in the generated code.
    Uses the configured leakage review model. Returns structured LeakageReviewResponse.
    """
    system_prompt = prompt_leakage_review()
    messages = [append_message("user", "Python Training Script: \n\n" + code)]

    return call_llm(
        model=_LEAKAGE_REVIEW_MODEL,
        system_instruction=system_prompt,
        function_declarations=[],
        messages=messages,
        text_format=LeakageReviewResponse,
    )
