import logging
import ast
import sys

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
    Ensure logging.basicConfig is executed before any top-level logging
    statements AND before any third-party imports.

    Policy:
    - If any top-level logging statement (logging.<level> or logger.<level> where logger
      is assigned from logging.getLogger at top-level) appears before a top-level
      logging.basicConfig call, flag as FAIL.
    - If logging.basicConfig appears after a third-party import (any import not in
      sys.stdlib_module_names), flag as FAIL — third-party libraries may configure
      logging on import, making a later basicConfig a no-op.
    - If no top-level logging statements are present, PASS (cannot assert runtime order).
    - If basicConfig appears only under __main__ guard and there are top-level logging
      statements, FAIL.
    Returns a dict report with status, basicConfig_line, and violations.
    """
    try:
        module = ast.parse(code)
    except SyntaxError as exc:
        return {
            "status": "fail",
            "basicConfig_line": None,
            "violations": [
                {
                    "line": getattr(exc, "lineno", 0) or 0,
                    "reason": f"syntax error while parsing generated code: {exc}",
                }
            ],
        }
    lines = code.splitlines()
    aliases = _collect_logger_aliases(module.body)

    basic_line: int | None = None
    logging_imported_line: int | None = None
    first_thirdparty: dict | None = None
    violations: list[dict] = []

    for node in module.body:
        # Track `import logging`
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "logging" and logging_imported_line is None:
                    logging_imported_line = node.lineno
                if (
                    logging_imported_line is not None
                    and first_thirdparty is None
                    and alias.name.split(".")[0] not in sys.stdlib_module_names
                ):
                    first_thirdparty = {
                        "module": alias.name,
                        "line": node.lineno,
                    }

        # Track `from X import ...`
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            top_level = node.module.split(".")[0]
            if top_level == "logging" and logging_imported_line is None:
                logging_imported_line = node.lineno
            if (
                logging_imported_line is not None
                and first_thirdparty is None
                and top_level not in sys.stdlib_module_names
            ):
                first_thirdparty = {
                    "module": node.module,
                    "line": node.lineno,
                }

        # Detect logging.basicConfig(...)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            if _is_logging_basicconfig_call(node.value) and basic_line is None:
                basic_line = node.lineno
                continue

        # Detect top-level logging calls before basicConfig
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

    # Check if basicConfig comes after a third-party import
    if (
        basic_line is not None
        and first_thirdparty is not None
        and basic_line > first_thirdparty["line"]
    ):
        violations.append(
            {
                "line": basic_line,
                "code": lines[basic_line - 1].strip()
                if 0 < basic_line <= len(lines)
                else "",
                "reason": f"""logging.basicConfig() on line {basic_line} appears after \
third-party import '{first_thirdparty['module']}' on line {first_thirdparty['line']}. \
Third-party libraries may configure logging on import, making basicConfig a no-op.""",
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


# ----------------------------------------------------
# Guardrails: SOLUTION.txt FileHandler in basicConfig
# ----------------------------------------------------


def _is_logging_filehandler_call(call: ast.Call) -> bool:
    """Detect ``logging.FileHandler(...)`` calls."""
    return (
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "logging"
        and call.func.attr == "FileHandler"
    )


def _ast_contains_string(node: ast.AST, target: str) -> bool:
    """Return True if any Constant node within ``node``'s subtree equals ``target``."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and sub.value == target:
            return True
    return False


def check_solution_txt_filehandler(code: str) -> dict:
    """Ensure ``logging.basicConfig`` registers a ``FileHandler`` for SOLUTION.txt.

    Walks the AST to find a top-level ``logging.basicConfig(handlers=[...])``
    call whose handlers list contains ``logging.FileHandler(...)`` whose
    arguments reference the literal string ``"SOLUTION.txt"``. The agent's
    SOLUTION.py is required to self-log via this FileHandler so the curated
    training log lands at ``Path(__file__).parent / "SOLUTION.txt"``.

    Returns a ``{status, violations}`` dict shaped like
    ``check_logging_basicconfig_order``.
    """
    try:
        module = ast.parse(code)
    except SyntaxError as exc:
        return {
            "status": "fail",
            "violations": [
                {
                    "line": getattr(exc, "lineno", 0) or 0,
                    "reason": f"syntax error while parsing generated code: {exc}",
                }
            ],
        }

    basic_calls = [
        node
        for node in module.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and _is_logging_basicconfig_call(node.value)
    ]

    if not basic_calls:
        return {
            "status": "fail",
            "violations": [
                {
                    "line": 0,
                    "reason": "No top-level logging.basicConfig() call found.",
                }
            ],
        }

    for call_expr in basic_calls:
        call = call_expr.value
        handlers_kw = next(
            (kw for kw in call.keywords if kw.arg == "handlers"), None
        )
        if handlers_kw is None:
            return {
                "status": "fail",
                "violations": [
                    {
                        "line": call.lineno,
                        "reason": (
                            "logging.basicConfig() must include a `handlers=[...]` "
                            "kwarg registering a FileHandler for SOLUTION.txt."
                        ),
                    }
                ],
            }
        if not isinstance(handlers_kw.value, ast.List):
            return {
                "status": "fail",
                "violations": [
                    {
                        "line": handlers_kw.value.lineno,
                        "reason": (
                            "logging.basicConfig() `handlers=` must be a list "
                            "literal of handler instances."
                        ),
                    }
                ],
            }
        for handler_node in handlers_kw.value.elts:
            if not isinstance(handler_node, ast.Call):
                continue
            if not _is_logging_filehandler_call(handler_node):
                continue
            if _ast_contains_string(handler_node, "SOLUTION.txt"):
                return {"status": "pass", "violations": []}
        return {
            "status": "fail",
            "violations": [
                {
                    "line": call.lineno,
                    "reason": (
                        "logging.basicConfig() `handlers=` must include a "
                        '`logging.FileHandler(... "SOLUTION.txt" ...)` entry.'
                    ),
                }
            ],
        }

    return {"status": "pass", "violations": []}


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
