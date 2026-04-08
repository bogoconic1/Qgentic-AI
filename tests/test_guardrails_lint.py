import pytest

pytest.importorskip("pyflakes")

from guardrails.linting import check_python_lint
from utils.guardrails import evaluate_guardrails, build_block_summary


def test_check_python_lint_passes_for_valid_code():
    result = check_python_lint("x = 1\nprint(x)\n")
    assert result["decision"] == "proceed"


def test_check_python_lint_blocks_undefined_name():
    result = check_python_lint("print(missing_name)\n")
    assert result["decision"] == "block"
    assert any("missing_name" in err for err in result["errors"])


def test_evaluate_guardrails_blocks_on_lint_issue(monkeypatch):
    monkeypatch.setattr(
        "utils.guardrails.check_code_safety",
        lambda _code: {"decision": "proceed", "reason": "ok"},
    )

    report = evaluate_guardrails(
        code_text="print(missing_name)\n",
        enable_logging_guard=False,
        enable_leakage_guard=False,
        enable_code_safety=True,
        enable_lint_guard=True,
    )

    assert report["decision"] == "block"
    assert report["lint_check"]["decision"] == "block"

    summary = build_block_summary(report)
    assert "pre-execution lint check" in summary
