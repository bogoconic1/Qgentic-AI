import re
import pytest
import sys
from pathlib import Path

# Ensure project root is importable when running tests from repo root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.outdated_code_helpers import analyze_code_string

def assert_has_code(report: str, code: str):
    assert report, "expected a non-empty report"
    assert code in report, f"expected report to contain code {code}\nREPORT:\n{report}"


def test_no_issues_returns_empty_string():
    src = """
def foo():
    x = 1
    y = x + 2
"""
    report = analyze_code_string(src)
    assert report == ""


def test_detect_method_subscript_astype():
    src = """
import pandas as pd
df = pd.DataFrame({"a":[1,2]})
x = df['a'].astype[str]
"""
    report = analyze_code_string(src, filename="example.py")
    assert_has_code(report, "MS001")
    assert "astype" in report


def test_detect_early_stopping_in_fit_call():
    src = """
from xgboost import XGBClassifier
clf = XGBClassifier()
clf.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_val, y_val)])
"""
    report = analyze_code_string(src, filename="fit_example.py")
    assert_has_code(report, "XGB001")
    assert "early_stopping_rounds" in report


def test_detect_bfloat16_usage():
    src = """
import torch
t = torch.tensor([1.0], dtype=torch.bfloat16)
# later conversion
arr = t.detach().cpu().numpy()
"""
    report = analyze_code_string(src, filename="torch_example.py")
    # TORCH001 or TORCH002 may be emitted depending on how bfloat16 appears in AST
    assert ("TORCH001" in report) or ("TORCH002" in report)


def test_detect_transformers_adamw_import():
    src = """
from transformers import AdamW, BertModel
"""
    report = analyze_code_string(src, filename="trf_example.py")
    assert_has_code(report, "TRF001")
    assert "AdamW" in report


def test_detect_deprecated_identifier_ix():
    src = """
df = ...
val = df.ix[0]
"""
    report = analyze_code_string(src, filename="dep_example.py")
    assert_has_code(report, "DEP002")


# Run the tests if invoked directly (optional)
if __name__ == "__main__":
    pytest.main(["-q", __file__])
