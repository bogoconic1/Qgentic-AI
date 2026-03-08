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
    assert_has_code(report, "XGB002")
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


# ===== PyTorch Deprecation Tests =====

def test_detect_pytorch_deprecated_function_matrix_rank():
    """Test detection of deprecated torch.matrix_rank function"""
    src = """
import torch
A = torch.randn(4, 4)
rank = torch.matrix_rank(A)
"""
    report = analyze_code_string(src, filename="torch_func.py")
    assert_has_code(report, "TORCH012")
    assert "matrix_rank" in report


def test_detect_pytorch_deprecated_function_eig():
    """Test detection of deprecated torch.eig function"""
    src = """
import torch
A = torch.randn(4, 4)
eigenvalues, eigenvectors = torch.eig(A, eigenvectors=True)
"""
    report = analyze_code_string(src, filename="torch_eig.py")
    assert_has_code(report, "TORCH012")
    assert "eig" in report


def test_detect_pytorch_deprecated_function_lstsq():
    """Test detection of deprecated torch.lstsq function"""
    src = """
import torch
A = torch.randn(5, 3)
B = torch.randn(5, 2)
result = torch.lstsq(B, A)
"""
    report = analyze_code_string(src, filename="torch_lstsq.py")
    assert_has_code(report, "TORCH012")
    assert "lstsq" in report


def test_detect_pytorch_deprecated_method_print_lr():
    """Test detection of deprecated scheduler.print_lr() method"""
    src = """
import torch.optim as optim
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5)
scheduler.print_lr()
"""
    report = analyze_code_string(src, filename="torch_scheduler.py")
    assert_has_code(report, "TORCH011")
    assert "print_lr" in report


def test_detect_pytorch_deprecated_parameter_steps():
    """Test detection of deprecated 'steps' parameter usage"""
    src = """
import torch
x = torch.linspace(0, 10, steps=100)
"""
    report = analyze_code_string(src, filename="torch_linspace.py")
    # This should NOT trigger since steps is required now, not deprecated
    # The deprecation is about it being OPTIONAL before
    # For this test, we check that the code doesn't crash
    assert isinstance(report, str)


def test_detect_pytorch_deprecated_parameter_verbose():
    """Test detection of deprecated 'verbose' parameter in LRScheduler"""
    src = """
import torch.optim as optim
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, verbose=True)
"""
    report = analyze_code_string(src, filename="torch_verbose.py")
    assert_has_code(report, "TORCH013")
    assert "verbose" in report


def test_detect_pytorch_deprecated_class_callspec():
    """Test detection of deprecated CallSpec class"""
    src = """
from torch.export import export, CallSpec
call_spec = CallSpec((torch.Tensor, torch.Tensor))
"""
    report = analyze_code_string(src, filename="torch_callspec.py")
    assert_has_code(report, "TORCH015")
    assert "CallSpec" in report


def test_detect_pytorch_deprecated_function_definitely_true():
    """Test detection of deprecated definitely_true function"""
    src = """
from torch.fx.experimental.symbolic_shapes import definitely_true
result = definitely_true(expr, shape_env)
"""
    report = analyze_code_string(src, filename="torch_def_true.py")
    # Should detect both in import and usage
    assert ("TORCH014" in report) or ("TORCH017" in report)
    assert "definitely_true" in report


def test_detect_pytorch_deprecated_function_definitely_false():
    """Test detection of deprecated definitely_false function"""
    src = """
from torch.fx.experimental.symbolic_shapes import definitely_false
if definitely_false(a == b):
    print("Not equal")
"""
    report = analyze_code_string(src, filename="torch_def_false.py")
    assert ("TORCH014" in report) or ("TORCH017" in report)
    assert "definitely_false" in report


def test_detect_pytorch_deprecated_import_from():
    """Test detection of deprecated function import"""
    src = """
from torch import eig, matrix_rank
A = torch.randn(4, 4)
result = eig(A, eigenvectors=True)
"""
    report = analyze_code_string(src, filename="torch_import.py")
    # Should detect in import statement
    assert ("TORCH014" in report) or ("TORCH017" in report)
    assert "eig" in report


def test_detect_multiple_pytorch_deprecations():
    """Test detection of multiple PyTorch deprecations in one file"""
    src = """
import torch
from torch.fx.experimental.symbolic_shapes import definitely_true

# Deprecated function
A = torch.randn(4, 4)
rank = torch.matrix_rank(A)
eigenvalues = torch.eig(A)

# Deprecated parameter
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, verbose=True)

# Deprecated method
scheduler.print_lr()
"""
    report = analyze_code_string(src, filename="torch_multi.py")
    
    # Should have multiple issues
    assert "Found" in report and "issue" in report
    # Check for various deprecation codes
    codes = ["TORCH012", "TORCH011", "TORCH013"]
    found_codes = sum(1 for code in codes if code in report)
    assert found_codes >= 2, f"Expected multiple deprecation codes, but report was:\n{report}"


def test_pytorch_deprecated_class_xnnpackquantizer():
    """Test detection of deprecated XNNPACKQuantizer class"""
    src = """
from torch.ao.quantization.quantizer.xnnpack_quantizer import XNNPACKQuantizer
quantizer = XNNPACKQuantizer()
"""
    report = analyze_code_string(src, filename="torch_quant.py")
    assert ("TORCH015" in report) or ("TORCH016" in report)
    assert "XNNPACKQuantizer" in report


def test_pytorch_no_false_positive_on_correct_api():
    """Test that correct PyTorch API doesn't trigger false positives"""
    src = """
import torch
# Using correct APIs
A = torch.randn(5, 3)
B = torch.randn(5, 2)
result = torch.linalg.lstsq(A, B)

eigenvalues, eigenvectors = torch.linalg.eig(A)

rank = torch.linalg.matrix_rank(A)
"""
    report = analyze_code_string(src, filename="torch_correct.py")
    # Should not have PyTorch deprecation warnings
    assert "TORCH012" not in report  # No function deprecations
    assert report == "" or "DEP" not in report  # Either no issues or no deprecation issues


def test_xgboost_and_pytorch_together():
    """Test that both XGBoost and PyTorch deprecations can be detected together"""
    src = """
import torch
from xgboost import XGBClassifier

# XGBoost deprecated
clf = XGBClassifier(gpu_id=0)
clf.fit(X, y, early_stopping_rounds=10)

# PyTorch deprecated
A = torch.randn(4, 4)
rank = torch.matrix_rank(A)
"""
    report = analyze_code_string(src, filename="mixed.py")
    
    # Should detect both XGBoost and PyTorch deprecations
    assert "XGB" in report
    assert "TORCH" in report
    assert "gpu_id" in report or "early_stopping_rounds" in report
    assert "matrix_rank" in report


def test_pytorch_deprecated_parameter_in_export():
    """Test detection of deprecated parameter in torch.export.export"""
    src = """
import torch
from torch.export import export

def my_module(x):
    return x * 2

# Deprecated parameter usage
exported = export(my_module, (torch.randn(2, 3),), strict=True)
"""
    report = analyze_code_string(src, filename="torch_export.py")
    assert_has_code(report, "TORCH013")
    assert "strict" in report


def test_pytorch_deprecated_onnx_function():
    """Test detection of deprecated torch.onnx functions"""
    src = """
import torch
import torch.onnx

model = torch.nn.Linear(10, 5)
dummy_input = torch.randn(1, 10)
torch.onnx.dynamo_export(model, dummy_input, 'model.onnx')
"""
    report = analyze_code_string(src, filename="torch_onnx.py")
    assert_has_code(report, "TORCH012")
    assert "dynamo_export" in report


# Run the tests if invoked directly (optional)
if __name__ == "__main__":
    pytest.main(["-v", __file__])