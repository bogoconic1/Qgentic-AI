import ast
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# import weave

BANNED_SUBSCRIPT_METHODS = {"astype"}
BANNED_IDENTIFIERS = {"Panel", "ix", "rolling_apply", "cross_validation", "GridSearchCV_old"}
SUSPICIOUS_TO_CSV_VAR_NAMES = {"submission", "sub", "submission_df"}


class Issue:
    def __init__(self, filename: str, lineno: int, col: int, code: str, message: str):
        self.filename = filename
        self.lineno = lineno
        self.col = col
        self.code = code
        self.message = message

    def to_string(self, snippet: Optional[str] = None) -> str:
        base = f"{self.filename}:{self.lineno}:{self.col} [{self.code}] {self.message}"
        if snippet:
            return base + "\n" + snippet + "\n"
        return base


def _file_snippet_from_source(source: str, lineno: int, context: int = 2) -> str:
    lines = source.splitlines()
    start = max(0, lineno - 1 - context)
    end = min(len(lines), lineno - 1 + context + 1)
    return "\n".join(f"{i+1:4}: {lines[i]}" for i in range(start, end))


class _Checker(ast.NodeVisitor):
    def __init__(self, filename: str, source: str):
        self.filename = filename
        self.source = source
        self.issues: List[Issue] = []
        self.assignments: Dict[str, Tuple[int, str]] = {}

    def visit_Assign(self, node: ast.Assign):
        try:
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if isinstance(node.value, ast.Attribute):
                    attr = node.value.attr
                    self.assignments[name] = (node.lineno, f"attr:{attr}")
                elif isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
                    attr = node.value.func.attr
                    self.assignments[name] = (node.lineno, f"call:{attr}")
        except Exception:
            pass
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript):
        try:
            if isinstance(node.value, ast.Attribute):
                attr = node.value.attr
                if attr in BANNED_SUBSCRIPT_METHODS:
                    msg = (f"Method '{attr}' is being subscripted (e.g. x.{attr}[...]). "
                           f"Likely you meant to *call* the method: x.{attr}(...).")
                    self.issues.append(Issue(self.filename, node.lineno, getattr(node, "col_offset", 0), "MS001", msg))
        except Exception:
            pass
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        try:
            if isinstance(node.func, ast.Attribute) and getattr(node.func, "attr", None) == "fit":
                for kw in node.keywords:
                    if kw.arg == "early_stopping_rounds":
                        msg = ("Call to .fit(...) with keyword 'early_stopping_rounds' detected. "
                               "Many xgboost/sklearn versions require callback-based early stopping or different APIs.")
                        self.issues.append(Issue(self.filename, node.lineno, getattr(node, "col_offset", 0), "XGB001", msg))
            if isinstance(node.func, ast.Attribute) and getattr(node.func, "attr", None) == "to_csv":
                val = node.func.value
                if isinstance(val, ast.Name) and val.id in SUSPICIOUS_TO_CSV_VAR_NAMES:
                    assign = self.assignments.get(val.id)
                    if assign and assign[1].startswith("attr:") and assign[1].split(":", 1)[1] in ("columns", "index"):
                        msg = (f"Variable '{val.id}' was assigned from .{assign[1].split(':',1)[1]} earlier and is now used with .to_csv(). "
                               "This may be an Index/Series being written; convert to DataFrame first e.g. pd.DataFrame({val.id}).to_csv(...) or reset_index().")
                        self.issues.append(Issue(self.filename, node.lineno, getattr(node, "col_offset", 0), "PANDAS001", msg))
        except Exception:
            pass
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        try:
            if node.module == "transformers":
                for alias in node.names:
                    if alias.name == "AdamW":
                        msg = ("Importing AdamW from 'transformers' detected. Many recent codebases use 'torch.optim.AdamW' instead. "
                               "Consider: 'from torch.optim import AdamW' or pin your transformers version.")
                        self.issues.append(Issue(self.filename, node.lineno, getattr(node, "col_offset", 0), "TRF001", msg))
        except Exception:
            pass
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        try:
            if node.id in BANNED_IDENTIFIERS:
                msg = f"Use of deprecated identifier/name '{node.id}' detected. Update to current APIs."
                self.issues.append(Issue(self.filename, node.lineno, getattr(node, "col_offset", 0), "DEP001", msg))
            if node.id == "bfloat16":
                msg = ("Usage of 'bfloat16' detected. Converting tensors with bfloat16 to numpy() can raise errors on CPU. "
                       "Cast to float32 before converting to numpy: tensor.to(dtype=torch.float32).detach().cpu().numpy()")
                self.issues.append(Issue(self.filename, node.lineno, getattr(node, "col_offset", 0), "TORCH001", msg))
        except Exception:
            pass
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        try:
            if node.attr in BANNED_IDENTIFIERS:
                msg = f"Use of deprecated attribute '{node.attr}' detected. Update to current APIs."
                self.issues.append(Issue(self.filename, node.lineno, getattr(node, "col_offset", 0), "DEP002", msg))
            if node.attr == "bfloat16":
                msg = ("Reference to attribute 'bfloat16' found. This may create tensors/dtypes that cause 'unsupported ScalarType BFloat16' when converting to numpy. "
                       "If using mixed precision, be careful to cast before sending to CPU/numpy.")
                self.issues.append(Issue(self.filename, node.lineno, getattr(node, "col_offset", 0), "TORCH002", msg))
        except Exception:
            pass
        self.generic_visit(node)

# @weave.op()
def analyze_code_string(source: str, filename: str = "<string>") -> str:
    """
    Analyze a Python source string and return a formatted report string.

    :param source: Python source code to analyze
    :param filename: optional filename used in report lines
    :return: report string (empty-string if no issues)
    """
    try:
        tree = ast.parse(source, filename=filename)
    except Exception as e:
        return f"ERR_PARSE: Could not parse source for {filename}: {e}"

    checker = _Checker(filename, source)
    checker.visit(tree)
    issues = checker.issues

    if not issues:
        return ""  # No issues found

    lines: List[str] = []
    lines.append(f"Found {len(issues)} issue(s) in {filename}:\n")
    counts: Dict[str, int] = {}
    for it in issues:
        snippet = _file_snippet_from_source(source, it.lineno)
        lines.append(it.to_string(snippet))
        counts[it.code] = counts.get(it.code, 0) + 1

    lines.append("Summary:")
    for code, cnt in sorted(counts.items()):
        lines.append(f"  {code}: {cnt}")

    return "\n".join(lines)