import ast
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# import weave

BANNED_SUBSCRIPT_METHODS = {"astype"}
BANNED_IDENTIFIERS = {"Panel", "ix", "rolling_apply", "cross_validation", "GridSearchCV_old"}
SUSPICIOUS_TO_CSV_VAR_NAMES = {"submission", "sub", "submission_df"}

# XGBoost deprecated APIs - loaded from dataset
XGBOOST_DEPRECATED_PARAMS = set()
XGBOOST_DEPRECATED_METHODS = set()
XGBOOST_DEPRECATED_CLASSES = set()
XGBOOST_DEPRECATION_INFO = {}  # Map of API name -> full deprecation info


def load_xgboost_deprecations():
    """
    Load XGBoost deprecation dataset and populate the banned sets.
    This is called automatically when the module is imported.
    """
    global XGBOOST_DEPRECATED_PARAMS, XGBOOST_DEPRECATED_METHODS, XGBOOST_DEPRECATED_CLASSES, XGBOOST_DEPRECATION_INFO
    
    # Try to find the dataset file
    current_dir = Path(__file__).parent
    dataset_path = current_dir / "data" / "xgboost_deprecations" / "xgboost_deprecations_dataset.json"
    
    if not dataset_path.exists():
        # Try relative to project root
        dataset_path = current_dir.parent / "tools" / "data" / "xgboost_deprecations" / "xgboost_deprecations_dataset.json"
    
    if not dataset_path.exists():
        # Dataset not found, silently continue without XGBoost deprecations
        return
    
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            deprecations = json.load(f)
        
        for dep in deprecations:
            api = dep.get('deprecated_api', '').strip()
            category = dep.get('category', '')
            
            if not api:
                continue
            
            # Store full info for detailed messages
            key = f"{api}|{category}"
            XGBOOST_DEPRECATION_INFO[key] = dep
            
            # Add to appropriate set based on category
            if category == 'parameter':
                XGBOOST_DEPRECATED_PARAMS.add(api)
            elif category == 'method':
                XGBOOST_DEPRECATED_METHODS.add(api)
            elif category == 'class':
                XGBOOST_DEPRECATED_CLASSES.add(api)
        
        # Add to main banned identifiers
        BANNED_IDENTIFIERS.update(XGBOOST_DEPRECATED_METHODS)
        BANNED_IDENTIFIERS.update(XGBOOST_DEPRECATED_CLASSES)
        
    except Exception as e:
        # Silently fail if dataset can't be loaded
        pass


# Load XGBoost deprecations when module is imported
load_xgboost_deprecations()


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


def _get_xgboost_deprecation_message(api: str, category: str, context: str = "") -> str:
    """
    Get a detailed deprecation message for an XGBoost API.
    
    :param api: The deprecated API name
    :param category: The category (parameter, method, class)
    :param context: Optional context about where it was found
    :return: Formatted deprecation message
    """
    key = f"{api}|{category}"
    info = XGBOOST_DEPRECATION_INFO.get(key)
    
    if info:
        version = info.get('version', 'unknown')
        replacement = info.get('replacement', 'See XGBoost documentation')
        reason = info.get('reason', '')
        dep_context = info.get('context', '')
        
        msg = f"XGBoost deprecated {category} '{api}' detected (deprecated in v{version}"
        if dep_context:
            msg += f", {dep_context}"
        msg += ")."
        
        if reason:
            msg += f" Reason: {reason}."
        
        msg += f" Replacement: {replacement}"
        
        return msg
    else:
        return f"XGBoost deprecated {category} '{api}' detected. Update to current XGBoost API."


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
            # Check for deprecated XGBoost parameters in fit() method
            if isinstance(node.func, ast.Attribute) and getattr(node.func, "attr", None) == "fit":
                for kw in node.keywords:
                    if kw.arg in XGBOOST_DEPRECATED_PARAMS:
                        msg = _get_xgboost_deprecation_message(kw.arg, "parameter", "in fit() method")
                        self.issues.append(Issue(self.filename, node.lineno, getattr(node, "col_offset", 0), "XGB002", msg))
                    
            
            # Check for deprecated parameters in any XGBoost constructor
            if isinstance(node.func, ast.Name):
                # Check if it's an XGBoost class (simple heuristic: starts with XGB)
                if node.func.id.startswith('XGB') or node.func.id in XGBOOST_DEPRECATED_CLASSES:
                    for kw in node.keywords:
                        if kw.arg in XGBOOST_DEPRECATED_PARAMS:
                            msg = _get_xgboost_deprecation_message(kw.arg, "parameter", f"in {node.func.id}() constructor")
                            self.issues.append(Issue(self.filename, node.lineno, getattr(node, "col_offset", 0), "XGB003", msg))
            
            # Check for deprecated XGBoost methods
            if isinstance(node.func, ast.Attribute):
                if node.func.attr in XGBOOST_DEPRECATED_METHODS:
                    msg = _get_xgboost_deprecation_message(node.func.attr, "method")
                    self.issues.append(Issue(self.filename, node.lineno, getattr(node, "col_offset", 0), "XGB004", msg))
            
            # Existing checks
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
            # Check for deprecated XGBoost imports
            if node.module == "xgboost":
                for alias in node.names:
                    if alias.name == "dask":
                        msg = _get_xgboost_deprecation_message("dask", "import", "from default xgboost import")
                        self.issues.append(Issue(self.filename, node.lineno, getattr(node, "col_offset", 0), "XGB005", msg))
            
            # Existing checks
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
            # Check for deprecated XGBoost classes
            if node.id in XGBOOST_DEPRECATED_CLASSES:
                msg = _get_xgboost_deprecation_message(node.id, "class")
                self.issues.append(Issue(self.filename, node.lineno, getattr(node, "col_offset", 0), "XGB006", msg))
            
            # Existing checks
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
            # Check for deprecated XGBoost methods/attributes
            if node.attr in XGBOOST_DEPRECATED_METHODS:
                msg = _get_xgboost_deprecation_message(node.attr, "method")
                self.issues.append(Issue(self.filename, node.lineno, getattr(node, "col_offset", 0), "XGB007", msg))
            
            # Existing checks
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