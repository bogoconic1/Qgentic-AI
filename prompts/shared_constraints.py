from __future__ import annotations


def get_hard_constraints() -> str:
    return """**Hard Constraints:**
- **DO NOT** explicitly set `os.environ['CUDA_VISIBLE_DEVICES']`.
- Place `import logging` and `logging.basicConfig(level=logging.INFO, ...)` at the very top of the script, BEFORE any third-party imports (torch, transformers, kagglehub, etc.). Third-party libraries configure logging on import, so basicConfig must come first.
- Log validation results (per fold and overall), model loading, train/test set size, and any computed quantities that can go wrong (e.g. class weights, thresholds).
- Log final validation results, best epoch number, total training time, and submission prediction distribution.
- Do not use `try/except` to suppress errors.
- External datasets: may be appended **only** to training set.
- If asked to download external datasets, use kagglehub:
```
import kagglehub
path = kagglehub.dataset_download("<author>/<dataset_name>")
```"""
