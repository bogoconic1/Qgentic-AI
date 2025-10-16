from __future__ import annotations


def leakage_review(task_description: str, code: str) -> str:
    return (
        """
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
  "findings": [
    {"rule_id": "<short_id>", "snippet": "<inline snippet or description>", "rationale": "<why this is risky>", "suggestion": "<how to fix>"}
  ],
  "severity": "block" | "warn" | "none",
}

Be concise and pragmatic; do not include prose outside JSON.
"""
        + "\n\n"
        + "Task Description:\n" + '"""' + f"\n{task_description}\n" + '"""' + "\n\n"
        + "Code:\n" + '"""' + f"\n{code}\n" + '"""'
    )


def debug_sequence_review(code: str) -> str:
    return (
        """
You are reviewing a Python training pipeline for compliance with two runtime rules:
1. The script must execute with DEBUG=True (using a tiny subset/config) before it executes with DEBUG=False (full run). Both executions should happen sequentially in the same process.
2. For deep learning pipelines, if at the end of the 1st epoch of fold 0, the loss or metric is NaN or exactly 0, raise an Exception to stop the run immediately.

Examine the code and determine whether both requirements are satisfied.

Output strictly as JSON in this schema:
{
  "findings": [
    {"rule_id": "debug_sequence" | "nan_guard", "snippet": "<excerpt>", "rationale": "<why non-compliant>", "suggestion": "<how to fix>"}
  ],
  "severity": "block" | "warn" | "none",
}

Set severity="block" if either requirement is missing or incorrect. Use severity="warn" if unsure. Use severity="none" only when both requirements (including the NaN/zero safeguard) are clearly implemented.
Be concise; no extra prose outside JSON.
"""
        + "\n\nCode:\n" + '"""' + f"\n{code}\n" + '"""'
    )


