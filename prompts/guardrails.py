from __future__ import annotations


def leakage_review() -> str:
    return """You are a senior machine learning engineer tasked with auditing a Python training script for data leakage.

Begin with a concise checklist (3-7 bullets) of what you will do; keep items conceptual, not implementation-level.

**Your objectives:**
- Detect any train/test contamination risks, such as:
  - Applying fit operations or transformations (e.g., scalers, encoders, PCA, imputers) on data combined from train and test sets.
  - Using any test labels or generating features derived from test labels in any part of the pipeline.
  - Performing feature selection or target encoding outside of cross-validation (CV) or out-of-fold (OOF) contexts.
  - Introducing data leaks via merges or aggregations that incorporate information from the test set or from the future.
  - Employing an incorrect data splitting strategy (e.g., random splits used with time series, which is inappropriate).
  - Using KFold instead of StratifiedKFold for classification tasks when labels are imbalanced.
- For each issue found, point to the relevant code snippet or describe the problematic portion. Provide a succinct rationale for why it is risky, along with a suggested fix.

After analyzing the script, validate your findings in 1-2 lines: confirm each detection meets your objectives and that suggested mitigations are appropriate.

**Return your output strictly as JSON in the following schema:**
```json
{
  "findings": [
    {
      "rule_id": "<short_id>",
      "snippet": "<inline code snippet or clear description>",
      "rationale": "<concise explanation of the risk>",
      "suggestion": "<practical fix or mitigation>"
    }
  ],
  "severity": "block" | "warn" | "none"
}
```

Keep your output concise and practical. Do not include any prose or explanation outside of the JSON response.
"""


def debug_sequence_review() -> str:
    return """Your task is to review a Python training pipeline and verify compliance with two runtime rules:

1. The script must first execute with `DEBUG=True` (using a minimal subset or configuration), followed by `DEBUG=False` (full run), both in the same process and in sequential order.
2. For deep learning pipelines, if at the end of the first epoch of fold 0 the loss is NaN, an Exception must be raised immediately to halt execution.

Begin with a concise checklist (3-7 bullets) of what you will do; keep items conceptual, not implementation-level.

Review the code and check for both requirements, reasoning carefully about each rule.

If editing or suggesting code: (1) state any critical assumptions, (2) propose or check for minimal, testable validation logic, (3) ensure suggestions follow idiomatic Python/style norms.

Your output must be strictly formatted as JSON using this schema:
```
{
  "findings": [
    {
      "rule_id": "debug_sequence" | "nan_guard",
      "snippet": "<excerpt>",
      "rationale": "<why non-compliant>",
      "suggestion": "<how to fix>"
    }
  ],
  "severity": "block" | "warn" | "none"
}
```

Include the output format exactly as shown. Output only the required JSON—no extra commentary. Set `severity` to `block` if either requirement is missing or incorrect, `warn` if you are unsure, and `none` only if both requirements (including strict NaN/zero loss/metric safeguards) are clearly implemented. If you are unable to determine compliance for any rule from the provided code, set the corresponding finding with `warn` severity and specify why.
"""


