from __future__ import annotations

from pathlib import Path

from prompts.shared_constraints import get_hard_constraints


def _get_hard_constraints(model_name: str, allow_multi_fold: bool = False) -> str:
    """
    Get the hard constraints section for developer agent system prompts.

    This is a wrapper around the shared get_hard_constraints function that
    maintains backward compatibility with the existing developer agent interface.

    Args:
        model_name: The model name to be used in constraints
        allow_multi_fold: Whether to allow multi-fold training

    Returns:
        Formatted hard constraints string
    """
    return get_hard_constraints(
        model_name=model_name,
        allow_multi_fold=allow_multi_fold,
        include_ensemble_copy_directive=False,
    )


def build_system(description: str, directory_listing: str, model_name: str, model_recommendations: str, slug: str, cpu_core_range: list[int] | None = None, gpu_identifier: str | None = None, gpu_isolation_mode: str = "none", allow_multi_fold: bool = False) -> str:
    # Build resource allocation info
    resource_info = ""
    if cpu_core_range is not None:
        resource_info = f"\nNumber of CPUs: {len(cpu_core_range)} cores"

    constraints = _get_hard_constraints(model_name, allow_multi_fold=allow_multi_fold)

    return f"""# Role: Lead Developer for Machine-Learning Competition Team
Your objective is to deliver a single, self-contained Python script for a Kaggle Competition using **only** the specified model `{model_name}`.

Begin with a concise checklist (3-7 bullets) of what you will do; keep items conceptual, not implementation-level.
You should perform web searches to determine how to set up and configure `{model_name}` in Python. If the model name doesn't exist, find the closest alternative and explain your choice in comments. It's alright if the alternative is larger, but please still use a pretrained version rather than training from scratch.

---
**Training and Inference Environment:**
Single GPU (24GB VRAM) {resource_info}

**Model Name:**
`{model_name}`

**Model Recommendations:**
{model_recommendations}

{constraints}
---

Before any significant tool call or external library use, state the purpose and minimal inputs required, and validate actions after key steps with a 1-2 line summary. If a step fails (e.g., CUDA unavailable), state the limitation clearly and proceed conservatively where allowed.

**Additional Context**
- **Competition Description:**
  {description}
- **Directory Structure for `{Path('task') / slug}`:**
  {directory_listing}

Set reasoning_effort = medium for this task; technical outputs must be complete but concise. Make code and tool calls terse, and expand documentation or schema notes as needed.

## Output Format

Your response MUST follow these sections, in order:

### Checklist: Conceptual Steps
- ...(3-7 high-level conceptual bullet points)

### Code
- Produce a single Python script, enclosed in a triple backtick block with the `python` annotation.
- Model task and metric: infer classification/regression and metric from `{description}`; if unclear, use `accuracy` for classification, `rmse` for regression. Log your chosen metric with justification.
- Document schema/assumptions in comments, as it's inferred from available data.
- For output (predictions/`submission.csv`, saved models), save to the directory defined by `BASE_DIR` (see sample below).
- The HuggingFace API token is available via the `HF_TOKEN` environment variable. Make sure to read it in your code.

**Required Outputs:**
1. **Submission file**: Save test predictions to the submission CSV path specified in the user prompt
2. **Validation predictions**: Save validation predictions to `models_{{version}}/valid_preds.csv` (where {{version}} is specified in user prompt) with detailed information:
   - Fold numbers (if using cross-validation)
   - Predictions for each validation sample
   - Ground truth labels/values (if available)
   - Row identifiers (sample IDs, indices, etc.)
   - Any other relevant metadata
3. **Trained models**: Save all trained models to `models_{{version}}/` directory:
   - Use appropriate extensions: .pkl (sklearn/xgboost/lightgbm/catboost), .pt (PyTorch), .h5 (TensorFlow/Keras)
   - For multi-fold training: save each fold separately (e.g., model_fold0.pkl, model_fold1.pkl)
   - For single model: use descriptive names (e.g., model.pkl, model.pt)

Example Output Block:
```python
# <YOUR CODE>
```
"""


def build_user(
    base_dir: str | Path,
    outputs_dir: str | Path,
    log_path: str | Path,
    submission_path: str | Path,
    threshold_directive: str = "",
    version: int = 1,
) -> str:
    models_dir = f"{outputs_dir}/models_{version}"
    base = f"""
Project structure:
- Base data dir: {base_dir}
- Outputs dir: {outputs_dir}
- The logs should be written to a file named {log_path}
- Required submission output: {submission_path}
- Models and validation predictions directory: {models_dir}
"""
    base += (
        "\nReturn the complete Python script that, when run, writes logs to "
        f"{log_path}, "
        "produces a submission CSV at "
        f"{submission_path}, "
        f"saves validation predictions to {models_dir}/valid_preds.csv, "
        f"and saves trained models to {models_dir}/."
    )

    if threshold_directive:
        base += f"\n{threshold_directive}"
    return base


def patch_mode_directive(base_filename: str) -> str:
    return (
        f"""**IMPORTANT**: Please write a git diff (patch) within ```diff to fix the above issues!
- Produce a unified diff (apply patch format) that updates {base_filename}. Do not include any prefixes in the diff other than {base_filename}.
- Return only the diff enclosed in ```diff fences; do not resend the full script.
- Ensure the diff applies cleanly with the `patch` utility using the file names above.
- Use standard hunk headers with explicit line numbers, e.g. @@ -12,7 +12,9 @@.
- Refer to the <previous_code_with_line_numbers> section above when calculating line numbers.

Like this
```diff
--- {base_filename}
+++ {base_filename}
@@ -1,3 +1,3 @@
 start
-first change
+new first change
 middle
@@ -7,4 +7,4 @@
 some content
-second change
+new second change
 more content
 end
 ```"""
    ).strip()


def guardrail_fix_suffix(next_log_path: str | Path, next_submission_path: str | Path, models_dir: str | Path) -> str:
    return (
        "\nPlease regenerate the script addressing the above guardrail issues. "
        f"Write logs to {next_log_path}, "
        f"produce submission at {next_submission_path}, "
        f"save validation predictions to {models_dir}/valid_preds.csv, "
        f"and save models to {models_dir}/."
    )


def execution_failure_suffix(next_log_path: str | Path, next_submission_path: str | Path, models_dir: str | Path) -> str:
    return (
        "\nPlease modify your code to fix the error!\n\n"
        "Remember:\n"
        f"- write logs to {next_log_path}\n"
        f"- produce the next submission at {next_submission_path}\n"
        f"- save validation predictions to {models_dir}/valid_preds.csv\n"
        f"- save models to {models_dir}/"
    )



