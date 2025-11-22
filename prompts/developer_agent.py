from __future__ import annotations

import json
from pathlib import Path

from prompts.shared_constraints import get_hard_constraints


def _limit_list_items(data, max_items=5):
    """Recursively limit list items to max_items for display purposes."""
    if isinstance(data, dict):
        return {k: _limit_list_items(v, max_items) for k, v in data.items()}
    elif isinstance(data, list):
        if len(data) > max_items:
            limited = data[:max_items]
            return limited + [f"... ({len(data) - max_items} more items)"]
        return [_limit_list_items(item, max_items) for item in data]
    else:
        return data


def _read_helper_files(slug: str, iteration: str | int | None = None) -> str:
    """Read cv_splits.json and metric.py from task directory and format for prompt.

    Args:
        slug: Competition slug (e.g., "test-demo")
        iteration: Iteration number (e.g., "17" or "16_2"). If provided, looks in outputs/{base_iteration}/

    Returns:
        Formatted string with helper file schemas and contents, or empty string if no files found
    """
    # Determine base directory
    if iteration is not None:
        # Extract base iteration (e.g., "16" from "16_2")
        iteration_str = str(iteration)
        base_iteration = iteration_str.split('_')[0]
        base_dir = Path(f"task/{slug}/outputs/{base_iteration}")
    else:
        base_dir = Path(f"task/{slug}")

    helper_sections = []

    # Check for cv_splits.json
    cv_splits_path = base_dir / "cv_splits.json"
    if cv_splits_path.exists():
        try:
            with open(cv_splits_path, 'r') as f:
                cv_data = json.load(f)

            # Limit lists to 5 items for display
            cv_data_limited = _limit_list_items(cv_data, max_items=5)

            cv_section = f"""
**cv_splits.json**: pre-defined cross-validation splits. Please read from here and DO NOT generate your own splits.
```json
{json.dumps(cv_data_limited, indent=2)}
```
"""
            helper_sections.append(cv_section)
        except Exception:
            pass  # Skip if file can't be read

    # Check for metric.py
    metric_path = base_dir / "metric.py"
    if metric_path.exists():
        try:
            with open(metric_path, 'r') as f:
                metric_content = f.read()

            metric_section = f"""
**metric.py**: competition-specific evaluation metric. Please use this metric for evaluation. DO NOT generate your own metric.

You should write the line
```
from metric import score
```
to import the metric function.

File contents:
```python
{metric_content}
```
"""
            helper_sections.append(metric_section)
        except Exception:
            pass  # Skip if file can't be read

    if helper_sections:
        return f"\n### 3. Available Helper Files in `{base_dir}`\n" + "\n".join(helper_sections)
    return ""


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


def build_system(description: str, directory_listing: str, model_name: str, slug: str, iteration: str | int | None = None, cpu_core_range: list[int] | None = None, gpu_identifier: str | None = None, gpu_isolation_mode: str = "none", allow_multi_fold: bool = False, hitl_instructions: list[str] | None = None) -> str:
    # Build resource allocation info
    resource_info = ""
    if cpu_core_range is not None:
        resource_info = f"\nNumber of CPUs: {len(cpu_core_range)} cores"

    # Build HITL instructions section if provided
    hitl_section = ""
    if hitl_instructions and len(hitl_instructions) > 0:
        hitl_items = "\n".join([f"{i+1}. {instr}" for i, instr in enumerate(hitl_instructions)])
        hitl_section = f"""
# Human-In-The-Loop Instructions

You have been provided with the following guidance for code implementation:

{hitl_items}

**These instructions should guide your implementation choices, model configuration, and training strategies. Incorporate them while following all hard constraints.**

---
"""

    constraints = _get_hard_constraints(model_name, allow_multi_fold=allow_multi_fold)

    # Read helper files (cv_splits.json, metric.py) if they exist
    helper_files_section = _read_helper_files(slug, iteration)

    return f"""# Role: Lead Developer for Machine-Learning Competition Team
Your objective is to deliver a complete, executable training script (train.py) for a Kaggle Competition using **only** the specified model `{model_name}`.

**Training and Inference Environment:**
- Single GPU (40GB VRAM) {resource_info}

**Model Name:**
`{model_name}`

{hitl_section}{constraints}

**Context:**
- **Competition Description:** 
  {description}
- **Directory Structure for `{Path('task') / slug}`:
  {directory_listing}

{helper_files_section}
---

## Output Requirements

You must output a **single valid Python string** (`train_py`) within ```python backticks that handles the entire pipeline.

Like this:
{{"train_py": "```python\\n<your code>\\n```"}}

### 1. The Code Structure
The `train.py` must follow this logical flow:
1.  **Module Docstring (The Checklist):** Start the file with a detailed docstring containing a 3-7 bullet point checklist of your conceptual approach.
2.  **Imports & Setup:** Import all necessary libraries (including `matplotlib.pyplot` for plotting). Define `HF_TOKEN` from env vars.
3.  **Configuration:** Define all hyperparameters (LR, epochs, batch size) as constants at the top.
4.  **Data Loading:** Load data using the specific paths provided in the context.
5.  **Training Loop:** Implement the training logic using the specified model.
6.  **Artifact Generation:** Ensure **all 5** required artifacts (listed below) are saved.

### 2. Required Artifacts (Files to Save)
The script must save the following files to the paths specified in the user prompt below:

1.  **`submission.csv`**: Test predictions formatted exactly as required by the competition.
2.  **`valid_preds.csv`**: Validation set predictions containing:
    - Fold numbers (if CV)
    - Raw predictions and Ground truth labels
    - Row identifiers (ids)
3.  **`model_*.{{ext}}`**: Saved model files (e.g., `.pt`, `.pkl`, `.h5`). Save per-fold if using CV.
4.  **`train_stats.json`**: A JSON file containing:
    - `model_name`, `cv_scores` (list), `cv_mean`, `cv_std`.
    - `submission_distribution` (stats/counts of the test preds).
    - Key hyperparameters used (e.g. class weights, sequence truncation, image resizing).
    - `errors`: A list of any errors/exceptions encountered during training (e.g., file not found, data loading errors). Capture error messages from try/except blocks.
5.  **Visualizations**:
    - `loss_curve.png`: 1 file with N subplots (use `plt.subplots(1, N)` where N = number of folds). Each subplot shows train/val loss for one fold.
    - `metric_curve.png`: 1 file with N subplots. Each subplot shows train/val metric for one fold.
    - **Important:** Use non-interactive backend (e.g., `plt.switch_backend('Agg')`) or simple save commands. Do not call `plt.show()`.

### 3. Technical Constraints
- **Web Search:** You may perform searches to find the best implementation for `{model_name}`, but the final output must be the code only.
- **Error Handling:** If the exact model name is invalid in a library, comment the alternative chosen.
"""


def build_user(
    base_dir: str | Path,
    outputs_dir: str | Path,
    log_path: str | Path,
    submission_path: str | Path,
    threshold_directive: str = "",
    version: int = 1,
    model_recommendations: str = "",
) -> str:
    # Build model recommendations section (only for version 1)
    recommendations_section = ""
    if version == 1 and model_recommendations:
        recommendations_section = f"""
**Model Recommendations:**
{model_recommendations}

"""

    # Extract version folder from log_path (e.g., outputs/16_2/1 from outputs/16_2/1/train.txt)
    from pathlib import Path
    version_folder = Path(log_path).parent

    base = f"""{recommendations_section}Project structure:
- Base data dir: {base_dir}
- Outputs dir: {outputs_dir}
- Version folder: {version_folder}
- The logs should be written to a file named {log_path}
- Required submission output: {submission_path}
"""
    base += (
        "\nReturn train.py that writes logs to "
        f"{log_path}, "
        "produces a submission CSV at "
        f"{submission_path}, "
        f"and saves the following artifacts to {version_folder}/:\n"
        f"  - valid_preds.csv (validation predictions with fold info, predictions, ground truth, IDs)\n"
        f"  - train_stats.json (model_name, cv_scores, cv_mean, cv_std, submission_distribution, hyperparameters)\n"
        f"  - trained model files (model_*.pkl/.pt/.h5)\n"
        f"  - loss_curve.png (training/validation loss plot)\n"
        f"  - metric_curve.png (training/validation metric plot)"
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


def guardrail_fix_suffix(next_log_path: str | Path, next_submission_path: str | Path, version_folder: str | Path) -> str:
    return (
        "\nPlease regenerate the script addressing the above guardrail issues. "
        f"Write logs to {next_log_path} "
        f"and save the following artifacts to {version_folder}/:\n"
        f"  - submission.csv\n"
        f"  - valid_preds.csv\n"
        f"  - train_stats.json\n"
        f"  - trained model files\n"
        f"  - loss_curve.png\n"
        f"  - metric_curve.png"
    )


def execution_failure_suffix(next_log_path: str | Path, next_submission_path: str | Path, version_folder: str | Path) -> str:
    return (
        "\nPlease modify your code to fix the error!\n\n"
        "Remember:\n"
        f"- write logs to {next_log_path}\n"
        f"- save artifacts to {version_folder}/: submission.csv, valid_preds.csv, train_stats.json, models, loss_curve.png, metric_curve.png"
    )