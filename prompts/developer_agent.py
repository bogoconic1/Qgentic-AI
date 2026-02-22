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


def _read_helper_files(slug: str) -> str:
    """Read cv_splits.json and metric.py from task/{slug} directory and format for prompt."""
    base_dir = Path(f"task/{slug}")

    cv_splits_path = base_dir / "cv_splits.json"
    with open(cv_splits_path, "r") as f:
        cv_data = json.load(f)

    cv_data_limited = _limit_list_items(cv_data, max_items=5)

    cv_section = f"""
**cv_splits.json**: pre-defined cross-validation splits. Please read from here and DO NOT generate your own splits.
```json
{json.dumps(cv_data_limited, indent=2)}
```
"""

    metric_path = base_dir / "metric.py"
    with open(metric_path, "r") as f:
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

    return f"\n### Helper Files in `{base_dir}`\n" + "\n".join([cv_section, metric_section])


def build_system(
    description: str,
    directory_listing: str,
    model_name: str,
    slug: str,
    cpu_core_range: list[int] | None = None,
    hitl_instructions: list[str] | None = None,
) -> str:
    resource_info = ""
    if cpu_core_range is not None:
        resource_info = f"\nNumber of CPUs: {len(cpu_core_range)} cores"

    hitl_section = ""
    if hitl_instructions:
        hitl_items = "\n".join(
            [f"{i + 1}. {instr}" for i, instr in enumerate(hitl_instructions)]
        )
        hitl_section = f"""
# Additional Instructions

{hitl_items}

---
"""

    constraints = get_hard_constraints(model_name=model_name)

    helper_files_section = _read_helper_files(slug)

    return f"""You write train.py for Kaggle competitions using **only** the specified model `{model_name}`.

**Environment:** Single GPU (40GB VRAM) {resource_info}

{hitl_section}{constraints}

**Competition Description:**
{description}

**Directory Structure for `{Path("task") / slug}`:**
{directory_listing}

{helper_files_section}
---

## Output

Return a single Python script within ```python backticks.

Start the file with a docstring containing your reasoning and approach.

### Required Artifacts
The script must save these files to the paths specified in the user prompt:

1. **`submission.csv`**: Test predictions formatted per competition requirements.
2. **`valid_preds.csv`**: Validation predictions with fold numbers, raw predictions, ground truth labels, and row IDs.
3. **`model_*.{{ext}}`**: Saved model files. Save per-fold if using CV.
4. **`train_stats.json`**: Must contain `model_name`, `cv_scores`, `cv_mean`, `cv_std`, `cv_worst`, `submission_distribution`, key hyperparameters, and `errors` (list of exceptions encountered).
5. **`loss_curve.png`** and **`metric_curve.png`**: One subplot per fold showing train/val curves. Use `plt.switch_backend('Agg')`. Do not call `plt.show()`.
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
    recommendations_section = ""
    if version == 1 and model_recommendations:
        recommendations_section = f"""
**Model Recommendations:**
{model_recommendations}

"""
    version_folder = Path(log_path).parent

    base = f"""{recommendations_section}Paths:
- Base data dir: {base_dir}
- Outputs dir: {outputs_dir}
- Version folder: {version_folder}
- Log file: {log_path}
- Submission output: {submission_path}

Save all required artifacts to {version_folder}/."""

    if threshold_directive:
        base += f"\n{threshold_directive}"
    return base
