"""
Shared prompt utilities for the Developer agent.

This module contains common constraint definitions used by developer_agent.py.
"""

from __future__ import annotations


def get_hard_constraints(
    model_name: str | None = None,
    allow_multi_fold: bool = False,
) -> str:
    """
    Get the hard constraints section for system prompts.

    Args:
        model_name: The model name to be used in constraints.
        allow_multi_fold: Whether to allow multi-fold training. If False, adds a
            constraint to only train on fold 0.

    Returns:
        Formatted hard constraints string ready for inclusion in system prompts.
    """
    # Build model-specific constraints
    model_constraint_lines = []
    if model_name:
        model_constraint_lines.append(
            f"- Use ONLY `{model_name}` (no substitutions or fallback models)."
        )

    # Build fold constraint (only for developer with model_name)
    fold_constraint = ""
    if model_name and not allow_multi_fold:
        fold_constraint = "- Just train and validate on fold 0. Skip other folds to save time. If suggested later, you can switch to multi-fold."

    # Build model substitution constraint (developer-specific)
    modular_pipeline_constraint = ""
    if model_name:
        modular_pipeline_constraint = f"- Modular pipeline: update preprocessing/postprocessing or hyperparameters, but do not swap out `{model_name}`."

    # Build common constraints
    common_constraints = """- Deliver a fully-contained, single-file script.
- Use CUDA if available.
- **DO NOT** explicitly set `os.environ['CUDA_VISIBLE_DEVICES']` in your code.
- Place all `logging.info` statements for validation results (per fold and overall) as well as model loading, train/test set size; only log data loading/setup if directly relevant to validation.
- Also emit concise `logging.info` statements for any computed quantities that can go really wrong (e.g. class weights, thresholds).
- Place `logging.basicConfig()` at the start of the script.
- Deep learning: **no** gradient checkpointing. Do not code fallback methods.
- **IMPORTANT:** If you're using XGBoost, LightGBM, or CatBoost, first train the model with the suggested parameters. Then, perform hyperparameter tuning using Optuna for up to 300 seconds. Finally, retrain the model using the best parameters from the tuning run and select the configuration with the best validation performance.
- If you use `transformers.Trainer`, use eval_strategy instead of evaluation_strategy.
- Do not use `try/except` to suppress errors.
- Log final validation results, best epoch number and total training time after training."""

    # Continue with more common constraints
    additional_common_constraints = """- Prefer pretrained models if available. Set pretrained=True if applicable.
- If an online implementation of the model is available (e.g. GitHub), use it. Do not code from scratch.
- External datasets: may be appended **only** to training set.
- **DL Only:** After 1st epoch on fold 0, if loss is NaN, STOP training and jump directly to inference to generate the submission file."""

    # Build the constraints with conditional sections
    constraints_parts = ["**Hard Constraints:**"]

    # Add model-specific constraints first (if applicable)
    if model_constraint_lines:
        constraints_parts.extend(model_constraint_lines)

    # Add common constraints
    constraints_parts.append(common_constraints)
    constraints_parts.append(additional_common_constraints)

    # Add fold constraint (if applicable)
    if fold_constraint:
        constraints_parts.append(fold_constraint)

    # Add modular pipeline constraint (if applicable)
    if modular_pipeline_constraint:
        constraints_parts.append(modular_pipeline_constraint)

    # Add final common constraints
    constraints_parts.append("""- Do not use any `while` loops in your code.""")

    # Add the 's' typo from the original code for backward compatibility
    if model_name:  # Only developer has this typo
        constraints_parts[-1] += "s"

    constraints_parts.append("""- At the end, log the distribution of the submission predictions (e.g., value counts for classification, summary statistics for regression).
- If asked to download external datasets, use kagglehub.
```
import kagglehub

# Download latest version
path = kagglehub.dataset_download("<author>/<dataset_name>")
```""")

    return "\n".join(constraints_parts)
