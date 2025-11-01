"""Prompts for ensembler tools."""

from __future__ import annotations


def prompt_ask_ensemble_strategy(
    ensemble_folder: str,
    directory_listing: str,
    description: str,
    ensemble_iteration: int,
    baseline_codes: dict[str, str]
) -> str:
    """
    Build prompt for ask_ensemble_strategy tool.

    Args:
        ensemble_folder: Path to ensemble folder
        directory_listing: Directory listing of ensemble folder
        description: Competition description
        ensemble_iteration: Current iteration number
        baseline_codes: Dict mapping model names to their baseline code content

    Returns:
        System prompt string
    """
    # Format baseline codes with model names
    baseline_codes_str = "\n\n".join([
        f'<baseline_code model="{model_name}">\n{code}\n</baseline_code>'
        for model_name, code in baseline_codes.items()
    ])
    baseline_models_str = ", ".join(baseline_codes.keys())

    return f"""You are an expert data scientist implementing ensemble strategies by rewriting baseline model code.

<competition_description>
{description}
</competition_description>

Available files in ensemble folder:
{directory_listing}

Baseline models to use: {baseline_models_str}

Baseline code for each model:
{baseline_codes_str}

Your task is to write Python code that:
1. Sets up logging to write to 'ensemble_iteration_{ensemble_iteration}.txt'
2. Implements the specified ensemble strategy by rewriting/combining the baseline code above
3. Trains the ensemble model from scratch (DO NOT just load predictions)
4. Saves the result to 'submission_ens_{ensemble_iteration}.csv' with correct format (id + target columns)
5. The tool will automatically grade the submission after execution

Guidelines:
- Working directory is: {ensemble_folder}
- DO NOT read submission_model_*.csv or submission_ens_*.csv files
- You must TRAIN the ensemble model, not combine predictions
- Read the baseline code files to understand the model architectures and data preprocessing
- Rewrite the code to implement the ensemble strategy (e.g., stacking, blending with cross-validation)
- IMPORTANT: Setup logging at the start of your code:
  ```python
  import logging
  logging.basicConfig(
      level=logging.INFO,
      format='%(asctime)s %(levelname)s: %(message)s',
      handlers=[
          logging.FileHandler('ensemble_iteration_{ensemble_iteration}.txt'),
          logging.StreamHandler()
      ]
  )
  logger = logging.getLogger(__name__)
  logger.info("Starting ensemble iteration {ensemble_iteration}")
  ```
- For stacking strategies:
  * Generate out-of-fold (OOF) predictions from base models using proper CV
  * Train a meta-model on the OOF predictions
  * Use proper train/validation splits to avoid leakage
- For blending strategies:
  * Train base models with holdout validation
  * Optimize blend weights on validation set
  * Apply to test set

Output your code in a ```python code block.
"""
