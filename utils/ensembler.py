import os
import json
import shutil
from pathlib import Path
from typing import Callable, Optional
import numpy as np
import pandas as pd


def move_best_code_to_ensemble_folder(slug: str, iteration: int):
    """
    Prepare ensemble folder by copying best code files and logs from each baseline model.

    Args:
        slug: Competition slug
        iteration: Iteration number

    Returns:
        Path to the created ensemble folder
    """
    # Define paths
    base_path = Path(f"task/{slug}/outputs/{iteration}")
    baseline_results_path = base_path / "baseline_results.json"
    ensemble_folder = base_path / "ensemble"

    # Check if baseline_results.json exists
    if not baseline_results_path.exists():
        raise FileNotFoundError(f"baseline_results.json not found at {baseline_results_path}")

    # Load baseline results
    with open(baseline_results_path, "r") as f:
        baseline_results = json.load(f)

    # Create ensemble folder if it doesn't exist
    ensemble_folder.mkdir(parents=True, exist_ok=True)

    # Prepare metadata for ensembler
    ensemble_metadata = {}

    # Process each model's results
    for model_name, model_data in baseline_results.items():
        best_code_file = model_data.get("best_code_file")
        best_score = model_data.get("best_score")

        if not best_code_file:
            print(f"Warning: No best_code_file found for model '{model_name}', skipping.")
            continue

        # Extract iteration suffix and version from code filename
        # Format: code_{iteration}_{model_index}_v{version}.py
        # Examples: code_2_1_v3.py, code_2_2_v10.py, code_2_3_v12.py
        code_filename = Path(best_code_file).name
        parts = code_filename.replace("code_", "").replace(".py", "").split("_v")
        iteration_suffix = parts[0]  # e.g., "2_1" from "code_2_1_v3.py"
        version_number = parts[1] if len(parts) > 1 else "1"  # e.g., "3", "10", "12"

        # Define source paths (code files are in sibling directories at outputs level)
        # Structure: outputs/2/baseline_results.json references files in outputs/2_1/, outputs/2_2/, etc.
        code_folder = base_path.parent / iteration_suffix  # outputs/2 -> outputs/2_1
        code_file_path = code_folder / best_code_file
        log_file_path = code_folder / best_code_file.replace(".py", ".txt")

        # Check if code file exists
        if not code_file_path.exists():
            print(f"Warning: Code file {code_file_path} not found, skipping.")
            continue

        # Copy code file to ensemble folder
        dest_code_file = ensemble_folder / best_code_file
        shutil.copy2(code_file_path, dest_code_file)
        print(f"Copied {best_code_file} to ensemble folder")

        # Copy log file if it exists
        if log_file_path.exists():
            dest_log_file = ensemble_folder / log_file_path.name
            shutil.copy2(log_file_path, dest_log_file)
            print(f"Copied {log_file_path.name} to ensemble folder")
        else:
            print(f"Warning: Log file {log_file_path} not found")

        # Find corresponding submission file using the version number
        # e.g., code_2_1_v3.py -> submission_3.csv
        #       code_2_2_v10.py -> submission_10.csv
        submission_file_name = f"submission_{version_number}.csv"
        submission_file_path = code_folder / submission_file_name

        # Copy submission file with renamed filename to indicate model
        # e.g., submission_3.csv -> submission_model_1.csv (based on model index)
        model_index = iteration_suffix.split("_")[-1]  # e.g., "2_1" -> "1"
        dest_submission_name = f"submission_model_{model_index}.csv"
        submission_file_copied = None

        if submission_file_path.exists():
            dest_submission_file = ensemble_folder / dest_submission_name
            shutil.copy2(submission_file_path, dest_submission_file)
            print(f"Copied {submission_file_name} to {dest_submission_name} in ensemble folder")
            submission_file_copied = dest_submission_name
        else:
            print(f"Warning: Submission file {submission_file_path} not found")

        # Add metadata for this model (only essential fields)
        ensemble_metadata[model_name] = {
            "best_code_file": best_code_file,
            "best_score": best_score,
            "submission_file": submission_file_copied,
            "blacklisted_ideas": model_data.get("blacklisted_ideas", []),
            "successful_ideas": model_data.get("successful_ideas", [])
        }

    # Save ensemble metadata to JSON file
    ensemble_metadata_path = ensemble_folder / "ensemble_metadata.json"
    with open(ensemble_metadata_path, "w") as f:
        json.dump(ensemble_metadata, f, indent=2)
    print(f"Saved ensemble metadata to {ensemble_metadata_path}")

    return ensemble_folder


def hill_climb_ensemble(
    submission_files: list[str],
    score_func: Callable[[pd.DataFrame], float],
    target_col: str = "target",
    id_col: str = "id",
    max_iterations: int = 100,
    step_size: float = 0.05,
    minimize: bool = False,
    verbose: bool = True
) -> tuple[pd.DataFrame, dict]:
    """
    Blend multiple submission files using hill climbing to find optimal weights.

    Args:
        submission_files: List of paths to submission CSV files
        score_func: Function that takes a DataFrame (with id and predictions) and returns a score
                   For validation: pass a function that computes metric against ground truth
                   For test: pass a function that evaluates predictions (e.g., consistency check)
        target_col: Name of the prediction column in submission files (default: "target")
        id_col: Name of the ID column in submission files (default: "id")
        max_iterations: Maximum number of hill climbing iterations (default: 100)
        step_size: Step size for weight adjustments (default: 0.05)
        minimize: If True, minimize score; if False, maximize score (default: False)
        verbose: Print progress information (default: True)

    Returns:
        Tuple of (blended_submission_df, metadata_dict)
        - blended_submission_df: DataFrame with id_col and target_col containing weighted predictions
        - metadata_dict: Dictionary with final weights and best score
    """
    if len(submission_files) < 2:
        raise ValueError("Need at least 2 submission files for ensembling")

    # Load all submissions
    submissions = []
    for file_path in submission_files:
        df = pd.read_csv(file_path)
        if id_col not in df.columns or target_col not in df.columns:
            raise ValueError(f"Submission {file_path} must contain '{id_col}' and '{target_col}' columns")
        submissions.append(df)

    # Verify all submissions have same IDs and order
    base_ids = submissions[0][id_col].values
    for i, sub in enumerate(submissions[1:], start=1):
        if not np.array_equal(sub[id_col].values, base_ids):
            raise ValueError(f"Submission {i} has different IDs or order than submission 0")

    # Extract predictions as numpy arrays
    predictions = np.array([sub[target_col].values for sub in submissions])  # shape: (n_models, n_samples)
    n_models = len(submissions)

    # Initialize weights uniformly
    weights = np.ones(n_models) / n_models

    def blend_predictions(w: np.ndarray) -> pd.DataFrame:
        """Create blended submission using given weights."""
        w_normalized = w / w.sum()  # Ensure weights sum to 1
        blended = (predictions * w_normalized[:, np.newaxis]).sum(axis=0)
        return pd.DataFrame({id_col: base_ids, target_col: blended})

    # Evaluate initial score
    best_weights = weights.copy()
    current_blend = blend_predictions(best_weights)
    best_score = score_func(current_blend)

    if verbose:
        print(f"Initial score: {best_score:.6f} (weights: {best_weights})")

    # Hill climbing
    for iteration in range(max_iterations):
        improved = False

        # Try adjusting each weight
        for i in range(n_models):
            # Try increasing weight i
            new_weights = best_weights.copy()
            new_weights[i] += step_size

            candidate_blend = blend_predictions(new_weights)
            candidate_score = score_func(candidate_blend)

            # Check if improvement (depends on minimize flag)
            is_better = (candidate_score < best_score) if minimize else (candidate_score > best_score)

            if is_better:
                best_score = candidate_score
                best_weights = new_weights
                improved = True
                if verbose:
                    print(f"Iteration {iteration+1}: New best score {best_score:.6f} (weights: {best_weights / best_weights.sum()})")
                continue

            # Try decreasing weight i (only if current weight allows it)
            if best_weights[i] > step_size:
                new_weights = best_weights.copy()
                new_weights[i] -= step_size

                candidate_blend = blend_predictions(new_weights)
                candidate_score = score_func(candidate_blend)

                is_better = (candidate_score < best_score) if minimize else (candidate_score > best_score)

                if is_better:
                    best_score = candidate_score
                    best_weights = new_weights
                    improved = True
                    if verbose:
                        print(f"Iteration {iteration+1}: New best score {best_score:.6f} (weights: {best_weights / best_weights.sum()})")

        # Early stopping if no improvement
        if not improved:
            if verbose:
                print(f"Converged after {iteration+1} iterations (no improvement)")
            break

    # Create final blended submission
    final_weights = best_weights / best_weights.sum()
    final_blend = blend_predictions(best_weights)

    metadata = {
        "weights": final_weights.tolist(),
        "raw_weights": best_weights.tolist(),
        "best_score": float(best_score),
        "submission_files": submission_files,
        "n_iterations": iteration + 1,
        "converged": not improved
    }

    if verbose:
        print(f"\nFinal weights: {final_weights}")
        print(f"Final score: {best_score:.6f}")

    return final_blend, metadata
