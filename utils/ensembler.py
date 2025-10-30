import os
import json
import shutil
import tempfile
from pathlib import Path
from typing import Callable, Optional
import numpy as np
import pandas as pd
from utils.grade import run_grade


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


def greedy_ensemble(
    submission_files: list[str],
    score_func: Callable[[pd.DataFrame], float],
    model_names: list[str],
    target_col: str = "target",
    id_col: str = "id",
    weight_min: float = -0.6,
    weight_max: float = 0.6,
    weight_step: float = 0.05,
    minimize: bool = True,
    verbose: bool = True
) -> tuple[pd.DataFrame, dict]:
    """
    Blend submissions using greedy forward selection.

    Starts with the best single model, then iteratively adds models that improve the ensemble.

    Args:
        submission_files: List of paths to submission CSV files
        score_func: Function that takes a DataFrame and returns a score
        model_names: List of model names (same order as submission_files)
        target_col: Name of the prediction column (default: "target")
        id_col: Name of the ID column (default: "id")
        weight_min: Minimum weight to try when adding a model (default: -0.6)
        weight_max: Maximum weight to try when adding a model (default: 0.6)
        weight_step: Step size for weight search (default: 0.001)
        minimize: If True, minimize score; if False, maximize (default: True)
        verbose: Print progress information (default: True)

    Returns:
        Tuple of (blended_submission_df, metadata_dict)
    """
    if len(submission_files) < 2:
        raise ValueError("Need at least 2 submission files for ensembling")

    # Load all submissions and sort by ID for consistent ordering
    submissions = []
    for file_path in submission_files:
        df = pd.read_csv(file_path)
        if id_col not in df.columns or target_col not in df.columns:
            raise ValueError(f"Submission {file_path} must contain '{id_col}' and '{target_col}' columns")
        df = df.sort_values(by=id_col).reset_index(drop=True)
        submissions.append(df)

    # Verify all submissions have same IDs
    base_ids = submissions[0][id_col].values
    for i, sub in enumerate(submissions[1:], start=1):
        if not np.array_equal(sub[id_col].values, base_ids):
            raise ValueError(f"Submission {i} has different IDs than submission 0 (even after sorting)")

    # Extract predictions as numpy arrays
    predictions = np.array([sub[target_col].values for sub in submissions])  # shape: (n_models, n_samples)
    n_models = len(submissions)

    # Step 1: Find best single model
    best_model_idx = None
    best_single_score = float('inf') if minimize else float('-inf')

    if verbose:
        print("Finding best single model...")

    for i in range(n_models):
        df = pd.DataFrame({id_col: base_ids, target_col: predictions[i]})
        score = score_func(df)
        if verbose:
            print(f"  {model_names[i]}: {score:.6f}")

        is_better = (score < best_single_score) if minimize else (score > best_single_score)
        if is_better:
            best_single_score = score
            best_model_idx = i

    # Initialize ensemble with best single model
    current_ensemble_preds = predictions[best_model_idx].copy()
    model_weights = {model_names[best_model_idx]: 1.0}
    remaining_indices = list(range(n_models))
    remaining_indices.remove(best_model_idx)

    if verbose:
        print(f"\nInitial best single model: {model_names[best_model_idx]}, Score: {best_single_score:.6f}\n")
        print("Starting greedy forward selection...")

    # Step 2: Greedy forward selection
    weight_range = np.arange(weight_min, weight_max, weight_step)
    iteration = 0

    while remaining_indices:
        iteration += 1
        best_improvement_score = best_single_score
        best_idx, best_weight = None, None

        # Try adding each remaining model
        for idx in remaining_indices:
            for wgt in weight_range:
                # Blend: (1-wgt) * current_ensemble + wgt * candidate_model
                candidate_preds = (1 - wgt) * current_ensemble_preds + wgt * predictions[idx]
                df = pd.DataFrame({id_col: base_ids, target_col: candidate_preds})
                score = score_func(df)

                is_better = (score < best_improvement_score) if minimize else (score > best_improvement_score)
                print(f"    Trying {model_names[idx]} with weight {wgt:.3f}: Score = {score:.6f}")
                if is_better:
                    best_improvement_score = score
                    best_idx = idx
                    best_weight = wgt

        # If improvement found, add the model
        if best_idx is not None:
            # Update ensemble predictions
            current_ensemble_preds = (1 - best_weight) * current_ensemble_preds + best_weight * predictions[best_idx]

            # Update weights: rescale existing weights by (1-best_weight), add new model
            model_weights = {name: weight * (1 - best_weight) for name, weight in model_weights.items()}
            model_weights[model_names[best_idx]] = best_weight

            # Remove added model from remaining
            remaining_indices.remove(best_idx)
            best_single_score = best_improvement_score

            if verbose:
                print(f"Iteration {iteration}: Added {model_names[best_idx]}, Weight: {best_weight:.5f}, Score: {best_improvement_score:.6f}")
        else:
            # No improvement, stop
            if verbose:
                print(f"No improvement found, stopping after {iteration} iterations")
            break

    # Create final blended submission
    final_blend = pd.DataFrame({id_col: base_ids, target_col: current_ensemble_preds})

    # Convert weights dict to list in original order
    weights_list = [model_weights.get(name, 0.0) for name in model_names]

    metadata = {
        "weights": weights_list,
        "model_weights": model_weights,
        "best_score": float(best_single_score),
        "submission_files": submission_files,
        "model_names": model_names,
        "n_iterations": iteration
    }

    if verbose:
        print(f"\nFinal weights:")
        for name, weight in model_weights.items():
            print(f"  {name}: {weight:.5f}")
        print(f"Final score: {best_single_score:.6f}")

    return final_blend, metadata


def main(
    slug: str,
    iteration: int,
    target_col: str = "target",
    id_col: str = "id",
    output_filename: str = "submission_ens.csv"
) -> Path:
    """
    Main workflow: Prepare ensemble folder, blend submissions using greedy forward selection, and save result.

    Automatically uses utils/grade.py to score candidates during optimization.

    Args:
        slug: Competition slug
        iteration: Iteration number
        target_col: Name of prediction column (default: "target")
        id_col: Name of ID column (default: "id")
        output_filename: Name of output file (default: "submission_ens.csv")

    Returns:
        Path to the saved ensemble submission file
    """
    print(f"\n{'='*60}")
    print(f"Ensemble Pipeline for {slug} - Iteration {iteration}")
    print(f"{'='*60}\n")

    # Step 1: Prepare ensemble folder
    print("Step 1: Preparing ensemble folder...")
    ensemble_folder = move_best_code_to_ensemble_folder(slug, iteration)
    print(f"Ensemble folder created at: {ensemble_folder}\n")

    # Step 2: Load ensemble metadata
    metadata_path = ensemble_folder / "ensemble_metadata.json"
    with open(metadata_path, "r") as f:
        ensemble_metadata = json.load(f)

    # Step 3: Collect submission files
    submission_files = []
    model_names = []
    model_scores = []

    for model_name, model_data in ensemble_metadata.items():
        submission_file = model_data.get("submission_file")
        if submission_file:
            submission_path = ensemble_folder / submission_file
            if submission_path.exists():
                submission_files.append(str(submission_path))
                model_names.append(model_name)
                model_scores.append(model_data.get("best_score"))
            else:
                print(f"Warning: Submission file {submission_file} not found, skipping {model_name}")
        else:
            print(f"Warning: No submission file for {model_name}, skipping")

    if len(submission_files) < 2:
        raise ValueError(f"Need at least 2 submissions for ensembling, found {len(submission_files)}")

    print(f"Step 2: Found {len(submission_files)} submissions to blend:")
    for name, score, file in zip(model_names, model_scores, submission_files):
        print(f"  - {name}: score={score} ({Path(file).name})")
    print()

    # Step 3: Create scoring function using utils/grade.py
    print("Step 3: Blending submissions using greedy forward selection with mlebench grading...")

    def grade_submission(submission_df: pd.DataFrame) -> float:
        """Score a submission using mlebench grade-sample."""
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            submission_df.to_csv(tmp.name, index=False)
            tmp_path = tmp.name

        try:
            # Grade using mlebench
            info, stdout, returncode, stderr = run_grade(tmp_path, slug)

            if returncode != 0 or info is None:
                print(f"  Warning: Grading failed (returncode={returncode})")
                return float('-inf')  # Return worst possible score for failed grading

            score = info.get('score')
            if score is None:
                print(f"  Warning: No score in grading output")
                return float('-inf')

            return float(score)
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # Step 4: Run greedy forward selection with automatic grading
    blended_df, blend_metadata = greedy_ensemble(
        submission_files=submission_files,
        score_func=grade_submission,
        model_names=model_names,
        target_col=target_col,
        id_col=id_col,
        minimize=True,  # Minimize score (lower is better)
        verbose=True
    )

    # Step 5: Save blended submission
    output_path = ensemble_folder / output_filename
    blended_df.to_csv(output_path, index=False)

    # Save metadata
    metadata_output_path = ensemble_folder / "ensemble_blend_metadata.json"
    with open(metadata_output_path, "w") as f:
        json.dump(blend_metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✓ Ensemble complete!")
    print(f"  Output: {output_path}")
    print(f"  Metadata: {metadata_output_path}")
    print(f"  Method: Greedy forward selection")
    print(f"  Final score: {blend_metadata.get('best_score', 'N/A')}")
    print(f"  Iterations: {blend_metadata.get('n_iterations', 'N/A')}")
    print(f"{'='*60}\n")

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ensemble multiple model submissions using greedy forward selection with automatic grading")
    parser.add_argument("slug", type=str, help="Competition slug")
    parser.add_argument("iteration", type=int, help="Iteration number")
    parser.add_argument("--target-col", type=str, default="target", help="Target column name (default: target)")
    parser.add_argument("--id-col", type=str, default="id", help="ID column name (default: id)")
    parser.add_argument("--output", type=str, default="submission_ens.csv", help="Output filename (default: submission_ens.csv)")

    args = parser.parse_args()

    main(
        slug=args.slug,
        iteration=args.iteration,
        target_col=args.target_col,
        id_col=args.id_col,
        output_filename=args.output
    )
