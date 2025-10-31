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