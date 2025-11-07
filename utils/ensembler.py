import os
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Callable, Optional
from datetime import datetime
import re
import numpy as np
import pandas as pd
from utils.grade import run_grade
from tools.helpers import call_llm_with_retry
from project_config import get_config
from schemas.ensembler import EnsembleStrategies

logger = logging.getLogger(__name__)

_CONFIG = get_config()
_LLM_CFG = _CONFIG.get("llm")
_PATH_CFG = _CONFIG.get("paths")
_TASK_ROOT = Path(_PATH_CFG.get("task_root"))
_OUTPUTS_DIRNAME = _PATH_CFG.get("outputs_dirname")
_ENSEMBLER_MODEL = _LLM_CFG.get("ensembler_model")


def _parse_training_time_from_log(log_file_path: Path) -> str:
    """
    Parse training time from log file by computing time difference between first and last lines.

    Args:
        log_file_path: Path to the log file

    Returns:
        Training time as a formatted string (e.g., "2h 15m 30s") or "N/A" if parsing fails
    """
    try:
        with open(log_file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        if len(lines) < 2:
            return "N/A"

        # Common timestamp formats to try
        timestamp_patterns = [
            r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?)',  # 2025-01-15 10:30:45 or 2025-01-15 10:30:45.123
            r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?)',  # 2025-01-15T10:30:45
            r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?)\]',  # [2025-01-15 10:30:45]
            r'(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})',  # 01/15/2025 10:30:45
        ]

        datetime_formats = [
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%m/%d/%Y %H:%M:%S",
        ]

        def extract_timestamp(line: str) -> Optional[datetime]:
            """Try to extract timestamp from a line using various patterns."""
            for pattern in timestamp_patterns:
                match = re.search(pattern, line)
                if match:
                    timestamp_str = match.group(1)
                    # Try parsing with different formats
                    for fmt in datetime_formats:
                        try:
                            return datetime.strptime(timestamp_str, fmt)
                        except ValueError:
                            continue
            return None

        # Extract first timestamp
        first_timestamp = None
        for line in lines[:50]:  # Check first 50 lines
            first_timestamp = extract_timestamp(line)
            if first_timestamp:
                break

        # Extract last timestamp
        last_timestamp = None
        for line in reversed(lines[-50:]):  # Check last 50 lines
            last_timestamp = extract_timestamp(line)
            if last_timestamp:
                break

        if not first_timestamp or not last_timestamp:
            return "N/A"

        # Calculate time difference
        time_diff = last_timestamp - first_timestamp
        total_seconds = int(time_diff.total_seconds())

        if total_seconds < 0:
            return "N/A"

        # Format as human-readable string
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        parts = []
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if seconds > 0 or not parts:  # Always show seconds if no other parts
            parts.append(f"{seconds}s")

        return " ".join(parts)

    except Exception as e:
        logger.debug("Failed to parse training time from %s: %s", log_file_path, str(e))
        return "N/A"


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

        # Copy JSON metadata file (contains num_header_lines for strip_header_from_code)
        json_file_path = code_file_path.with_suffix('.json')
        if json_file_path.exists():
            dest_json_file = ensemble_folder / json_file_path.name
            shutil.copy2(json_file_path, dest_json_file)
            print(f"Copied {json_file_path.name} to ensemble folder")
        else:
            print(f"Warning: JSON metadata file {json_file_path} not found")

        # Copy log file if it exists and parse training time
        training_time = "N/A"
        if log_file_path.exists():
            dest_log_file = ensemble_folder / log_file_path.name
            shutil.copy2(log_file_path, dest_log_file)
            print(f"Copied {log_file_path.name} to ensemble folder")

            # Parse training time from log
            training_time = _parse_training_time_from_log(log_file_path)
            print(f"Parsed training time: {training_time}")
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
            "training_time": training_time,
            "blacklisted_ideas": model_data.get("blacklisted_ideas", []),
            "successful_ideas": model_data.get("successful_ideas", [])
        }

    # Save ensemble metadata to JSON file
    ensemble_metadata_path = ensemble_folder / "ensemble_metadata.json"
    with open(ensemble_metadata_path, "w") as f:
        json.dump(ensemble_metadata, f, indent=2)
    print(f"Saved ensemble metadata to {ensemble_metadata_path}")

    return ensemble_folder


def recommend_ensemble_strategies(slug: str, iteration: int) -> list[dict]:
    """
    Generate 8 diverse ensemble strategy recommendations using LLM with web search.

    Similar to ModelRecommenderAgent, this function:
    1. Reads ensemble_metadata.json and competition description
    2. Uses LLM with web search to find SOTA ensemble/tuning strategies
    3. Generates 8 diverse strategies (stacking, blending, hyperparameter tuning, pre-training, etc.)
    4. Saves strategies to ensemble_metadata.json under "strategies" field

    Args:
        slug: Competition slug
        iteration: Iteration number

    Returns:
        List of 8 strategy dicts with format: {"strategy": "...", "models_needed": [...]}
    """
    logger.info("Generating ensemble strategy recommendations for slug=%s iteration=%s", slug, iteration)

    # Define paths
    base_path = _TASK_ROOT / slug
    outputs_dir = base_path / _OUTPUTS_DIRNAME / str(iteration)
    ensemble_folder = outputs_dir / "ensemble"
    metadata_path = ensemble_folder / "ensemble_metadata.json"
    description_path = base_path / "description.md"

    # Check if ensemble metadata exists
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"ensemble_metadata.json not found at {metadata_path}. "
            "Please run move_best_code_to_ensemble_folder() first."
        )

    # Load ensemble metadata
    with open(metadata_path, "r") as f:
        ensemble_metadata = json.load(f)

    # Load competition description
    if description_path.exists():
        with open(description_path, "r") as f:
            description = f.read()
    else:
        logger.warning("No description.md found at %s", description_path)
        description = ""

    # Build prompt for strategy recommendation
    metadata_summary = []
    all_blacklisted_ideas = []

    for model_name, model_data in ensemble_metadata.items():
        # Skip the "strategies" field if it exists
        if model_name == "strategies" or not isinstance(model_data, dict):
            continue

        score = model_data.get("best_score", "N/A")
        training_time = model_data.get("training_time", "N/A")
        blacklisted = model_data.get("blacklisted_ideas", [])

        metadata_summary.append(f"- {model_name}: score={score}, training_time={training_time}")

        # Collect all blacklisted ideas
        for idea in blacklisted:
            all_blacklisted_ideas.append(f"  [{model_name}] {idea}")

    metadata_summary_str = "\n".join(metadata_summary)

    # Format blacklisted ideas
    if all_blacklisted_ideas:
        blacklisted_ideas_str = "\n".join(all_blacklisted_ideas)
    else:
        blacklisted_ideas_str = "None"

    system_prompt = f"""You are an expert Kaggle competitor specializing in ensemble methods and model optimization.

# Role and Objective
Develop and recommend 8 diverse, independent, and actionable ensemble strategies to outperform current baseline models in a Kaggle competition setting.

# Instructions
- Begin with a concise conceptual checklist (3-7 bullets) outlining your overall approach (not implementation specifics).
- For each strategy, provide:
  - A detailed description
  - Specific models used (as strings)
  - The ensemble methodology (as a named string)
  - Implementation guidance
- Draw inspiration from these ensemble topics:
  1. **Hyperparameter tuning**
  2. **Stacking**
  3. **Blending**
  4. **Multi-Stage training**
  5. **Pseudo Labeling**
  6. **Advanced ensembling techniques**
- Ensure recommendations:
  - Are specific and actionable (no vague or generic suggestions)
  - Are feasible to implement
  - Span multiple ensemble categories above

- Present strategies as a numbered list (1-8), following the format guidelines below.
- Before any web search or recommendation of contemporary techniques (2024-2025), explicitly state your intention and the minimal necessary inputs for the action.
- Do not search for or reference winning solutions for this particular competition.
- Avoid suggesting strategies similar to those in the provided blacklist; if an item is unfillable, state: "No suitable strategy could be found for this item."
- After listing, validate that all strategies are distinct, actionable, independent, and do not overlap the blacklist. If duplicates or overlaps are found, substitute accordingly to maximize diversity.
- After completing the strategies, provide a brief validation summary indicating success or the need for substitutions.
- Set reasoning_effort = medium and keep outputs neither terse nor overly verbose.

# Context
- <competition_description>
  {description}
  </competition_description>
- <baseline_models> (model names [string] and training times [string, e.g., '2h'])
  {metadata_summary_str}
  </baseline_models>
- <blacklisted_ideas>
  The following did NOT improve performance. Avoid similar approaches:
  {blacklisted_ideas_str}
  </blacklisted_ideas>

# Output Format
- List exactly 8 strategies as a numbered list, detailed and content-rich.
- Remember to make tool calls only as allowed, and state the intent before any significant tool invocation, describing both the rationale and minimal inputs.
- Models listed under models_needed must correspond to actual model names already defined in the <baseline_models> section.
- If you want to propose new models that are not yet defined, explicitly mark them as new (e.g., NEW: <model_name>).
- Provide a `strategies` field containing an array of exactly 8 strategy objects, where each object has:
  - `strategy`: A richly detailed string describing the ensemble strategy
  - `models_needed`: A list of model names (strings) required for this strategy

## Example Strategies
1. Use a LightGBM, CatBoost, and XGBoost three-way weighted average (models: 'LightGBM', 'CatBoost', 'XGBoost') after tuning each individually by Optuna; weights determined by holdout set RMSE.
2. Implement two-stage stacking: first-level learners are fine-tuned CNN and transformer models ('ResNet50', 'EfficientNet', 'ConvNext'); then train a CatBoost with the outputs of the first stage as features.
...
8. [Eighth strategy]
"""

    user_prompt = "Generate 8 diverse ensemble strategies for this competition."

    # Call LLM with web search enabled and structured outputs
    response = call_llm_with_retry(
        model=_ENSEMBLER_MODEL,
        instructions=system_prompt,
        tools=[],
        messages=[{"role": "user", "content": user_prompt}],
        web_search_enabled=True,
        text_format=EnsembleStrategies,
    )

    # Use structured output
    strategies = []
    if response and hasattr(response, 'output_parsed') and response.output_parsed:
        parsed = response.output_parsed
        # Convert Pydantic models to dicts
        for strategy_model in parsed.strategies:
            strategies.append({
                "strategy": strategy_model.strategy,
                "models_needed": strategy_model.models_needed
            })
        logger.info("Parsed %d strategies from structured output", len(strategies))
    else:
        logger.warning("No structured output received from ensembler")

    if len(strategies) < 8:
        logger.warning("Only found %d strategies, expected 8", len(strategies))

    strategies = strategies[:8]  # Take first 8

    logger.info("Generated %d ensemble strategies", len(strategies))
    for i, strategy_obj in enumerate(strategies, 1):
        strategy_text = strategy_obj.get("strategy", "") if isinstance(strategy_obj, dict) else str(strategy_obj)
        logger.info("  %d. %s", i, strategy_text[:100] + "..." if len(strategy_text) > 100 else strategy_text)

    # Add strategies to ensemble_metadata.json
    ensemble_metadata["strategies"] = strategies

    # Save updated metadata
    with open(metadata_path, "w") as f:
        json.dump(ensemble_metadata, f, indent=2)

    logger.info("Saved strategies to %s", metadata_path)

    return strategies