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
from concurrent.futures import ThreadPoolExecutor
from utils.grade import run_grade
from utils.code_utils import strip_header_from_code
from tools.helpers import call_llm_with_retry
from project_config import get_config
from schemas.ensembler import EnsembleStrategies
from prompts.ensembler_agent import (
    ensembler_code_summarize_system_prompt,
    ensembler_code_summarize_user_prompt,
)

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
    model_summaries = []

    for model_name, model_data in ensemble_metadata.items():
        # Skip the "strategies" field if it exists
        if model_name in ["strategies", "ensemble_checklist", "validation_summary"] or not isinstance(model_data, dict):
            continue

        score = model_data.get("best_score", "N/A")
        training_time = model_data.get("training_time", "N/A")
        blacklisted = model_data.get("blacklisted_ideas", [])
        technical_summary = model_data.get("technical_summary", "")

        metadata_summary.append(f"- {model_name}: score={score}, training_time={training_time}")

        # Collect all blacklisted ideas
        for idea in blacklisted:
            all_blacklisted_ideas.append(f"  [{model_name}] {idea}")

        # Collect technical summaries if available
        if technical_summary:
            model_summaries.append(f"### {model_name}\n{technical_summary}")

    metadata_summary_str = "\n".join(metadata_summary)

    # Format blacklisted ideas
    if all_blacklisted_ideas:
        blacklisted_ideas_str = "\n".join(all_blacklisted_ideas)
    else:
        blacklisted_ideas_str = "None"

    # Format model summaries
    if model_summaries:
        model_summaries_str = "\n\n".join(model_summaries)
    else:
        model_summaries_str = "No technical summaries available"

    system_prompt = f"""You are an expert Kaggle competitor specializing in ensemble methods and model optimization.

# Role and Objective
Develop and recommend 8 diverse, independent, and actionable ensemble strategies designed to outperform the current baseline models in a Kaggle competition.

# Instructions
- Begin with a concise conceptual checklist (3-7 bullet points) summarizing your overall approach (exclude implementation details).
- For each strategy, provide:
  - A comprehensive description
  - Exact model names required (as strings)
  - The ensemble technique employed (named method, e.g., 'weighted blending', 'stacking')
  - Key actionable steps or tips for implementation

# Ensembling Categories
1. **Model Upgrade** (minimum two strategies)
   - E.g., swap baseline architectures for more powerful alternatives (e.g., 'deberta-v3-base' to 'deberta-v3-large').
   - **IMPORTANT**: When upgrading a model, you MUST reference the CODE implementation from the baseline model being upgraded. Copy the exact preprocessing, augmentation, loss function, and training pipeline from the original baseline model's technical summary.
   - You must list the original baseline model in `models_needed`.
   - If unsuitable, state: "No suitable strategy could be found for this item."

2. **Traditional Ensembling** (minimum two strategies)
   - E.g., blend or average outputs from different models using conventional ensemble techniques.
   - You can ensemble at most 3 models per strategy.
   - If inappropriate, state: "No suitable strategy could be found for this item."

3. **Advanced Techniques** (remaining strategies)
   - E.g., multi-stage training, pseudo-labeling, knowledge distillation, etc.
   - You can ensemble at most 3 models per strategy.

- Ensure that:
  - Recommendations are specific and actionable—not general or vague.
  - You can ensemble at most 3 models per strategy.
  - Recommendations span multiple ensemble categories as listed.

- Present strategies as a numbered list (1–8) following the provided structure.
- Before performing any web search or recommending a contemporary (2024-2025) methodology, state your intent clearly and specify the minimal required inputs.
- Use only the tools explicitly allowed for this task. Do not reference or search for past winning solutions for this competition.
- Do not suggest blacklisted or substantially similar strategies; if unfillable, use: "No suitable strategy could be found for this item."
- After enumerating the strategies, validate that all are distinct, actionable, independent, and do not overlap blacklisted ideas. Substitute as needed to maximize diversity, and proceed or self-correct if the validation fails.
- Conclude with a brief validation summary stating whether requirements were met or substitutions were made.
- Set reasoning_effort = medium. Adjust the detail of your reasoning and output to match this complexity for balance.

# Context
- <competition_description>
  {description}
  </competition_description>
- <baseline_models> (model names [string] and training times [e.g., '2h'])
  {metadata_summary_str}
  </baseline_models>
- <blacklisted_ideas>
  The following approaches did NOT improve performance—avoid similar methods:
  {blacklisted_ideas_str}
  </blacklisted_ideas>
- <model_technical_summaries>
  Technical implementation details for each baseline model:
  {model_summaries_str}
  </model_technical_summaries>

## Output Format
Return a JSON object structured as follows:
- `checklist`: Array of 3–7 short bullet points capturing conceptual ensemble strategy development steps (exclude implementation detail).
- `strategies`: Array of exactly 8 strategy objects, each including:
  - `strategy` (string): Detailed, high-level description and actionable guidance. Add any instructions of model changes/upgrades here.
  - `models_needed` (array of strings): Required model names. The models must be present in <baseline_models>.
  - `ensemble_method` (string): Name of ensemble technique (e.g., 'weighted blending', 'stacking', 'bagging', etc.).
  - `implementation_guidance` (string): Actionable steps or practical tips.
- `validation_summary`: Summarize whether all strategies are distinct, actionable, independent, avoid the blacklist, and meet requirements. If substitutions were needed, note specifics.
- If any strategy category cannot be filled, include an object with `strategy`: 'No suitable strategy could be found for this item.' and leave arrays/strings for other fields empty.
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
    checklist = []
    validation_summary = ""

    if response and hasattr(response, 'output_parsed') and response.output_parsed:
        parsed = response.output_parsed

        # Extract checklist and validation summary
        checklist = parsed.checklist if hasattr(parsed, 'checklist') else []
        validation_summary = parsed.validation_summary if hasattr(parsed, 'validation_summary') else ""

        # Convert Pydantic models to dicts
        for strategy_model in parsed.strategies:
            strategies.append({
                "strategy": strategy_model.strategy,
                "models_needed": strategy_model.models_needed,
                "ensemble_method": strategy_model.ensemble_method,
                "implementation_guidance": strategy_model.implementation_guidance
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

    # Add strategies, checklist, and validation summary to ensemble_metadata.json
    ensemble_metadata["ensemble_checklist"] = checklist
    ensemble_metadata["strategies"] = strategies
    ensemble_metadata["validation_summary"] = validation_summary

    # Save updated metadata
    with open(metadata_path, "w") as f:
        json.dump(ensemble_metadata, f, indent=2)

    logger.info("Saved strategies to %s", metadata_path)

    return strategies


def generate_model_summary(
    description: str,
    model_name: str,
    code: str,
    logs: str,
) -> str:
    """
    Generate a technical summary of a model's implementation.

    Args:
        description: Competition description
        model_name: Model name (e.g., "microsoft/deberta-v3-base")
        code: Model implementation code
        logs: Execution logs

    Returns:
        Technical summary markdown string
    """
    system_prompt = ensembler_code_summarize_system_prompt()
    user_prompt = ensembler_code_summarize_user_prompt(description, model_name, code, logs)

    response = call_llm_with_retry(
        model=_ENSEMBLER_MODEL,
        instructions=system_prompt,
        tools=[],
        messages=[{"role": "user", "content": user_prompt}]
    )

    return response.output_text


def _generate_summary_task(
    idx: int,
    model_name: str,
    baseline_info: dict,
    description: str,
    ensemble_dir: Path,
    summaries_dir: Path,
):
    """Helper function to generate a single model summary (for parallel execution)."""
    print(f"Generating summary for Model {idx}: {model_name}")

    # Read code file
    best_code_file = baseline_info.get("best_code_file")
    if not best_code_file:
        print(f"  WARNING: No code file for {model_name}, skipping")
        return idx, model_name, None

    # Code is in ensemble/ folder
    code_path = ensemble_dir / best_code_file
    if not code_path.exists():
        print(f"  WARNING: Code file not found: {code_path}, skipping")
        return idx, model_name, None

    # Read code and strip headers
    code = strip_header_from_code(code_path)

    # Read logs file
    log_file = best_code_file.replace('.py', '.txt')
    log_path = ensemble_dir / log_file
    if not log_path.exists():
        print(f"  WARNING: Log file not found: {log_path}, using empty logs")
        logs = ""
    else:
        with open(log_path, 'r') as f:
            logs = f.read()

    # Generate summary
    summary = generate_model_summary(
        description=description,
        model_name=model_name,
        code=code,
        logs=logs
    )

    # Save summary
    summary_path = summaries_dir / f"summary_model_{idx}.md"
    with open(summary_path, 'w') as f:
        f.write(summary)

    print(f"  Saved summary to {summary_path}")
    return idx, model_name, summary


def generate_ensemble_summaries(
    slug: str,
    iteration: int,
) -> dict:
    """
    Generate technical summaries for all baseline models in ensemble folder.

    Args:
        slug: Competition slug
        iteration: Iteration number

    Returns:
        Dict mapping model_name to summary markdown
    """
    # Define paths
    base_path = _TASK_ROOT / slug
    outputs_dir = base_path / _OUTPUTS_DIRNAME / str(iteration)
    ensemble_dir = outputs_dir / "ensemble"
    metadata_path = ensemble_dir / "ensemble_metadata.json"
    description_path = base_path / "description.md"

    # Load description
    with open(description_path, 'r') as f:
        description = f.read()

    # Load ensemble metadata
    with open(metadata_path, 'r') as f:
        ensemble_metadata = json.load(f)

    summaries = {}
    summaries_dir = ensemble_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    # Filter out non-model entries
    baseline_results = {
        k: v for k, v in ensemble_metadata.items()
        if k not in ["strategies", "ensemble_checklist", "validation_summary"]
        and isinstance(v, dict)
    }

    print(f"Generating summaries for {len(baseline_results)} models in parallel")

    # Run summary generation in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(baseline_results)) as executor:
        futures = []
        for idx, (model_name, baseline_info) in enumerate(baseline_results.items(), start=1):
            future = executor.submit(
                _generate_summary_task,
                idx, model_name, baseline_info, description,
                ensemble_dir, summaries_dir
            )
            futures.append(future)

        # Collect results
        for future in futures:
            try:
                idx, model_name, summary = future.result()
                if summary:
                    summaries[model_name] = summary
            except Exception as e:
                print(f"  ERROR: Summary generation failed: {e}")
                continue

    # Update ensemble_metadata.json with summaries
    for model_name, summary in summaries.items():
        if model_name in ensemble_metadata:
            ensemble_metadata[model_name]["technical_summary"] = summary

    # Save updated metadata
    with open(metadata_path, "w") as f:
        json.dump(ensemble_metadata, f, indent=2)

    logger.info(f"Generated {len(summaries)} summaries and updated ensemble_metadata.json")

    return summaries