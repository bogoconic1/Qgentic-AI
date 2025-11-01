"""Tools for EnsemblerAgent: Ensemble strategy testing."""

import logging
import re
from pathlib import Path

import weave
from project_config import get_config
from tools.developer import execute_code
from tools.helpers import call_llm_with_retry, _build_directory_listing
from utils.code_utils import strip_header_from_code
from utils.grade import run_grade
from prompts.tools_ensembler import prompt_ask_ensemble_strategy

logger = logging.getLogger(__name__)

_CONFIG = get_config()
_LLM_CFG = _CONFIG.get("llm")
_RUNTIME_CFG = _CONFIG.get("runtime")

_RESEARCHER_TOOL_OFFLINE_MODEL = _LLM_CFG.get("researcher_tool_offline_model")
_DEFAULT_ASK_ATTEMPTS = _RUNTIME_CFG.get("ask_eda_max_attempts", 3)


@weave.op()
def ask_ensemble_strategy(
    query: str,
    baseline_code_files: list[str],
    ensemble_iteration: int,
    ensemble_folder: Path,
    slug: str,
    description: str,
    metadata: dict,
    max_attempts: int | None = None,
    timeout_seconds: int | None = None
) -> str:
    """
    Test an ensemble strategy with retry logic (similar to ask_eda).

    Takes a strategy description and baseline code files, generates new code
    that implements the ensemble strategy by rewriting/combining baseline code,
    executes it to train the ensemble model, and returns the score.

    Args:
        query: Description of ensemble strategy to implement
        baseline_code_files: List of baseline code files to use (e.g., ["code_5_1_v8.py"])
        ensemble_iteration: Current iteration number (used for submission filename)
        ensemble_folder: Path to ensemble artifacts folder
        slug: Competition slug
        description: Competition description
        metadata: ensemble_metadata.json contents for mapping files to model names
        max_attempts: Maximum retry attempts (default from config)
        timeout_seconds: Execution timeout per attempt (default: baseline_time_limit // 2)

    Returns:
        String with score and insights if successful, or error message
    """
    logger.info(f"Testing ensemble strategy (iteration {ensemble_iteration}): {query}")
    logger.info(f"Using baseline code files: {baseline_code_files}")

    # Default timeout is half of baseline time limit
    if timeout_seconds is None:
        from project_config import get_config
        runtime_cfg = get_config().get("runtime")
        baseline_time_limit = runtime_cfg.get("baseline_time_limit", 14400)  # 4 hours default
        timeout_seconds = baseline_time_limit // 2  # 2 hours default
        logger.info(f"Using timeout: {timeout_seconds}s (baseline_time_limit // 2)")

    # Build directory listing for context
    directory_listing = _build_directory_listing(str(ensemble_folder))
    logger.debug("Prepared directory listing for %s (length=%s)", ensemble_folder, len(directory_listing))

    # Build reverse mapping from code filename to model name
    filename_to_model = {}
    for model_name, model_info in metadata.items():
        best_code_file = model_info.get("best_code_file")
        if best_code_file:
            filename_to_model[best_code_file] = model_name

    # Read baseline code files and strip headers
    baseline_codes = {}
    for code_filename in baseline_code_files:
        code_path = ensemble_folder / code_filename
        if not code_path.exists():
            logger.error(f"Baseline code file not found: {code_path}")
            return f"Error: Baseline code file '{code_filename}' not found in ensemble folder."

        # Map filename to model name
        model_name = filename_to_model.get(code_filename)
        if not model_name:
            logger.error(f"Could not find model name for code file: {code_filename}")
            return f"Error: Could not find model name for '{code_filename}' in ensemble_metadata.json"

        try:
            clean_code = strip_header_from_code(code_path)
            baseline_codes[model_name] = clean_code  # Use model name as key
            logger.info(f"Read baseline code from {code_filename} (model={model_name}, length={len(clean_code)})")
        except Exception as e:
            logger.error(f"Failed to read baseline code from {code_filename}: {e}")
            return f"Error: Failed to read baseline code from '{code_filename}': {str(e)}"

    # Build prompt
    PROMPT = prompt_ask_ensemble_strategy(
        ensemble_folder=str(ensemble_folder),
        directory_listing=directory_listing,
        description=description,
        ensemble_iteration=ensemble_iteration,
        baseline_codes=baseline_codes
    )

    attempts = max_attempts or _DEFAULT_ASK_ATTEMPTS
    input_list = [{"role": "user", "content": "Strategy: " + query}]
    pattern = r'```python\s*(.*?)\s*```'

    for attempt in range(1, attempts + 1):
        logger.info("ask_ensemble_strategy attempt %s/%s", attempt, attempts)

        response = call_llm_with_retry(
            model=_RESEARCHER_TOOL_OFFLINE_MODEL,
            instructions=PROMPT,
            tools=[],
            messages=input_list,
        )
        response_text = response.output_text or ""
        input_list += response.output

        # Extract code from response
        matches = re.findall(pattern, response_text, re.DOTALL)
        code = "\n\n".join(matches).strip()
        logger.debug("ask_ensemble_strategy generated code (truncated): %s...", code[:500])

        if not code:
            input_list.append({"role": "user", "content": "No python code block found. Please try again."})
            logger.warning("ask_ensemble_strategy found no python code block in response.")
            continue

        # Write code to file (following DeveloperAgent format)
        code_file = ensemble_folder / f"ensemble_iteration_{ensemble_iteration}.py"
        with open(code_file, "w") as f:
            f.write(code)

        # Execute code (logs will be written to ensemble_iteration_{ensemble_iteration}.txt by the code itself)
        result = execute_code(str(code_file), timeout_seconds=timeout_seconds)

        # Check if execution timed out
        if "timed out" in result.lower():
            logger.error("ask_ensemble_strategy execution timed out")
            return result

        # Check if execution had errors (execute_code returns error traces with guidance)
        if "Traceback" in result or "Error" in result:
            logger.warning("ask_ensemble_strategy execution failed, retrying with error feedback")
            input_list.append({"role": "user", "content": result})
            continue

        # Success - grade the submission
        submission_file = ensemble_folder / f"submission_ens_{ensemble_iteration}.csv"

        if not submission_file.exists():
            logger.error("Submission file not created: %s", submission_file)
            error_msg = f"Error: Code executed successfully but did not create {submission_file.name}"
            input_list.append({"role": "user", "content": error_msg})
            continue

        # Grade the submission
        logger.info("Grading submission: %s", submission_file)
        grade_info, stdout, returncode, stderr = run_grade(str(submission_file), slug)

        if returncode != 0:
            logger.error("Grading failed: %s", stderr)
            error_msg = f"Grading failed:\n{stdout}\n{stderr}"
            input_list.append({"role": "user", "content": error_msg})
            continue

        # Append grading result to execution output
        grading_result = f"\n\n=== Grading Result ===\n{stdout}"
        final_result = result + grading_result

        logger.info("ask_ensemble_strategy succeeded on attempt %s", attempt)
        if grade_info:
            logger.info("Score: %s", grade_info)

        return final_result

    logger.error("ask_ensemble_strategy exhausted all attempts without success")
    return "Unable to test the strategy after multiple attempts. Please try rephrasing or simplifying."


def get_tools() -> list[dict]:
    """Return tool definitions for EnsemblerAgent."""
    return [
        {
            "type": "function",
            "function": {
                "name": "ask_eda",
                "description": "Analyze ensemble data including correlations, patterns, and distributions. Can execute pandas/numpy queries on submission files in the ensemble folder. Use this to understand model diversity, identify patterns, and guide ensemble strategy. The tool will automatically retry if code execution fails.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "EDA question to answer. Examples: 'Calculate pairwise Pearson and Spearman correlation between all submission files', 'Show prediction variance across models for each sample', 'Find 100 samples where models disagree most', 'Analyze prediction distributions for each model'"
                        }
                    },
                    "required": ["question"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "ask_ensemble_strategy",
                "description": "Test an ensemble strategy by rewriting baseline model code, training the ensemble model, and grading the result. Provide the strategy description and baseline code files to use. The tool will generate new code that implements the ensemble, train it from scratch, create submission_ens_{iteration}.csv, grade it with mlebench, and return the score. Automatically retries if code fails.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Description of ensemble strategy to implement. Examples: 'stacking with LightGBM meta-model', 'use first model for first stage, then feed its OOF predictions to second model as a feature', 'blending with optimized weights on validation set'. The tool will rewrite baseline code to implement this strategy and train the ensemble model."
                        },
                        "baseline_code_files": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of baseline code files to use for the ensemble. Examples: ['code_5_1_v8.py'], ['code_5_1_v8.py', 'code_5_2_v12.py']. These files will be read to understand model architectures and rewritten to implement the ensemble strategy."
                        }
                    },
                    "required": ["query", "baseline_code_files"]
                }
            }
        }
    ]
