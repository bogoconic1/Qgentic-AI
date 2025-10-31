"""Tools for EnsemblerAgent: Ensemble strategy testing."""

import json
import logging
import re
import uuid
from pathlib import Path

import weave
from project_config import get_config
from tools.developer import execute_code
from tools.helpers import call_llm_with_retry, _build_directory_listing
from utils.grade import run_grade

logger = logging.getLogger(__name__)

_CONFIG = get_config()
_LLM_CFG = _CONFIG.get("llm")
_RUNTIME_CFG = _CONFIG.get("runtime")

_RESEARCHER_TOOL_OFFLINE_MODEL = _LLM_CFG.get("researcher_tool_offline_model")
_DEFAULT_ASK_ATTEMPTS = _RUNTIME_CFG.get("ask_eda_max_attempts", 3)


@weave.op()
def test_ensemble_strategy(
    strategy: str,
    ensemble_folder: Path,
    slug: str,
    max_attempts: int | None = None,
    timeout_seconds: int = 300
) -> str:
    """
    Test an ensemble strategy with retry logic (similar to ask_eda).

    Takes a strategy description, generates code via LLM, executes it,
    grades the result, and returns the score. Automatically retries on errors.

    Args:
        strategy: Description of ensemble strategy to test
        ensemble_folder: Path to ensemble artifacts
        slug: Competition slug
        max_attempts: Maximum retry attempts (default from config)
        timeout_seconds: Execution timeout per attempt (default: 5 minutes)

    Returns:
        String with score and insights if successful, or error message
    """
    logger.info(f"Testing ensemble strategy: {strategy}")

    # Build directory listing for context
    directory_listing = _build_directory_listing(str(ensemble_folder))
    logger.debug("Prepared directory listing for %s (length=%s)", ensemble_folder, len(directory_listing))

    # Build prompt for code generation
    PROMPT = f"""You are an expert data scientist testing ensemble strategies.

Available files in ensemble folder:
{directory_listing}

Competition slug: {slug}

Your task is to write Python code that:
1. Loads submission_model_*.csv files from the current directory
2. Implements the specified ensemble strategy
3. Saves the result to '_temp_submission.csv' with correct format (id + target columns)
4. Grades the submission using mlebench and prints the score

Guidelines:
- Use pandas, numpy for ensemble computation
- Working directory is: {ensemble_folder}
- After creating '_temp_submission.csv', grade it using:
  ```python
  import subprocess
  result = subprocess.run(['mlebench', 'grade-sample', '_temp_submission.csv', '{slug}'],
                          capture_output=True, text=True)
  print(result.stdout)  # This contains the score
  ```
- Print the score clearly
- Keep output concise and relevant

Output your code in a ```python code block.
"""

    attempts = max_attempts or _DEFAULT_ASK_ATTEMPTS
    input_list = [{"role": "user", "content": "Strategy: " + strategy}]
    pattern = r'```python\s*(.*?)\s*```'

    for attempt in range(1, attempts + 1):
        logger.info("test_ensemble_strategy attempt %s/%s", attempt, attempts)

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
        logger.debug("test_ensemble_strategy generated code (truncated): %s...", code[:500])

        if not code:
            input_list.append({"role": "user", "content": "No python code block found. Please try again."})
            logger.warning("test_ensemble_strategy found no python code block in response.")
            continue

        # Write code to temporary file
        temp_filename = f"_temp_ensemble_test_{uuid.uuid4().hex[:8]}.py"
        temp_code_path = ensemble_folder / temp_filename

        # Prepend setup
        full_code = f"""import os
os.chdir(r'{ensemble_folder}')

{code}
"""

        try:
            with open(temp_code_path, "w") as f:
                f.write(full_code)

            # Execute code
            result = execute_code(str(temp_code_path), timeout_seconds=timeout_seconds)

            # Check if execution timed out
            if "timed out" in result.lower():
                logger.error("test_ensemble_strategy execution timed out")
                return result

            # Check if execution had errors (execute_code returns error traces with guidance)
            if "Traceback" in result or "Error" in result:
                logger.warning("test_ensemble_strategy execution failed, retrying with error feedback")
                input_list.append({"role": "user", "content": result})
                continue

            # Success - return the output
            logger.info("test_ensemble_strategy succeeded on attempt %s", attempt)
            return result

        finally:
            # Clean up temp files
            temp_code_path.unlink(missing_ok=True)
            temp_sub_path = ensemble_folder / "_temp_submission.csv"
            temp_sub_path.unlink(missing_ok=True)

    logger.error("test_ensemble_strategy exhausted all attempts without success")
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
                "name": "test_ensemble_strategy",
                "description": "Test an ensemble strategy by generating code, executing it, and grading the result. Describe the strategy you want to test (e.g., 'weighted average by model scores', 'rank averaging', 'simple mean'). The tool will generate code, create submission, grade it with mlebench, and return the score. Automatically retries if code fails.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "strategy": {
                            "type": "string",
                            "description": "Description of ensemble strategy to test. Examples: 'weighted average using model scores as weights', 'simple arithmetic mean of all predictions', 'rank averaging across all models', 'median of predictions'. The tool will generate code to implement this strategy, save to _temp_submission.csv, grade it, and print the score."
                        }
                    },
                    "required": ["strategy"]
                }
            }
        }
    ]
