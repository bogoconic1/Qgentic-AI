import logging
import os
import re
import shlex
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
import kagglehub
import json
import kaggle
import yaml
import pandas as pd
import weave
from project_config import get_config
from tools.helpers import call_llm_with_retry, _build_directory_listing
from prompts.tools_researcher import (
    ask_eda_template as prompt_ask_eda,
    datasets_prompt as prompt_datasets,
)
load_dotenv()


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

_CONFIG = get_config()
_LLM_CFG = _CONFIG.get("llm")
_RUNTIME_CFG = _CONFIG.get("runtime")
_PATH_CFG = _CONFIG.get("paths")

_RESEARCHER_TOOL_OFFLINE_MODEL = _LLM_CFG.get("researcher_tool_offline_model")
_RESEARCHER_TOOL_ONLINE_MODEL = _LLM_CFG.get("researcher_tool_online_model")
_DEFAULT_ASK_ATTEMPTS = _RUNTIME_CFG.get("ask_eda_max_attempts")
_DEFAULT_DOWNLOAD_ATTEMPTS = _RUNTIME_CFG.get("download_datasets_max_attempts")
_TASK_ROOT = Path(_PATH_CFG.get("task_root"))
_EXTERNAL_DIRNAME = _PATH_CFG.get("external_data_dirname")


@weave.op()
def ask_eda(question: str, description: str, data_path: str, max_attempts: int | None = None, timeout_seconds: int = 600) -> str:
    """Asks a question about the data provided for exploratory data analysis (EDA)

    Args:
        question: The EDA question to answer
        description: Competition description
        data_path: Path to the data directory
        max_attempts: Maximum number of attempts (default from config)
        timeout_seconds: Timeout for code execution in seconds (default 600 = 10 minutes)
    """
    # Prepare media directory for EDA charts and expose to executed code
    try:
        preset_media = os.environ.get("MEDIA_DIR", "").strip()
        if preset_media:
            media_dir = Path(preset_media)
        else:
            media_dir = (Path(data_path) / "media")
            os.environ["MEDIA_DIR"] = str(media_dir)
        media_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        logger.exception("Failed to ensure MEDIA_DIR under %s", data_path)

    directory_listing = _build_directory_listing(data_path)
    logger.debug("Prepared directory listing for %s (length=%s)", data_path, len(directory_listing))

    attempts = max_attempts or _DEFAULT_ASK_ATTEMPTS
    PROMPT = prompt_ask_eda(data_path, directory_listing, description)
    input_list = [{"role": "user", "content": "Question: " + question}]

    pattern = r'```python\s*(.*?)\s*```'

    for attempt in range(1, attempts + 1):
        logger.info("ask_eda attempt %s/%s", attempt, attempts)
        response = call_llm_with_retry(
            model=_RESEARCHER_TOOL_OFFLINE_MODEL,
            instructions=PROMPT,
            tools=[],
            messages=input_list,
        )
        response_text = response.output_text or ""
        input_list += response.output

        matches = re.findall(pattern, response_text, re.DOTALL)
        code = "\n\n".join(matches).strip()
        logger.debug("ask_eda generated code (truncated): %s", code)

        if not code:
            input_list.append({"role": "user", "content": "No python code block found. Please try again."})
            logger.warning("ask_eda found no python code block in response.")
            continue
        else:
            # Write code to temporary file
            code_file = Path(data_path) / "eda_temp.py"
            with open(code_file, "w") as f:
                f.write(code)

            # Import execute_code here to avoid circular import at module level
            from tools.developer import execute_code

            # Execute code using execute_code() function
            result = execute_code(str(code_file), timeout_seconds=timeout_seconds)

            # Check if execution timed out
            if "timed out" in result.lower():
                logger.error("ask_eda execution timed out")
                return result

            # Check if execution had errors (execute_code returns error traces with guidance)
            # If there's an error, it will contain stack trace with web search guidance
            if "Traceback" in result or "Error" in result:
                logger.warning("ask_eda execution failed, retrying with error feedback")
                input_list.append({"role": "user", "content": result})
                continue

            # Success - return the output
            logger.info("ask_eda succeeded on attempt %s", attempt)
            return result

    logger.error("ask_eda exhausted all attempts without success")
    return "Your question cannot be answered."

@weave.op()
def download_external_datasets(question_1: str, question_2: str, question_3: str, slug: str, max_attempts: int | None = None) -> str:
    """Downloads external datasets using 3 different phrasings to maximize search coverage"""
    logger.debug("Dataset queries: q1=%s, q2=%s, q3=%s", question_1, question_2, question_3)
    attempts = max_attempts or _DEFAULT_DOWNLOAD_ATTEMPTS
    

    comp_dir = _TASK_ROOT / slug
    with open(comp_dir / "comp_metadata.yaml", "r") as f:
        COMP_METADATA = yaml.safe_load(f)

    PROMPT = prompt_datasets()
    pattern = r'```json\s*(\{.*?\})\s*```'

    relevant_datasets = []

    # Loop through 3 different query phrasings
    for q_idx, query in enumerate([question_1, question_2, question_3], start=1):
        logger.info("Processing query phrasing %s/3: %s", q_idx, query)
        input_list = [{"role": "user", "content": "Dataset Name: " + query}]

        # For each query, retry up to 'attempts' times
        for attempt in range(1, attempts + 1):
            logger.info("Dataset discovery attempt %s/%s for query %s", attempt, attempts, q_idx)
            response = call_llm_with_retry(
                model=_RESEARCHER_TOOL_ONLINE_MODEL,
                instructions=PROMPT,
                tools=[],
                messages=input_list,
                web_search_enabled=True,
            )
            completion = response.output_text or ""

            matches = re.findall(pattern, completion, re.DOTALL)
            if not matches:
                logger.warning("No JSON block found in web search response for query %s attempt %s.", q_idx, attempt)
                continue
            else:
                data = json.loads(matches[0])
                new_datasets = data.get("datasets", [])
                logger.info("Query %s found %s datasets: %s", q_idx, len(new_datasets), new_datasets)
                relevant_datasets.extend(new_datasets)
                break  # Success, move to next query

    # Deduplicate datasets
    relevant_datasets = list(set(relevant_datasets))
    logger.info("Found %s unique datasets from all 3 query phrasings.", len(relevant_datasets))

    # this should fallback - because we will still read the directory
    external_root_env = os.environ.get("EXTERNAL_DATA_DIR").strip()
    if external_root_env:
        external_root = Path(external_root_env)
        dest_path = external_root
    else:
        dest_path = comp_dir
        external_root = dest_path / _EXTERNAL_DIRNAME
    external_root.mkdir(parents=True, exist_ok=True)

    for dataset in relevant_datasets:
        try:
            kaggle_url = "/".join(dataset.split("/")[-2:])
            datasets_list = kaggle.api.dataset_list(search=kaggle_url)
            dataset_metadata = vars(datasets_list[0])
            if pd.to_datetime(dataset_metadata['lastUpdated']) > pd.to_datetime(COMP_METADATA['END_DATE']):
                logger.info("Skipping dataset %s as it was published after competition end date. last update: %s, comp end: %s", kaggle_url, dataset_metadata['lastUpdated'], COMP_METADATA['END_DATE'])
                continue

            logger.info("Downloading dataset: %s", kaggle_url)
            path = kagglehub.dataset_download(kaggle_url)
            logger.info("Dataset downloaded to temporary path: %s", path)
            # os cp -r to configured task directory
            path_parts = Path(path).parts
            folder_name = path_parts[-3] if len(path_parts) >= 3 else Path(path).name
            dest_folder = external_root / folder_name
            dest_folder.mkdir(parents=True, exist_ok=True)
            src_path = shlex.quote(str(path))
            dest_path_str = shlex.quote(str(dest_folder))
            os.system(f"cp -r {src_path}/* {dest_path_str}")
            logger.info("Dataset downloaded to: %s", dest_path)
        except:
            logger.exception("Failed to download dataset: %s", dataset)
            continue

    dest_files = _build_directory_listing(str(dest_path))
    logger.info(f"Current files in {dest_path}:\n{dest_files}")
    return f'Relevant Datasets are downloaded: now {dest_path} contains: \n{dest_files}'



def get_tools():
    return [
        {
            "type": "function",
            "name": "ask_eda",
            "description": "Ask a question to the EDA expert",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The question to ask the EDA expert"}
                },
            },
            "additionalProperties": False,
            "required": ['question']
        },
        {
            "type": "function",
            "name": "run_ab_test",
            "description": "Run short (≤10-minute) A/B tests to validate modeling or feature engineering choices by comparing their impact on performance",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The A/B testing question (e.g., 'Run a 5-fold CV on 50,000 rows using XGBoost comparing raw vs log-transformed target and report RMSE')"}
                },
            },
            "additionalProperties": False,
            "required": ['question']
        },
        {
            "type": "function",
            "name": "download_external_datasets",
            "description": "Download external data to working directory by searching with 3 different phrasings to maximize search coverage",
            "parameters": {
                "type": "object",
                "properties": {
                    "question_1": {"type": "string", "description": "First phrasing of the dataset query"},
                    "question_2": {"type": "string", "description": "Second phrasing with different wording"},
                    "question_3": {"type": "string", "description": "Third phrasing using alternative keywords"}
                },
            },
            "additionalProperties": False,
            "required": ["question_1", "question_2", "question_3"],
        },
    ]
