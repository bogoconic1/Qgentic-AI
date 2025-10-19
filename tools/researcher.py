import logging
import os
import re
import shlex
import sys
import traceback
from io import StringIO
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
from tools.developer import web_search_stack_trace
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
_LLM_CFG = _CONFIG.get("llm", {}) if isinstance(_CONFIG, dict) else {}
_RUNTIME_CFG = _CONFIG.get("runtime", {}) if isinstance(_CONFIG, dict) else {}
_PATH_CFG = _CONFIG.get("paths", {}) if isinstance(_CONFIG, dict) else {}

_BASE_URL = _LLM_CFG.get("base_url", "https://openrouter.ai/api/v1")
_API_KEY_ENV = _LLM_CFG.get("api_key_env", "OPENROUTER_API_KEY")
_RESEARCHER_MODEL = _LLM_CFG.get("researcher_model", "google/gemini-2.5-pro")
_RESEARCHER_TOOL_OFFLINE_MODEL = _LLM_CFG.get(
    "researcher_tool_offline_model",
    _RESEARCHER_MODEL,
)
_RESEARCHER_TOOL_ONLINE_MODEL = _LLM_CFG.get(
    "researcher_tool_online_model",
    _RESEARCHER_MODEL,
)
_DEFAULT_ASK_ATTEMPTS = _RUNTIME_CFG.get("ask_eda_max_attempts", 5)
_TASK_ROOT = Path(_PATH_CFG.get("task_root", "task"))
_EXTERNAL_DIRNAME = _PATH_CFG.get("external_data_dirname", "external-data")


@weave.op()
def ask_eda(question: str, description: str, data_path: str, max_attempts: int | None = None) -> str:
    """Asks a question about the data provided for exploratory data analysis (EDA)"""
    client = OpenAI(api_key=os.environ.get(_API_KEY_ENV), base_url=_BASE_URL)
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
    logger.debug(
        "Prepared directory listing for %s (length=%s)", data_path, len(directory_listing)
    )
    attempts = max_attempts or _DEFAULT_ASK_ATTEMPTS
    PROMPT = prompt_ask_eda(data_path, directory_listing, description)
    all_messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": question},
    ]

    pattern = r'```python\s*(.*?)\s*```'

    last_error = ""
    for attempt in range(1, attempts + 1):
        logger.info("ask_eda attempt %s/%s", attempt, attempts)
        completion = call_llm_with_retry(
            client,
            model=_RESEARCHER_TOOL_OFFLINE_MODEL,
            messages=all_messages,
        )
        try:
            msg = completion.choices[0].message
            response_text = msg.content or ""
        except Exception:
            msg = ""
            response_text = ""
        assistant_message = {"role": "assistant", "content": response_text}

        matches = re.findall(pattern, response_text, re.DOTALL)
        code = "\n\n".join(matches).strip()
        logger.debug("ask_eda generated code (truncated): %s", code)

        if not code:
            last_error = "No python code block found in the model response."
            logger.warning(last_error)
            all_messages.append(assistant_message)
        else:
            try:
                # Simple policy assertion carried forward
                assert "train_labels.csv" not in code or "!= 'Missing'" in code, (
                    "You must remove records with type = 'Missing' before doing any analysis."
                )

                # Save current stdout
                old_stdout = sys.stdout
                sys.stdout = captured_output = StringIO()

                with open(f"code_abc.py", "w") as f:
                    f.write(code)

                exec(open(f"code_abc.py").read(), globals())
                output = captured_output.getvalue()

                sys.stdout = old_stdout

                logger.info("ask_eda succeeded on attempt %s", attempt)
                return output

            except AssertionError as e:
                try:
                    sys.stdout = old_stdout
                except Exception:
                    pass
                last_error = f"Assertion failed: {e}"
                logger.warning(last_error)
            except Exception as e:
                try:
                    sys.stdout = old_stdout
                except Exception:
                    pass
                all_messages.append(assistant_message)
                stack_trace = traceback.format_exc()
                last_error = f"Error executing code: {str(e)}"
                logger.exception("ask_eda execution failed: %s", e)

                try:
                    search_result = web_search_stack_trace(stack_trace)
                    if search_result:
                        logger.info("ask_eda web search guidance retrieved.")
                        last_error = (
                            f"{last_error}\n\nStack trace:\n{stack_trace}\n\n"
                            f"Web search guidance:\n{search_result}"
                        )
                        all_messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "The previous code attempt failed with this stack trace:\n"
                                    f"{stack_trace}\n\nHere is external guidance that may help fix it:\n"
                                    f"{search_result}\n\n"
                                    "Please provide a corrected solution."
                                ),
                            }
                        )
                except Exception:
                    logger.exception("ask_eda web search for stack trace failed.")

    logger.error(
        "ask_eda exhausted all attempts without success. Last error: %s", last_error
    )
    return "Your question cannot be answered."

@weave.op()
def download_external_datasets(query: str, slug: str) -> str:
    """Downloads external datasets"""
    client = OpenAI(api_key=os.environ.get(_API_KEY_ENV), base_url=_BASE_URL)
    logger.debug("Dataset query: %s", query)

    comp_dir = _TASK_ROOT / slug
    with open(comp_dir / "comp_metadata.yaml", "r") as f:
        COMP_METADATA = yaml.safe_load(f)

    messages = [
        {
            "role": "user",
            "content": prompt_datasets(query),
        }
    ]
    logger.debug("Web search messages: %s", messages)

    try:
        content = ""
        while content == "":
            completion = call_llm_with_retry(
                client,
                model=_RESEARCHER_TOOL_ONLINE_MODEL,
                messages=messages,
            )
            try:
                msg = completion.choices[0].message
                content = msg.content or ""
            except Exception:
                msg = ""
                content = ""

        logger.info("Received web search response for dataset query.")
        logger.debug("Web search raw response: %s", content)
    except Exception:
        logger.exception("Web search request for dataset failed.")
        return "No relevant datasets found."

    completion = content
    relevant_datasets = []
    max_dataset_attempts = 3
    for dataset_attempt in range(1, max_dataset_attempts + 1):
        logger.debug("Web search completion text: %s", completion)
        pattern = r'```json\s*(\{.*?\})\s*```'
        matches = re.findall(pattern, completion, re.DOTALL)

        if not matches:
            logger.warning("No JSON block found in web search response.")
            relevant_datasets = []
        else:
            data = json.loads(matches[0])
            relevant_datasets = data.get("datasets", [])

        if len(relevant_datasets) == 0 and dataset_attempt < max_dataset_attempts:
            logger.info(
                "Datasets key is empty; retrying dataset discovery (%s/%s)",
                dataset_attempt,
                max_dataset_attempts,
            )
            # Re-query for datasets and try parsing again
            content = ""
            while content == "":
                completion_obj = call_llm_with_retry(
                    client,
                    model=_RESEARCHER_TOOL_ONLINE_MODEL,
                    messages=messages,
                )
                try:
                    msg = completion_obj.choices[0].message
                    content = msg.content or ""
                except Exception:
                    msg = ""
                    content = ""
            completion = content
            continue
        break

    if relevant_datasets is None:
        logger.info("Datasets still None after %s attempts.", max_dataset_attempts)
        return "No relevant datasets found."
    if not relevant_datasets:
        logger.info("No datasets found in web search JSON response.")
        return "No relevant datasets found."
    logger.info("Found %s relevant datasets from web search.", len(relevant_datasets))
    external_root_env = os.environ.get("EXTERNAL_DATA_DIR", "").strip()
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
            "function": {
                "name": "ask_eda",
                "description": "Ask a question to the EDA expert",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "The question to ask the EDA expert"}
                    },
                    "required": ["question"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "download_external_datasets",
                "description": "Download external data to working directory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Short description of the external data need"}
                    },
                    "required": ["query"],
                },
            },
        },
    ]
