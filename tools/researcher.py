import logging
import os
import re
import sys
from io import StringIO

from dotenv import load_dotenv
from openai import OpenAI
import kagglehub
import json
import kaggle
import yaml
import pandas as pd

load_dotenv()


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _build_directory_listing(root: str) -> str:
    lines: list[str] = []
    for current_root, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d != "outputs")
        rel_root = os.path.relpath(current_root, root)
        depth = 0 if rel_root in (".", "") else rel_root.count(os.sep) + 1
        indent = "    " * depth
        folder_display = "." if rel_root in (".", "") else rel_root
        lines.append(f"{indent}{folder_display}/")
        for name in sorted(files):
            lines.append(f"{indent}    {name}")
    return "\n".join(lines)


def ask_eda(question: str, description: str, data_path: str, max_attempts: int = 5) -> str:
    """Asks a question about the data provided for exploratory data analysis (EDA)"""
    client = OpenAI(api_key=os.environ.get("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")
    directory_listing = _build_directory_listing(data_path)
    logger.debug(
        "Prepared directory listing for %s (length=%s)", data_path, len(directory_listing)
    )
    PROMPT = f"""You are an experienced Kaggle Competitions Grandmaster. Your goal is to write code that answers questions about the data provided.
Competition Description:
{description}

You will be given one or more questions related to the data. Your task is to generate code that, when executed, will answer all questions using the data provided.
Before generating the code, provide around 5 lines of reasoning about the approach.

The data files are stored in the directory "{data_path}".

Directory listing:
{directory_listing}

So make sure you put your code within a python code block like this:
```python
data_path = "{data_path}"
<YOUR CODE HERE>
```

IMPORTANT: Always provide descriptive answers. Instead of just printing a number like "100", print a complete sentence like "There are a total of 100 records". Make your final answer clear and informative.
"""
    all_messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": question},
    ]

    pattern = r'```python\s*(.*?)\s*```'

    last_error = ""
    for attempt in range(1, max_attempts + 1):
        logger.info("ask_eda attempt %s/%s", attempt, max_attempts)
        completion = client.chat.completions.create(
            extra_body={},
            model="openai/gpt-5",
            messages=all_messages
        )
        response_text = completion.choices[0].message.content or ""

        matches = re.findall(pattern, response_text, re.DOTALL)
        code = "\n\n".join(matches).strip()
        logger.debug("ask_eda generated code (truncated): %s", code[:500])

        if not code:
            last_error = "No python code block found in the model response."
            logger.warning(last_error)
        else:
            try:
                # Simple policy assertion carried forward
                assert "train_labels.csv" not in code or "!= 'Missing'" in code, (
                    "You must remove records with type = 'Missing' before doing any analysis."
                )

                # Save current stdout
                old_stdout = sys.stdout
                sys.stdout = captured_output = StringIO()

                exec(code)
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
                last_error = f"Error executing code: {str(e)}"
                logger.exception("ask_eda execution failed: %s", e)

    logger.error(
        "ask_eda exhausted all attempts without success. Last error: %s", last_error
    )
    return "Your question cannot be answered."

def download_external_datasets(query: str, slug: str) -> str:
    """Downloads external datasets"""
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    logger.debug("Dataset query: %s", query)

    with open(f"task/{slug}/comp_metadata.yaml", "r") as f:
        COMP_METADATA = yaml.safe_load(f)

    messages = [
        {
            "role": "user",
            "content": f"""Search for Kaggle datasets which are relevant to the following query: {query}. Provide up till 5 dataset URLs.
At the end of your reasoning/explanation, your response should be in strict JSON format within backticks:
```json
{{"datasets": ["https://www.kaggle.com/datasets/<user1>/<dataset1>", "https://www.kaggle.com/datasets/<user2>/<dataset2>", ...]}}
```
""",
        }
    ]
    logger.debug("Web search messages: %s", messages)

    try:
        response = client.responses.create(
            model="gpt-5",
            input=messages,
            tools=[{"type": "web_search"}],
        )

        logger.info("Received web search response for dataset query.")
        logger.debug("Web search raw response: %s", response)
    except Exception:
        logger.exception("Web search request for dataset failed.")
        raise

    completion = response.output[-1].content[0].text
    logger.debug("Web search completion text: %s", completion)
    pattern = r'```json\s*(\{.*?\})\s*```'
    matches = re.findall(pattern, completion, re.DOTALL)
    if not matches:
        logger.warning("No JSON block found in web search response.")
        return "No relevant datasets found."

    data = json.loads(matches[0])
    datasets = data.get("datasets", [])
    if not datasets:
        logger.info("No datasets found in web search JSON response.")
        return "No relevant datasets found."
    logger.info("Found %s relevant datasets from web search.", len(datasets))
    dest_path = f"task/{slug}"
    for dataset in datasets:
        try:
            kaggle_url = "/".join(dataset.split("/")[-2:])
            datasets = kaggle.api.dataset_list(search=kaggle_url)
            dataset_metadata = vars(datasets[0])
            if pd.to_datetime(dataset_metadata['lastUpdated']) > pd.to_datetime(COMP_METADATA['END_DATE']):
                logger.info("Skipping dataset %s as it was published after competition end date. last update: %s, comp end: %s", kaggle_url, dataset_metadata['lastUpdated'], COMP_METADATA['END_DATE'])
                continue

            logger.info("Downloading dataset: %s", kaggle_url)
            path = kagglehub.dataset_download(kaggle_url)
            logger.info("Dataset downloaded to temporary path: %s", path)
            # os cp -r to task/{slug}
            os.system(f"cp -r {path}/* {dest_path}")
            logger.info("Dataset downloaded to: %s", dest_path)
        except:
            logger.exception("Failed to download dataset: %s", dataset)
            continue
    dest_files = "\n- ".join(os.listdir(dest_path))
    logger.info(f"Current files in task/{slug}:\n- %s", dest_files)
    return f'Relevant Datasets are downloaded: now task/{slug} contains: \n{dest_files}'
    

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
