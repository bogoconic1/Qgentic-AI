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
from tools.gemini_google_search import GeminiPaperSummaryClient
from prompts.tools_researcher import (
    ask_eda_template as prompt_ask_eda,
    datasets_prompt as prompt_datasets,
)
from schemas.researcher import DatasetDiscovery
from utils.code_utils import extract_python_code
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
def ask_eda(question: str, description: str, data_path: str, max_attempts: int | None = None, timeout_seconds: int = 1800, previous_ab_tests: list[dict] | None = None, file_suffix: str = "", cpu_core_range: list[int] | None = None, gpu_identifier: str | None = None) -> str:
    """Asks a question about the data provided for exploratory data analysis (EDA)

    Args:
        question: The EDA question to answer
        description: Competition description
        data_path: Path to the data directory
        max_attempts: Maximum number of attempts (default from config)
        timeout_seconds: Timeout for code execution in seconds (default 1800 = 30 minutes)
        previous_ab_tests: List of previous AB test dicts with 'question' and 'code' keys (empty for EDA, last 8 for AB tests)
        file_suffix: Optional suffix for the temp file (e.g., "_1", "_2" for parallel execution)
        cpu_core_range: List of CPU cores to use (e.g., [0,1,2,...,41]) for CPU affinity
        gpu_identifier: GPU identifier: MIG UUID or GPU ID (as string) for GPU isolation
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

    # Build input_list with AB test history: prepend previous AB tests (question + code)
    input_list = []
    if previous_ab_tests:
        for ab_test in previous_ab_tests:
            input_list.append({"role": "user", "content": "Question: " + ab_test['question']})
            input_list.append({"role": "assistant", "content": f"```python\n{ab_test['code']}\n```"})

    # Add the current question
    input_list.append({"role": "user", "content": "Question: " + question})

    for attempt in range(1, attempts + 1):
        logger.info("ask_eda attempt %s/%s", attempt, attempts)
        response = call_llm_with_retry(
            model=_RESEARCHER_TOOL_OFFLINE_MODEL,
            instructions=PROMPT,
            tools=[],
            messages=input_list,
            web_search_enabled=True,
        )
        response_text = response.output_text or ""
        input_list += response.output

        code = extract_python_code(response_text)
        logger.debug("ask_eda generated code (truncated): %s", code[:200] if code else "")

        if not code:
            input_list.append({"role": "user", "content": "No python code block found. Please try again."})
            logger.warning("ask_eda found no python code block in response.")
            continue
        else:
            # Write code to temporary file with resource allocation and OpenBLAS thread limiting prefix
            filename = f"eda_temp{file_suffix}.py" if file_suffix else "eda_temp.py"
            code_file = Path(data_path) / filename

            # Build resource allocation header (similar to DeveloperAgent._postprocess_code)
            header_lines = []
            header_lines.append('import os')

            # CPU affinity
            if cpu_core_range is not None:
                header_lines.append('import psutil  # For CPU affinity')
                header_lines.append('')
                header_lines.append('# CPU affinity (pin to specific cores to prevent resource overlap)')
                header_lines.append(f'psutil.Process(os.getpid()).cpu_affinity({cpu_core_range})')

            # GPU assignment (works for both MIG and multi-GPU)
            if gpu_identifier is not None:
                gpu_device = gpu_identifier
            else:
                gpu_device = '0'  # Default to GPU 0

            header_lines.append(f'os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu_device}"')
            header_lines.append('os.environ["OPENBLAS_NUM_THREADS"] = "32"')
            header_lines.append('')

            header = '\n'.join(header_lines)

            with open(code_file, "w") as f:
                f.write(header + '\n' + code)

            # Import execute_code here to avoid circular import at module level
            from tools.developer import execute_code

            # Execute code using execute_code() function
            result = execute_code(str(code_file), timeout_seconds=timeout_seconds)

            # Check if execution timed out
            if "timed out" in result.lower():
                logger.error("ask_eda execution timed out")
                return result

            # Check if execution had errors (execute_code returns error traces with guidance)
            # If there's an error, it will contain stack trace (Traceback)
            if "Traceback" in result:
                logger.warning("ask_eda execution failed, retrying with error feedback")
                input_list.append({"role": "user", "content": result + "\n\nPlease regenerate the code to fix the errors above."})
                continue

            # Success - truncate to last 30000 characters and return
            logger.info("ask_eda succeeded on attempt %s", attempt)
            if len(result) > 30000:
                result = "... (output truncated to last 30000 characters)\n\n" + result[-30000:]
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
                text_format=DatasetDiscovery,
            )

            # Use structured output parsing
            if response and hasattr(response, 'output_parsed') and response.output_parsed:
                new_datasets = response.output_parsed.datasets
                logger.info("Query %s found %s datasets: %s", q_idx, len(new_datasets), new_datasets)
                relevant_datasets.extend(new_datasets)
                break  # Success, move to next query
            else:
                logger.warning("No valid structured output for query %s attempt %s.", q_idx, attempt)
                continue

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


@weave.op()
def read_research_paper(arxiv_link: str) -> str:
    """Read and summarize a research paper from arxiv.

    Args:
        arxiv_link: ArXiv paper link (e.g., "https://arxiv.org/pdf/2510.22916" or just "2510.22916")

    Returns:
        Structured markdown summary with sections: Abstract, Introduction, Related Work,
        Method/Architecture, Experiments/Results, and Conclusion.
    """
    # Extract arxiv ID from link (handles full URLs or just IDs)
    arxiv_id_match = re.search(r'(\d{4}\.\d{4,5})', arxiv_link)
    if arxiv_id_match:
        arxiv_id = arxiv_id_match.group(1)
    else:
        # Assume it's already just an ID
        arxiv_id = arxiv_link

    logger.info("Reading research paper with arxiv ID: %s", arxiv_id)

    try:
        client = GeminiPaperSummaryClient(is_model=False)
        summary = client.generate_summary(model_name=arxiv_id)
        logger.info("Successfully generated paper summary (length: %d chars)", len(summary))
        return summary
    except Exception as e:
        logger.exception("Failed to read research paper: %s", arxiv_id)
        return f"Error reading paper: {str(e)}"


@weave.op()
def scrape_web_page(url: str) -> str:
    """Scrape a web page and return LLM-ready markdown content.

    Useful for reading blog posts, documentation, competition forums, winner solutions,
    technical tutorials, and other domain-specific web content that complements arxiv papers.

    Args:
        url: The webpage URL to scrape (e.g., blog posts, Kaggle discussions, documentation)

    Returns:
        Markdown content from the page with title and metadata
    """
    logger.info("Scraping web page: %s", url)

    try:
        from firecrawl import Firecrawl

        api_key = os.environ.get("FIRECRAWL_API_KEY")
        if not api_key:
            logger.error("FIRECRAWL_API_KEY not set in environment")
            return "Error: FIRECRAWL_API_KEY environment variable is not set. Cannot scrape web pages."

        app = Firecrawl(api_key=api_key)
        doc = app.scrape(url, formats=["markdown"])

        if doc.markdown:
            title = doc.metadata.title if doc.metadata and doc.metadata.title else url
            content = doc.markdown[:30000]  # Limit to first 30000 characters
            truncated = len(doc.markdown) > 30000
            logger.info("Successfully scraped page (length: %d chars, truncated: %s, title: %s)",
                       len(doc.markdown), truncated, title)

            result = f"# {title}\n\nSource: {url}\n\n{content}"
            if truncated:
                result += f"\n\n... (content truncated at 30000 characters, original length: {len(doc.markdown)} chars)"
            return result
        else:
            logger.warning("No markdown content returned from Firecrawl for URL: %s", url)
            return f"Error: Failed to scrape page at {url}. The page may be inaccessible or contain no readable content."

    except ImportError:
        logger.error("Firecrawl package not installed")
        return "Error: Firecrawl package is not installed. Install with: pip install firecrawl-py"
    except Exception as e:
        logger.exception("Failed to scrape web page: %s", url)
        return f"Error scraping page: {str(e)}"


def get_tools(max_parallel_workers: int = 1):
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
            "description": f"Run A/B tests to validate modeling or feature engineering choices by comparing their impact on performance. You can ask up to {max_parallel_workers} questions in parallel for efficiency.",
            "parameters": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "description": f"List of A/B testing questions to run in parallel (max {max_parallel_workers}). Each question should be a comparison test",
                        "items": {"type": "string"},
                    }
                },
            },
            "additionalProperties": False,
            "required": ['questions']
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
        {
            "type": "function",
            "name": "read_research_paper",
            "description": "Read and summarize a research paper from arxiv. Returns structured markdown summary with Abstract, Introduction, Related Work, Method/Architecture, Experiments/Results, and Conclusion sections.",
            "parameters": {
                "type": "object",
                "properties": {
                    "arxiv_link": {"type": "string", "description": "ArXiv paper link (e.g., 'https://arxiv.org/pdf/2510.22916' or just '2510.22916')"}
                },
            },
            "additionalProperties": False,
            "required": ["arxiv_link"],
        },
        {
            "type": "function",
            "name": "scrape_web_page",
            "description": "Scrape a web page and return markdown content. Useful for reading blog posts, documentation, technical tutorials, and other domain-specific web content that complements arxiv papers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The webpage URL to scrape (e.g., 'https://developer.nvidia.com/blog/...')"}
                },
            },
            "additionalProperties": False,
            "required": ["url"],
        },
    ]
