"""Utility helpers for executing generated code with rich logging."""

import logging
import os
import pickle
import subprocess
import textwrap
import traceback

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Configure logging once at import. Downstream callers can override if needed.
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def web_search_stack_trace(query: str) -> str:
    """Research how to fix a bug based on the stack trace and error message."""
    logger.info("Dispatching web search for stack trace remediation guidance.")
    logger.debug("Stack trace query: %s", query)

    messages = [
        {
            "role": "user",
            "content": f"I am currently facing this bug: {query}. How do I fix it?",
        }
    ]
    logger.debug("Web search messages: %s", messages)

    try:
        response = client.responses.create(
            model="gpt-5",
            input=messages,
            tools=[{"type": "web_search"}],
        )
        with open("web_search_response.pkl", "wb") as f:
            pickle.dump(response, f)
        logger.info("Received web search response for stack trace query.")
        logger.debug("Web search raw response: %s", response)
    except Exception:
        logger.exception("Web search request for stack trace failed.")
        raise

    return response.output[-1].content[0].text


def search_sota_suggestions(description: str, context: str) -> str:
    """Use web search to surface potential SOTA improvements for the competition."""
    logger.info("Dispatching SOTA web search")
    messages = [
        {
            "role": "user",
            "content": f"""You are assisting with improvements for a Kaggle competition.
Competition Description: 
{description}

Refer to the experiment code and notes below and suggest up to three state-of-the-art models or architectures that could meaningfully improve the score.
If you are listing a huggingface model, include the model in the format <author>/<model>.

Experiment code and notes:
{context}

Output it in valid YAML format as:
```yaml
models:
    - <author>/<model>
    - <author>/<model>
    - <author>/<model>

techniques:
    - <technique 1>
    - <technique 2>
    - <technique 3>
```
If you cannot find any relevant suggestions, return an empty list for both sections.
"""
        }
    ]

    logger.debug("SOTA search messages: %s", messages)

    try:
        response = client.responses.create(
            model="gpt-5",
            input=messages,
            tools=[{"type": "web_search"}],
        )
        logger.debug("SOTA search raw response: %s", response)

        raw_output = response.output[-1].content[0].text
        # parse yaml
        if "```yaml" in raw_output:
            yaml_content = raw_output.split("```yaml")[1].split("```")[0]
        else:
            yaml_content = raw_output
        logger.debug("SOTA search YAML content: %s", yaml_content)
        return yaml_content
    except Exception:
        logger.exception("SOTA suggestions web search failed")
        return ""



def execute_code(filepath: str) -> str:
    """Execute a generated Python file and enrich errors with search guidance."""
    logger.info("Executing generated script: %s", filepath)
    try:
        logger.debug("Running subprocess command: python %s", filepath)
        result = subprocess.run(
            ["python", filepath],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            logger.info("Execution succeeded for %s", filepath)
            logger.debug("Execution stdout: %s", result.stdout)
            return result.stdout

        trace = result.stderr
        logger.warning(
            "Execution failed for %s with return code %s", filepath, result.returncode
        )
        logger.debug("Execution stderr: %s", trace)
        search_result = web_search_stack_trace(trace)
        return trace + "\n" + search_result

    except Exception:
        trace = traceback.format_exc()
        logger.exception("Unexpected error while executing %s", filepath)
        search_result = web_search_stack_trace(trace)
        return trace + "\n" + search_result


if __name__ == "__main__":
    logger.info("Running tools.developer as a script for manual execution test.")
    output = execute_code("code.py")
    logger.info("Execution output: %s", output)
