"""Utility helpers for executing generated code with rich logging."""

import logging
import os
import subprocess
import traceback

from dotenv import load_dotenv
from openai import OpenAI
import pickle

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
        return trace + "\n"


if __name__ == "__main__":
    logger.info("Running tools.developer as a script for manual execution test.")
    output = execute_code("code.py")
    logger.info("Execution output: %s", output)
