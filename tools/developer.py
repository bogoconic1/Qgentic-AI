"""Utility helpers for executing generated code with rich logging."""

import logging
import os
import pickle
import subprocess
import textwrap
import traceback

from dotenv import load_dotenv
from openai import OpenAI
from tools.helpers import call_llm_with_retry
import weave

load_dotenv()

# Configure logging once at import. Downstream callers can override if needed.
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


client = OpenAI(api_key=os.environ.get("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")

@weave.op()
def web_search_stack_trace(query: str) -> str:
    """Research how to fix a bug based on the stack trace and error message."""
    logger.info("Dispatching web search for stack trace remediation guidance.")
    logger.debug("Stack trace query: %s", query)

    messages = [
        {
            "role": "user",
            "content": f"""Begin with a concise checklist (3-7 bullets) of what you will do; keep items conceptual, not implementation-level.

I am currently encountering the following bug:

{query}

After proposing a solution, validate that the recommendation directly addresses the stack trace and describe any needed next steps or confirm if the issue should be resolved. How can I resolve this issue?
            """,
        }
    ]
    logger.debug("Web search messages: %s", messages)

    try:
        content = ""
        while content == "":
            completion = call_llm_with_retry(
                client,
                model="openai/gpt-5:online",
                messages=messages,
            )
            msg = completion.choices[0].message
            content = msg.content or ""
            logger.debug("Web search raw response: %s", completion)
        logger.info("Received web search response for stack trace query.")
        logger.debug("Web search raw response: %s", content)
    except Exception:
        logger.exception("Web search request for stack trace failed.")
        return "Please try to fix the bug yourself."

    return content

@weave.op()
def search_sota_suggestions(description: str, context: str, failed_ideas: str, use_sota: bool) -> str:
    """Use web search to surface potential SOTA improvements for the competition."""
    logger.info("Dispatching SOTA web search")
    prompt = f"""You are provided with a Kaggle competition description and an initial script for the task.

<competition description>
{description}
</competition description>

<initial script and logs>
{context}
</initial script and logs>

<previous failed ideas> DO NOT TRY THESE AGAIN ❌  ❌  ❌ 
{failed_ideas}
</previous failed ideas>

Begin with a concise checklist (3-7 bullets) of the possible red flags in the logs and what you will do; keep items conceptual, not implementation-level.
Your task is to provide a single, impactful suggestion — along with sample code — to improve the model's performance with respect to the competition metric. In approximately 100 words, explain why your suggestion would help.
If you have no suggestions, simply reply with "No suggestions."
After proposing your suggestion and code, briefly validate its relevance to the competition details and metric in 1-2 lines. If your suggestion cannot be validated, state why and consider whether to proceed or reply "No suggestions."
Carefully consider the competition details and context to deliver the most impactful recommendation possible.

**IMPORTANT**: Do not repeat any ideas listed in the <previous failed ideas> section.
**IMPORTANT**: Do not suggest scaling up the number of folds unless you have no other ideas."""
    if use_sota:
        prompt += "\n**IMPORTANT**: Please do some research on SOTA techniques that can be applied to this competition."
        
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    try:
        content = ""
        while content == "":
            completion = call_llm_with_retry(
                client,
                model="openai/gpt-5:online",
                messages=messages,
            )
            msg = completion.choices[0].message
            logger.debug("SOTA search raw response: %s", completion)
            content = msg.content or ""
        logger.debug("SOTA search raw response: %s", content)

        return content

    except Exception:
        logger.exception("SOTA suggestions web search failed")
        return ""

@weave.op()
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
