"""Utility helpers for executing generated code with rich logging."""

import logging
import os
import subprocess
import traceback

from dotenv import load_dotenv
from openai import OpenAI
from project_config import get_config
from tools.helpers import call_llm_with_retry
import weave

load_dotenv()

# Configure logging once at import. Downstream callers can override if needed.
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


_CONFIG = get_config()
_LLM_CFG = _CONFIG.get("llm", {}) if isinstance(_CONFIG, dict) else {}
_BASE_URL = _LLM_CFG.get("base_url", "https://openrouter.ai/api/v1")
_API_KEY_ENV = _LLM_CFG.get("api_key_env", "OPENROUTER_API_KEY")
_ONLINE_MODEL = _LLM_CFG.get("online_model", "openai/gpt-5:online")

client = OpenAI(api_key=os.environ.get(_API_KEY_ENV), base_url=_BASE_URL)

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
                model=_ONLINE_MODEL,
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
def search_sota_suggestions(description: str, context: str, failed_ideas: str) -> str:
    """Use web search to surface potential SOTA improvements for the competition."""
    logger.info("Dispatching SOTA web search")
    prompt = f"""
You will receive a Kaggle competition description along with an initial script and its logs.

<competition description>
{description}
</competition description>

<initial script and logs>
{context}
</initial script and logs>

<previous failed ideas> DO NOT TRY THESE AGAIN 
{failed_ideas}
</previous failed ideas>

- Begin with a concise checklist (3-7 bullet points) summarizing high-level conceptual red flags found in the logs and your intended overall strategies for addressing them. Keep bullets conceptual, not implementation-specific. Use "- " to denote each bullet.

- Web search recent effective models, architectures, or techniques relevant to this competition that could address the red flags. Before suggesting an approach, explicitly state its purpose and why you have selected it, referencing information from the competition description and context. Propose a single, high-impact suggestion to improve performance for the competition's metric. Whenever feasible, include a sample Python code block (unless otherwise indicated in the context) between triple backticks (```), and accompany it with an approximately 100-word explanation of why this approach is beneficial.

- After presenting your suggestion and code, validate its relevance to the specific competition details and metric in 1-2 sentences. State clear criteria for validation, referencing key details from the input where possible. If validation is not possible because the input is insufficient, mention this clearly and return "No suggestions.".

- If the <competition description> or <initial script and logs> are missing or clearly inadequate, state this before the checklist. Then, output only the section headings with "No suggestions." and a summary JSON as described below.

- At the end, provide a single-line summary of your recommendation using the following strict JSON format within backticks (if no suggestion, use an empty string for the field):
```json
{{
    "suggestion": "<your suggestion here>"
}}
```

- Never repeat any ideas from <previous failed ideas>.

## Output Format
Your output must follow these sections, strictly in this order:

1. **Checklist**: 3-7 bullet points on conceptual red flags and strategies.
2. **Research & Suggestion**:
    - Name and briefly describe the recent effective model or technique.
    - Offer one high-impact improvement suggestion with purpose stated upfront.
    - Include a code block (default Python; adapt if another language is evident in context).
    - Provide a concise explanation (~100 words).
3. **Validation**: 1-2 sentence relevance check against the competition description and metric, with explicit validation criteria.
4. **JSON Summary**: One-line summary in the specified JSON format; use an empty string if no suggestion.

If no actionable suggestion is possible, provide all section headings with "No suggestions." in the appropriate places and ensure the JSON uses an empty string.

Always comply with required section order and formatting, clearly handle missing input cases, and ensure never to repeat any idea from <previous failed ideas>.
"""
        
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
                model=_ONLINE_MODEL,
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