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
def search_sota_suggestions(
    description: str,
    context: str,
    executed_suggestion: str | None,
    failed_to_improve_score: bool,
    failed_ideas: list[str],
    best_code: str | None = None,
    executed_code: str | None = None,
) -> str:
    """Use web search to surface potential SOTA improvements for the competition."""
    logger.info("Dispatching SOTA web search")
    failed_ideas_text = "No prior ideas are blacklisted."
    executed_suggestion_text = executed_suggestion or "No previous suggestion executed; this is the first attempt."
    if best_code:
        best_code_text = "Best-performing code accompanies this prompt; inspect the context payload above."
    else:
        best_code_text = "Best-performing code snippet not available yet."
    executed_code_text = executed_code or "No explicit code snippet was provided for the last attempt."
    if failed_ideas:
        filtered = [idea for idea in failed_ideas if idea][-10:]
        if filtered:
            failed_ideas_text = "\n".join(f"- {idea}" for idea in filtered)

    prompt = f"""
You will receive a Kaggle competition description, an initial script, and its logs for analysis.

<competition description>
{description}
</competition description>

<previous failed ideas> DO NOT TRY THESE AGAIN
{failed_ideas_text}
</previous failed ideas>

<previous suggestion executed>
{executed_suggestion_text}
</previous suggestion executed>

<previous code snippet applied>
{executed_code_text}
</previous code snippet applied>

{context}

Outcome status: {"No improvement" if failed_to_improve_score else "Improved or matched"}

<best code reference>
{best_code_text}
</best code reference>

### Checklist
- Begin with a concise checklist of 3-7 bullet points summarizing high-level conceptual red flags from the code/logs and your intended strategies to address them. These should be conceptual (not implementation-specific). Use "- " for each bullet.
- Checklist: If fewer than three meaningful points arise, include as many as possible and explicitly state: "Fewer than 3 high-level red flags or strategies identified."

### Research and Suggestion
- Before any web search or external query, briefly state the purpose and the minimal search terms you will use.
- Perform a web search for recent, effective models, architectures, techniques or hyperparameters relevant to this competition or similar tasks, addressing the identified red flags.
- Before recommending any approach, clearly state its purpose and justification, referencing the competition description and context.
- Propose one high-impact suggestion to improve performance for the competition's metric, along with an approximately 100-word explanation of its benefits.

### Validation
- After offering your suggestion, validate its relevance to the competition details and metric in 1-2 sentences.
- Clearly specify the validation criteria and reference key input details where possible.
- If you cannot validate because of missing or insufficient input, clearly state so and return "No suggestions."

### Error Handling
- If the <competition description> or <initial script and logs> are missing or clearly inadequate, mention this before the checklist.
- In such cases, display only the section headings (as markdown headings) with "No suggestions." under each, and in the JSON summary output an empty string as shown:

### Output Format
Structure your output as follows:

### Checklist
- ... (3-7 conceptual bullet points; see above)

### Research and Suggestion
- ... (prose explanation)

### Validation
- ... (validation statements, or "No suggestions.")

### Previous Suggestion Review
Decide whether the most recently executed suggestion (see <previous suggestion executed>) should be blacklisted. Base your decision on the validation outcomes and logs provided in the context.

Output your decision in the following strict JSON format, enclosed in backticks:
```json
{{
    "blacklist": <true or false>,
    "reason": "<succinct justification; use empty string if blacklist is false>"
}}
```

### New Suggestion Summary
Propose the single best next idea for improving the competition score. Do not repeat blacklisted ideas or the previous suggestion.

Output your new idea in the following strict JSON format, enclosed in backticks:
```json
{{
    "suggestion": "<your suggestion here>"
}}
```
If you have no viable suggestion, leave the value as an empty string.

### Code
Provide a concise Python snippet (enclosed in ```python backticks) that implements the suggested change.

- Never repeat any idea from <previous failed ideas>.
- If blacklist is true, ensure the new suggestion avoids that approach.
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
