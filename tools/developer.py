"""Utility helpers for executing generated code with rich logging."""

import json
import logging
import os
import re
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
_DEVELOPER_MODEL = _LLM_CFG.get("developer_model", "google/gemini-2.5-pro")
_DEVELOPER_TOOL_MODEL = _LLM_CFG.get("developer_tool_model", _DEVELOPER_MODEL)

client = OpenAI(api_key=os.environ.get(_API_KEY_ENV), base_url=_BASE_URL)

@weave.op()
def web_search_stack_trace(query: str) -> str:
    """Research how to fix a bug based on the stack trace and error message."""
    logger.info("Dispatching web search for stack trace remediation guidance.")
    logger.debug("Stack trace query: %s", query)

    messages = [
        {
            "role": "user",
            "content": f"""Begin by generating a concise checklist (3-7 items) of the high-level conceptual actions you will take to address the bug report. Present this checklist as a JSON array of strings, focusing on core debugging steps rather than low-level implementation details.
You will receive a bug description in the "query" field. This field contains a plain text description of the encountered bug, which may include a stack trace or error message.
When a stack trace or error message is present in the query, perform a web search using relevant keywords extracted from the stack trace or error message. Incorporate findings from the web search into your reasoning and solution, citing any particularly useful information or community guidance that directly informs your debugging approach or proposed fix.

{{
"query": "{query}"
}}

Before composing your solution, set reasoning_effort = medium and ensure that your explanation is sufficiently detailed to guide a developer through both the 'why' and the 'how' of the proposed fix. After presenting the solution, validate in 1-2 lines that your recommendation specifically addresses any stack trace or error message found in the query. Clearly outline additional steps if required, or confirm that your resolution is sufficient and that the issue should now be considered resolved.
Do not suggest downgrading packages unless absolutely necessary, and only after exploring all other avenues.

{{
"checklist": [
"Conceptual step #1",
"Conceptual step #2",
"..."
],
"solution": "Detailed explanation of steps to resolve the issue, with clear reasoning to help a developer understand both the why and the how. Incorporate any relevant insights or advice acquired from the web search. Avoid overly terse answers.",
"validation": "Explicit analysis demonstrating how your solution resolves the bug or handles all aspects of the stack trace, plus any further action or confirmation of completion. Reference web search findings if they directly impact the resolution."
}}

""",
        }
    ]
    logger.debug("Web search messages: %s", messages)

    try:
        content = ""
        while content == "":
            completion = call_llm_with_retry(
                client,
                model=_DEVELOPER_TOOL_MODEL,
                messages=messages,
            )
            try:
                msg = completion.choices[0].message
                content = msg.content or ""
            except Exception:
                msg = ""
                content = ""
            logger.debug("Web search raw response: %s", completion)
        logger.info("Received web search response for stack trace query.")
        logger.debug("Web search raw response: %s", content)
    except Exception:
        logger.exception("Web search request for stack trace failed.")
        return "Please try to fix the bug yourself."

    solution_text = content
    parsed_payload = None
    try:
        parsed_payload = json.loads(content)
    except json.JSONDecodeError:
        logger.debug("Raw response is not bare JSON; attempting fenced-block extraction.")
        try:
            match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL | re.IGNORECASE)
            if match:
                parsed_payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            logger.debug("Failed to parse JSON from fenced block in web search response.")

    if isinstance(parsed_payload, dict):
        solution_candidate = parsed_payload.get("solution")
        if isinstance(solution_candidate, str) and solution_candidate.strip():
            logger.debug("Returning solution field extracted from web search response.")
            return solution_candidate.strip()
        logger.debug("Solution field missing or empty in parsed payload.")

    return solution_text

@weave.op()
def search_sota_suggestions(
    description: str,
    context: str,
    executed_suggestion: str | None,
    failed_to_improve_score: bool,
    failed_ideas: list[str],
    executed_code: str | None = None,
    plans: list[str] | None = None,
) -> str:
    """Use web search to surface potential SOTA improvements for the competition."""
    logger.info("Dispatching SOTA web search")
    failed_ideas_text = "No prior ideas are blacklisted."
    executed_suggestion_text = executed_suggestion or "No previous suggestion executed; this is the first attempt."
    executed_code_text = executed_code or "No explicit code snippet was provided for the last attempt."
    if failed_ideas:
        filtered = [idea for idea in failed_ideas if idea][-10:]
        if filtered:
            failed_ideas_text = "\n".join(f"- {idea}" for idea in filtered)

    # Optional: include researcher plans as a separate section for plan-aware suggestions
    plans_section = ""
    if plans:
        blocks: list[str] = []
        for idx, p in enumerate(plans, start=1):
            # Truncate to avoid excessive token usage
            text = p if len(p) <= 30000 else p[-30000:]
            blocks.append(f"<plan id=\"{idx}\">\n{text}\n</plan>")
        plans_section = "\n<researcher_plans>\n" + "\n\n".join(blocks) + "\n</researcher_plans>\n"

    system_prompt = f"""Developer: You will receive a Kaggle competition description, one or more researcher plans, an initial script, and its logs for analysis.

Begin with a concise checklist (3-7 bullets) summarizing high-level conceptual red flags identified from the code/logs, as well as your intended strategies for addressing them. These should focus on conceptual aspects rather than specific implementations. Use '- ' for each bullet. If fewer than three meaningful points arise, list as many as possible and explicitly state: "Fewer than 3 high-level red flags or strategies identified."

Always review the <researcher_plans> first, if provided. Set reasoning_effort = medium to ensure thoughtful but efficient analysis. For any web search or external query, briefly state the purpose and the minimal search terms you will use before proceeding. Use only approved resources and provide a one-line preamble before significant information-sourcing steps, referencing the competition context.

Conduct a web search for recent, effective models, architectures, techniques, or hyperparameters relevant to the competition or similar tasks, directly addressing the outlined red flags. Clearly explain the purpose of every recommended approach and justify its relevance by referencing the competition description and context.

Generate FOUR distinct suggestions, each from a separate strategic category:
1. Data / Feature Engineering Enhancement - focuses on improving input representation or data quality.
2. Architectural Enhancement - proposes improvements to the model backbone or design.
3. Ensembling Enhancement - addresses model aggregation, stacking, blending, or bagging.
4. SOTA Model Enhancement - highlights a recent effective model from an arXiv paper/GitHub repository/Blog Post that has been successfully applied on similar tasks.

For each category:
- Provide one high-impact suggestion for improving performance on the competition task and metric, with an explanation of approximately 100 words outlining its benefits.
- Clearly differentiate suggestions using numbered headings (e.g., '#### 1. Data / Feature Engineering Suggestion').
- Ensure the four suggestions are complementary and not overlapping.

After presenting suggestions, validate each one's relevance to the competition details and metric in 1-2 sentences. Clearly specify the criteria used for validation and reference key input details when possible. If validation cannot be performed due to missing or inadequate inputs, clearly state this and return "No suggestions."

If the <competition description> or <initial script and logs> are missing or clearly inadequate, mention this before providing the checklist and return "No suggestions."

After code edits or substantial analysis, validate the intended outcome or impact in 1-2 sentences and self-correct if validation fails or key requirements are unmet.

Follow the exact output structure and examples outlined below for all scenarios, including missing input or error conditions:

## Output Format
Your output MUST include the following sections in order:

### Checklist
- ... (3-7 high-level conceptual bullet points, see above)

### Research and Suggestion
#### 1. Data / Feature Engineering Suggestion
- ... (prose explanation)

#### 2. Architectural Enhancement Suggestion
- ... (prose explanation)

#### 3. Ensembling/Blending Enhancement Suggestion
- ... (prose explanation)

#### 4. SOTA Model Enhancement
- ... (prose explanation)

### Validation
- ... (validation statements for each suggestion, or "No suggestions.")

### Previous Suggestion Review
Determine whether the most recent suggestion (see <previous suggestion executed>) should be blacklisted, based on validation outcomes and logs provided in context.

Output your decision in the following strict JSON format (enclosed in backticks):
```json
{{
    "blacklist": <true or false>,
    "reason": "<succinct justification; if blacklist is false, use empty string>"
}}
```

### New Suggestion Summary
Propose the single best next idea (only one) for improving the competition score, synthesizing insights from the four categories above. Do not repeat blacklisted ideas or the previous suggestion.

Return your new idea using the following strict JSON format (enclosed in backticks):
```json
{{
    "suggestion": "<your proposed best next idea>",
    "reasoning": "<explanation for selecting this idea as the best compared to other promising ideas>"
}}
```
If there is no viable suggestion, use empty strings for the values.

### Code
Present a concise Python code snippet (within triple backticks marked 'python') that implements your proposed best next idea. If no suggestion is made, leave this section empty (no code block).

Never repeat any idea from <previous failed ideas>. If a suggestion is blacklisted, ensure your new recommendation avoids that approach.
"""

    prompt = f"""<competition description>
{description}
</competition description>

{plans_section}

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
"""
        
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    try:
        content = ""
        while content == "":
            completion = call_llm_with_retry(
                client,
                model=_DEVELOPER_TOOL_MODEL,
                messages=messages,
            )
            try:
                msg = completion.choices[0].message
                logger.debug("SOTA search raw response: %s", completion)
                content = msg.content or ""
            except Exception:
                msg = ""
                content = ""
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
