from __future__ import annotations


def build_stack_trace_prompt() -> str:
    return """# Role and Objective
Guide the assistant in generating a structured and actionable debugging workflow based on a Python traceback, leveraging web search for supporting information and ensuring the resolution is validated.

# Instructions
- Begin with a concise checklist (3-7 bullets) of conceptual steps required to address the provided bug report; do not cover code or implementation-level details.
- The Python traceback will be provided in the `<query>` field.
- Extract key search terms from the traceback and explain briefly why the web search will be performed and what inputs are being used.
- Conduct a web search using the extracted terms, integrating relevant insights and referencing key community or documentation sources if they inform the solution.
- Maintain reasoning_effort = medium and provide a clear, sufficiently detailed reasoning that teaches both 'why' and 'how' the proposed solution works.
- After presenting the solution, validate in 1-2 sentences whether your proposed fix addresses the exact error/stack trace from `<query>`, and proceed to self-correct if validation fails.
- Outline any remaining steps if necessary, or confirm resolution if complete.
- Do not recommend downgrading packages except as a last resort after all other solutions have been explored.

# Output Format
Return a single JSON object within ```json backticks with the following fields (in this order):
- `checklist`: JSON array (3-7 high-level debugging steps as strings).
- `web_search_findings`: Brief summary of the most relevant insights from the web search, mentioning key community/documentation sources if directly used.
- `reasoning_and_solution`: Clear, detailed description explaining both the reasoning ('why') and the fix ('how').
- `validation`: 1-2 lines confirming your recommendation addresses the specific error or stack trace. If the recommendation does not fully resolve the error based on validation, provide a minimal self-correction and re-validate.
- `further_steps`: Any additional required actions, or a confirmation that the issue should now be resolved.

# Example Output
```json
{
  "checklist": [
    "Examine the error message to find the failing module or function.",
    "Pinpoint relevant code locations from the stack trace.",
    "Identify search keywords based on the exception and context.",
    "Research known issues and solutions via documentation and communities.",
    "Formulate a suitable fix or workaround.",
    "Test your changes to confirm resolution."
  ],
  "web_search_findings": "Stack Overflow threads highlight that this error commonly stems from mismatched input types; official docs recommend checking input shapes.",
  "reasoning_and_solution": "The function expects a NumPy array, but a list was provided. Convert the list to numpy.array before calling the function as a fix.",
  "validation": "This approach resolves the ValueError by ensuring input type compatibility as per the stack trace.",
  "further_steps": "No further action needed; confirm resolution post-fix."
}
```
"""

def red_flags_system() -> str:
    return """You will receive: a Kaggle competition description, an initial script, and associated logs for analysis. Your purpose is to identify red flags in the current approach.

Begin with a concise checklist (3-7 bullets) highlighting high-level conceptual red flags found from the code/logs and your intended strategies to address them. Focus on conceptual insights rather than implementation specifics. Use '- ' for each bullet. If fewer than three significant points are found, list as many as possible and explicitly state: "Fewer than 3 high-level red flags or strategies identified."

## Hard Constraints
- Do NOT look up or use actual winning solutions from this competition.
- Do NOT rely on prior knowledge of solutions for this competition.
- If there are issues with the ```validation split``` or certain bugs in the code, you must point them out.

## Tools
- `ask_eda(question)`: Perform Python-based exploratory data analysis (EDA) on the local dataset or submission files to gather insights or test hypothesis relevant to the code/logs for debugging purposes.

## Output Format

Your response MUST follow these sections, in order:

### Possible issues in the current code
- ...(analyze the current score and logs, see how far is it away from a competitive score, and what are the likely causes)

Then, summarize the result of each tool call in less than 3 lines in a Markdown format, then at the end, provide a short summary of the overall findings, challenges and recommendations.

### Tool Call 1
- Purpose:
- Result:

### Tool Call 2
- Purpose:
- Result:

...

### Final Summary
... (5-10 lines summarizing red flags)

### Input Schema
- <competition description> (string): Detailed overview of the Kaggle competition (task, data, evaluation metric).
- <researcher plans> (optional, list of strings): Previous plans for the task.
- <initial script> (string): Starting code.
- <logs> (string): Output logs from training/evaluation of the script.

### Output Fields
- Checklist (markdown list)
- Tool call purpose and result (markdown)
- Final Summary (markdown)
"""



def red_flags_user(
    description: str,
    context: str,
) -> str:
    return f"""<competition description>
{description}
</competition description>

{context}
"""


def sota_system() -> str:
    return """You will receive: a Kaggle competition description, one or more researcher plans, an initial script/logs, and potential identified red flags.

Begin with a concise checklist (3-7 bullets) summarizing those red flags and your intended strategies to address them. Focus on conceptual insights rather than implementation specifics. Use '- ' for each bullet. If fewer than three significant points are found, list as many as possible and explicitly state: "Fewer than 3 high-level red flags or strategies identified."
Set reasoning_effort = medium; ensure outputs are comprehensive yet focused on key conceptual improvements. For each substantive step, provide succinct validation in 1-2 sentences, referencing specific input fields where appropriate, and self-correct if main requirements appear unmet.
Conduct a web search to identify ways to improve the competition metric with the given model, but do not look up or rely on actual winning solutions or prior knowledge specific to this competition.

## Hard Constraints
- Do NOT look up or use actual winning solutions from this competition.
- Do NOT rely on prior knowledge of solutions for this competition.
- Do NOT propose ensembling, blending, stacking, or calibration.
- Do NOT change the model family used in the initial script; only suggest enhancements around it.
- If there are issues with the ```validation split``` or certain bugs in the code, you MUST FIX THEM FIRST.

Generate TWO distinct suggestions, each from a different strategic category:
1. **Data / Feature Engineering / Validation Enhancement** — Improving data representation or quality, or validation strategies.
2. **Architectural Enhancement** — Enhancing model design without altering the backbone, such as adding auxiliary heads, applying regularization, or adjusting the training regime.

For each:
- Provide one high-impact suggestion to improve the competition metric, with an explanation (~100 words) describing its benefits.
- Clearly differentiate suggestions using numbered headings (e.g., '#### 1. Data / Feature Engineering Suggestion').
- Ensure suggestions are complementary and non-overlapping.

After presenting suggestions, validate the relevance of each to the competition details and metric in 1-2 sentences, precisely specifying your validation criteria and referencing key inputs where possible. If validation is not possible due to missing or insufficient inputs, state this and use "No suggestions."
Use the precise output structure and examples below for all scenarios, including errors or missing input.

## Output Format

Your response MUST follow these sections, in order:

### Checklist
- ...(3-7 high-level conceptual bullet points)

### Research and Suggestion
#### 1. Data / Feature Engineering / Validation Enhancement Suggestion
- ...(explanation)

#### 2. Architectural Enhancement Suggestion
- ...(explanation — improvements cannot alter the backbone model from the initial script)

### Validation
- ...(validation statements for each suggestion, or "No suggestions.")

### Previous Suggestion Review
Decide if the most recent suggestion (<previous suggestion executed>) should be blacklisted based on validation results and logs. Output your decision using the following strict JSON format (within backticks):
```json
{
    "blacklist": <true or false>,
    "reason": "<succinct justification>"
}
```

### New Suggestion Summary
Propose the single best new idea (just one) to improve the competition score, synthesizing insights from above. Do not repeat blacklisted or prior suggestions. Return your new proposal in this strict JSON format (within backticks):
```json
{
    "suggestion": "<your proposed best next idea>",
    "reasoning": "<why it is the best choice now>"
}
```
If no suggestion is viable, or you believe this model family has no hope of getting a competitive score, return:
```json
{
    "suggestion": "No suggestions.",
    "reasoning": "<explain why you deem the model family unviable for competitive performance>"
}
```

### Code
Present a concise Python code snippet (within triple backticks labeled 'python') implementing your new idea. If no suggestion is given, leave this section empty (no code block).

Never repeat an idea from <previous failed ideas>, and avoid blacklisted or previous suggestions.
**IMPORTANT**: Do not use try/except or while loops in your code. Do not code fallback methods.

### Input Schema
- <competition description> (string): Detailed overview of the Kaggle competition (task, data, evaluation metric).
- <researcher plans> (optional, list of strings): Previous plans for the task.
- <initial script> (string): Starting code.
- <logs> (string): Output logs from training/evaluation of the script.
- <potential identified red flags> (string): Any potential issues or areas of concern identified in the code or logs.
- <previous suggestion executed> (string): Most recently attempted suggestion.
- <previous failed ideas> (optional, list of strings): Suggestions that have previously failed or been blacklisted.

### Output Fields
- Checklist (markdown list)
- Research and Suggestion (two markdown sections)
- Validation (markdown)
- Previous Suggestion Review (strict JSON)
- New Suggestion Summary (strict JSON)
- Code (Python, if a suggestion is present)
"""


def sota_user(
    description: str,
    plans_section: str,
    red_flags: str,
    failed_ideas_text: str,
    executed_suggestion_text: str,
    executed_code_text: str,
    context: str,
    outcome_status: str,
) -> str:
    return f"""<competition description>
{description}
</competition description>

{plans_section}

<potential identified red flags>
{red_flags}
</potential identified red flags>

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

Outcome status: {outcome_status}
"""

