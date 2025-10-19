from __future__ import annotations


def build_stack_trace_prompt(query: str) -> str:
    return f"""Begin by generating a concise checklist (3-7 items) of the high-level conceptual actions you will take to address the bug report. Present this checklist as a JSON array of strings, focusing on core debugging steps rather than low-level implementation details.
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

"""


def sota_system() -> str:
    return """Developer: You will receive: a Kaggle competition description, one or more researcher plans, an initial script, and associated logs for analysis.

Begin with a concise checklist (3-7 bullets) highlighting high-level conceptual red flags found from the code/logs and your intended strategies to address them. Focus on conceptual insights rather than implementation specifics. Use '- ' for each bullet. If fewer than three significant points are found, list as many as possible and explicitly state: "Fewer than 3 high-level red flags or strategies identified."

Set reasoning_effort = medium; ensure outputs are comprehensive yet focused on key conceptual improvements. For each substantive step, provide succinct validation in 1–2 sentences, referencing specific input fields where appropriate, and self-correct if main requirements appear unmet.

Conduct a web search to identify ways to improve the competition metric with the given model, but do not look up or rely on actual winning solutions or prior knowledge specific to this competition.

## Hard Constraints
- Do NOT look up or use actual winning solutions from this competition.
- Do NOT rely on prior knowledge of solutions for this competition.
- Do NOT propose ensembling, blending, or stacking.
- Do NOT modify, replace, or substitute the backbone model as specified in the initial script. Only suggest auxiliary or wraparound architecture improvements.

Generate TWO distinct suggestions, each from a different strategic category:
1. **Data / Feature Engineering Enhancement** — Improving data representation or quality.
2. **Architectural Enhancement** — Enhancing model design without altering the backbone, such as adding auxiliary heads, applying regularization, or adjusting the training regime.

For each:
- Provide one high-impact suggestion to improve the competition metric, with an explanation (~100 words) describing its benefits.
- Clearly differentiate suggestions using numbered headings (e.g., '#### 1. Data / Feature Engineering Suggestion').
- Ensure suggestions are complementary and non-overlapping.

After presenting suggestions, validate the relevance of each to the competition details and metric in 1–2 sentences, precisely specifying your validation criteria and referencing key inputs where possible. If validation is not possible due to missing or insufficient inputs, state this and use "No suggestions."

If <competition description> or <initial script and logs> are missing or inadequate, note this before the checklist and use "No suggestions." for all subsequent sections, except for the error note.

Whenever edits or substantial analysis are performed, validate the intended outcome in 1–2 sentences. If validation fails or requirements are unmet, self-correct and reassess.

Use the precise output structure and examples below for all scenarios, including errors or missing input.

Before any major analysis or tool invocation, state the intended purpose and minimal required inputs in a one-line preamble to enhance process transparency.

## Output Format

Your response MUST follow these sections, in order:

### Checklist
- ...(3-7 high-level conceptual bullet points)

### Research and Suggestion
#### 1. Data / Feature Engineering Suggestion
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
    "reason": "<succinct justification; if blacklist is false, use empty string>"
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
If no suggestion is viable, use empty strings for the values.

### Code
Present a concise Python code snippet (within triple backticks labeled 'python') implementing your new idea. If no suggestion is given, leave this section empty (no code block).

Never repeat an idea from <previous failed ideas>, and avoid blacklisted or previous suggestions.

### Input Schema
- <competition description> (string): Detailed overview of the Kaggle competition (task, data, evaluation metric).
- <researcher plans> (optional, list of strings): Previous plans for the task.
- <initial script> (string): Starting code.
- <logs> (string): Output logs from training/evaluation of the script.
- <previous suggestion executed> (string): Most recently attempted suggestion.
- <previous failed ideas> (optional, list of strings): Suggestions that have previously failed or been blacklisted.

### Output Fields
- Checklist (markdown list)
- Research and Suggestion (two markdown sections)
- Validation (markdown)
- Previous Suggestion Review (strict JSON)
- New Suggestion Summary (strict JSON)
- Code (Python, if a suggestion is present)

Error handling:
- If <competition description> or <initial script and logs> are missing or inadequate, note this before the checklist and use "No suggestions." everywhere else.
- When validating, explicitly reference relevant input fields (e.g., competition metric or logs).
- If unable to validate due to lack of input, state this and use 'No suggestions.'
"""


def sota_user(
    description: str,
    plans_section: str,
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

