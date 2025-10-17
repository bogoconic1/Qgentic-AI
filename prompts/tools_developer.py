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
    return """You will receive a Kaggle competition description, one or more researcher plans, and an ablation summary for analysis.

Begin with a concise checklist (3-7 bullets) summarizing high-level conceptual red flags identified from the code/logs, as well as your intended strategies for addressing them. These should focus on conceptual aspects rather than specific implementations. Use '- ' for each bullet. If fewer than three meaningful points arise, list as many as possible and explicitly state: "Fewer than 3 high-level red flags or strategies identified."

Always review the <researcher_plans> and <ablation_summary> first, if provided. Set reasoning_effort = medium to ensure thoughtful but efficient analysis. For any web search or external query, briefly state the purpose and the minimal search terms you will use before proceeding. Use only approved resources and provide a one-line preamble before significant information-sourcing steps, referencing the competition context.

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
Identify any previously tried ideas from <ablation_summary> that should remain blacklisted or be newly blacklisted. Provide a list of exact idea strings to blacklist and a corresponding list of reasons aligned by index.

Output your decision in the following strict JSON format (enclosed in backticks):
```json
{
    "blacklist": ["<idea to blacklist>", "<another idea to blacklist>", ...],
    "reasons": ["<reason for first idea>", "<reason for second idea>", ...]
}
```

### New Suggestions Summary
Summarize the four new suggestions in a concise manner and provide a Python code snippet for each suggestion. Do not repeat blacklisted ideas or the previous suggestion.

Use these exact keys: data_feature_suggestion/data_feature_code, arch_suggestion/arch_code, ensembling_suggestion/ensembling_code, sota_suggestion/sota_code. Return your new suggestions using the following strict JSON format (enclosed in backticks):
```json
{
    "data": {
        "suggestion": <Data / Feature Engineering Suggestion>,
        "code": <Python code snippet for the data / feature engineering suggestion>
    },
    "architecture": {
        "suggestion": <Architectural Enhancement Suggestion>,
        "code": <Python code snippet for the architectural enhancement suggestion>
    },
    "ensembling": {
        "suggestion": <Ensembling/Blending Enhancement Suggestion>,
        "code": <Python code snippet for the ensembling/blending enhancement suggestion>
    },
    "sota": {
        "suggestion": <SOTA Model Enhancement Suggestion>,
        "code": <Python code snippet for the SOTA model enhancement suggestion>
    }
}
```
If there is no viable suggestion, use empty strings for the values.
Never repeat any idea from <previous failed ideas>. If a suggestion is blacklisted, ensure your new recommendation avoids that approach.
"""


def sota_user(
    description: str,
    plans_section: str,
    failed_ideas_text: str,
    ablation_summary: str | None = None,
) -> str:
    ablation_block = f"\n<ablation_summary>\n{ablation_summary}\n</ablation_summary>\n" if ablation_summary else ""
    return f"""<competition description>
{description}
</competition description>

{ablation_block}

{plans_section}

<previous failed ideas> DO NOT TRY THESE AGAIN
{failed_ideas_text}
</previous failed ideas>
"""


def ablation_baseline_prompt() -> str:
    return """You will be provided with a piece of code and its corresponding logs.

Inputs:
- <code>: The code to analyze.
- <logs>: The logs to analyze.

Begin with a concise checklist (3-7 bullets) of what you will do; keep items conceptual, not implementation-level.

Instructions:
1. Summarize both the code and the logs concisely.
2. Output the result in JSON format as specified below.
3. Incorporate all relevant key findings and takeaways from both the code and the logs.
4. If either the code or the logs is missing, return an empty string ('') in the respective summary field.
5. Do not return null values or omit any keys from the output.
6. If the provided code or logs are excessively long or malformed, summarize using the available information and clearly note any limitations due to input quality.

After generating the summaries, verify that both summary fields are present, use empty strings where required, and that all JSON keys match the output schema exactly. If any issues are found, self-correct and revalidate before outputting the final response.

## Output Format
Your output MUST include the following sections in order:

### Checklist
- ... (3-7 high-level conceptual bullet points, see above)

### Code and Logs Summary
Output your summaries in the following strict JSON format (enclosed in backticks):
```json
{
  "code_summary": "Summary of the code or '' if code is absent",
  "logs_summary": "Summary of the logs or '' if logs are absent"
}
```
"""


