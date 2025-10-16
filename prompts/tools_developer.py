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
    return """Developer: You will receive a Kaggle competition description, one or more researcher plans, and an ablation summary for analysis.

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

### New Suggestion Summary
Propose the single best next idea (only one) for improving the competition score, synthesizing insights from the four categories above. Do not repeat blacklisted ideas or the previous suggestion.

Return your new idea using the following strict JSON format (enclosed in backticks):
```json
{
    "suggestion": "<your proposed best next idea>",
    "reasoning": "<explanation for selecting this idea as the best compared to other promising ideas>"
}
```
If there is no viable suggestion, use empty strings for the values.

### Code
Present a concise Python code snippet (within triple backticks marked 'python') that implements your proposed best next idea. If no suggestion is made, leave this section empty (no code block).

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
    return """You will receive a structured input:

<baseline>: The baseline run, with a 'score' field (preferably numeric; if not, provide a text description).

Begin with a concise checklist (3-7 bullets) outlining your intended approach before drafting the steer.

Draft a concise steer (maximum 180 words) covering:
- Baseline score
- Notes on stability
- 3-5 brief bullet points (each on its own line, starting with '-') offering takeaways or next steps for the next iteration

Set reasoning_effort = minimal, as appropriate for a summary-focused task.
Before extracting or reporting key results, verify their presence and format in the provided baseline data.
Respond using only plain text.

## Output Format
Return your answer strictly in the following JSON format:
```
{
  "checklist": [
    "your first checklist item",
    "your second checklist item",
    ...(3-7 items)
  ],
  "steer": {
    "baseline_score": <numeric or string, as found in <baseline>>,
    "stability_notes": "1-3 sentences on stability (keep it brief)",
    "takeaways": [
      "- recommended next step or insight",
      "- another key takeaway (3-5 total)",
      ...
    ]
  }
}
```
- Use the data format (numeric, float, or string) for "baseline_score" as appears in <baseline>.
- Output strictly the structured JSON. Do not add explanations or narrative outside the JSON structure."""


def ablation_batch_prompt() -> str:
    return """You will receive two structured inputs:
- <baseline>: The baseline run, with a 'score' field (preferably numeric; if not, provide a text description).
- <batch_results>: An array of four candidate runs, each with at least:
    - identifier: A unique string or integer (may be named 'id', 'name', etc.; use as provided).
    - score: Numeric value when available (if absent or not numeric, report as 'missing').
    - status: String indicating run outcome (use as provided, e.g., 'success', 'failure').

Begin with a concise checklist (3-7 conceptual bullets) outlining your intended steps; keep items conceptual and do not include implementation details.

When reviewing candidates, if a significant data or field issue arises—such as missing identifiers, scores, or statuses—briefly state the nature of the problem and suggest next actions or verification as appropriate.

Produce a concise ablation summary in JSON output (maximum 220 words in the summary fields), including:
- The baseline score (or a descriptive text if unavailable or not numeric) in a field called 'baseline_score'.
- For each candidate (in order as per batch_results), provide an object with its identifier (or 'identifier missing'), score ('missing' if absent), and status. If identifier is missing, write 'identifier missing'. Each object should also include an analysis of the candidate's effect (positive, negative, or neutral) under the key 'effect' with 1-2 reasons for each candidate under 'reasons'.
- Conclude with a 'guidance' field, containing a 1-2 sentence recommendation for the next iteration.

After completing the summary, review for completeness and clarity; validate that all requested elements are addressed, and self-correct for any omissions or unclear points.

Highlight as a data issue if a candidate has both a missing score and a status not marked as 'success'; advise verification in such cases.

If critical input fields are missing or malformed (e.g., baseline missing score, or a candidate missing identifier, score, and status), clearly state this in the summary (in a 'data_issues' field) and propose corrective action. If the score is non-numeric, display it as-is and specify its type or explain the reason for non-numeric status (in a 'notes' or similar field).

Restrict output to JSON only; do not use extra formatting, markdown, or informal conversation.

Template for output:

{
  "baseline_score": [numeric value, or text if unavailable/malformed],
  "candidates": [
    {
      "identifier": [value or 'identifier missing'],
      "score": [numeric value or 'missing'],
      "status": [status],
      "effect": [positive/negative/neutral],
      "reasons": [list of 1-2 concise rationales]
    },
    ...
  ],
  "guidance": [concise recommendation for next steps],
  "data_issues": [list of data issues if present],
  "notes": [list of explanatory notes if applicable]
}

- If baseline score is missing or non-numeric, display it as-is and note this in 'notes'.
- If a candidate is missing an identifier, use 'identifier missing' in 'identifier'.
- If both score and status are missing or unclear, specify the malformed entry in 'data_issues' and recommend correction.
- All output must be valid JSON, no extra formatting, markdown, or conversational elements."""


