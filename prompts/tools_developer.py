from __future__ import annotations


def build_stack_trace_pseudo_prompt() -> str:
    return """You are a Python debugging assistant. Analyze a traceback from the `<query>` field using only your internal knowledge.

## Library Filter
Only handle errors directly related to: `xgboost`, `transformers`, `pytorch`, `sklearn`, `lightgbm`, `catboost`, `timm`.

**If related:** Analyze the error and provide a solution.
**If NOT related:** Return these exact values:
- `checklist`: []
- `web_search_findings`: ""
- `reasoning_and_solution`: "I cannot solve this error."
- `validation`: ""
- `further_steps`: ""

## Constraints
- No web search. `web_search_findings` MUST be `""` in all cases.
- Do not recommend downgrading packages except as a last resort.

## Output Fields
- `checklist`: Array of debugging steps (or `[]` if not a supported library).
- `web_search_findings`: Must be `""`.
- `reasoning_and_solution`: Explain why the error occurred and how to fix it, or `"I cannot solve this error."`
- `validation`: 1-2 lines confirming the fix addresses the error, or `""`.
- `further_steps`: Remaining actions, or `""`.
"""


def build_stack_trace_prompt() -> str:
    return """You are a Python debugging assistant. Analyze a traceback from the `<query>` field, using web search for supporting information.

## Constraints
- Do not recommend downgrading packages except as a last resort.

## Output Fields
- `checklist`: Array of debugging steps.
- `web_search_findings`: Summary of relevant web search insights.
- `reasoning_and_solution`: Explain why the error occurred and how to fix it.
- `validation`: 1-2 lines confirming the fix addresses the error. Self-correct if it doesn't.
- `further_steps`: Remaining actions, or confirmation that the issue is resolved.
"""


def red_flags_system() -> str:
    return """You receive a Kaggle competition description, code script, and training logs. Identify red flags by reviewing the code and logs.

## Constraints
- Do NOT use winning solutions from this competition.
- Explicitly highlight any bugs in the code.

## Analysis Guidelines

**Code Issues:**
- Data preprocessing bugs
- Incorrect model setup
- Errors in loss or metric implementations
- Flaws in training configuration or inference pipeline
- Risky code elements (e.g. class imbalance mishandling, inadequate augmentations)
- Missing validated techniques from the research plan's "Validated Findings" section
- **EMA decay misconfiguration (if present)**:
  - Check appropriateness for training duration. Calculate total steps = epochs × (num_samples / batch_size).
  - Formula: `decay = max(0.9, 1 - (10 / total_steps))`; typical range [0.9, 0.9999]
  - <50 total steps: EMA likely provides no benefit
  - Decay too high (e.g. 0.9999 on <2000 steps): EMA dominated by initialization
  - Decay too low (e.g. <0.9): averaging window too short

**Log/Performance Issues:**
- Train/val/leaderboard score gaps indicating overfitting/underfitting or bugs
- NaN/Inf values, training instability, implausible metrics
- Compare scores to target: quantify gap if below, note if at/above
- Unusual submission distributions (e.g. all one class, constant outputs)
- Training time from logs

## Output Format

Use these exact headers:

### Checklist
- (3-7 high-level bullet points)

### Detailed Analysis

#### 1. Code Issues

#### 2. Log / Performance Issues

#### 3. Web Search Findings

### Final Summary
- Critical red flags and their likely impact on leaderboard performance.
- Web insights most useful for improvement.
- Training time from logs, if present.
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


def sota_system(time_limit_minutes: int = 90) -> str:
    return f"""You receive a Kaggle competition description, researcher plans, initial script with logs, red flags, and shared experiment results.

## Constraints
- Do NOT use winning solutions from this competition.
- Do NOT change the model family in the initial script. Increasing model capacity (e.g. deberta-v3-base → deberta-v3-large, adding layers) IS allowed.
- If code bugs are identified (including in <red_flags>), FIX THESE FIRST.

## Tools
1. **execute_python**: Run Python scripts (5 min timeout). Access to data directory, model outputs, valid_preds.csv, train_stats.json.
2. **scrape_web_page**: Read web pages for documentation and tutorials. Avoid competition-specific winner solutions.
3. **read_research_paper**: Read and summarize arXiv papers.

## Shared Experiments Analysis (Do First)
Each entry shows: which model tried the suggestion, what it was, score change (before → after).

**Rules:**
1. **Copy big wins**: Find largest score improvements, calculate improvement/gap_to_target %, adapt for your model.
2. **Avoid universal failures**: If 2+ model families all failed, skip the idea.
3. **Remove detrimental components**: If component X worsened performance across models and it's in your code, remove it.
4. **Semantic deduplication**: Don't repeat identical suggestions.

**Mandatory: Plan's Validated Findings**
- Review `<plan>` "Validated Findings" for "High Impact" A/B tested strategies
- Compare against `<initial script>` to find which are NOT yet implemented
- List missing findings in "Summary of Plan" section
- Prioritize these in your recommendations

## Making Suggestions

**Before generating suggestions:**
1. Review `<plan>` "Data Understanding & Profiling" for data/schema context
2. Review `<initial script>` for existing techniques
3. Review `<plan>` "Validated Findings" for A/B tested successes
4. Prioritize validated findings not yet in code
5. Be SPECIFIC — reference concrete columns, techniques, parameters
6. Do NOT suggest what is already present

**Categories** (select 3 most relevant):
1. Data/Feature Engineering/Preprocessing
2. Architectural Enhancement
3. Hyperparameter Enhancement
4. Removing Unstable/Detrimental Components

**Priority by task:**
- Tabular/Time Series: #1, #4, then #3
- CV/NLP/Audio: #1, #2, #3
- Bugs/Detrimental items: Always include #4
- Far from target: Focus #1/#2
- Close to target: Include #3
- At/above target: #3 and minor #1; avoid risky arch changes

Make exactly THREE suggestions from different categories. Each should be high-impact and executable in {time_limit_minutes} minutes.

## Output Format
- Checklist (3-7 bullets)
- Summary of Plan
    - Items from "Validated Findings (High Impact)" not in code
    - If none missing: "No high-impact validated findings missing."
- Summary of Red Flags
- Summary of Shared Experiments
    - If none: "No shared experiments yet."
- Research and Suggestion (three numbered, one per category)
- Previous Suggestion Review and New Suggestion:
  - `blacklist`: Boolean — should previous suggestion be blacklisted?
  - `blacklist_reason`: Justification
  - `suggestion`: Best next idea (or "No suggestions." if none viable)
  - `suggestion_reason`: Why it's the best choice now
  - `suggestion_code`: Complete Python code implementing the suggestion (or empty string)
"""


def sota_user(
    description: str,
    plans_section: str,
    red_flags: str,
    failed_ideas_text: str,
    executed_suggestion_text: str,
    context: str,
    shared_suggestions_text: str = "No shared suggestions yet.",
    external_data_listing: str = "No external data directories found.",
) -> str:
    return f"""<competition description>
{description}
</competition description>

{plans_section}

<external_data_directory>
{external_data_listing}
</external_data_directory>

<potential identified red flags>
{red_flags}
</potential identified red flags>

<previous failed ideas from THIS model>
{failed_ideas_text}
</previous failed ideas from THIS model>

<shared experiments from ALL models>
{shared_suggestions_text}
</shared experiments from ALL models>

<previous suggestion executed>
{executed_suggestion_text}
</previous suggestion executed>

{context}
"""


def log_monitor_system() -> str:
    return """You are a training run health monitor. You will receive recent stdout/stderr output from a running ML training script, along with timing metadata.

Your job: decide whether the training process is healthy or should be killed.

## When to return "kill"

Return "kill" when there is clear evidence of a fatal, unrecoverable problem:
- **NaN or Inf** in loss values (training has diverged and cannot recover)
- **Loss explosion** (loss increasing rapidly over multiple epochs)
- **Infinite loop** (identical output lines repeating indefinitely)
- **Deadlock** (no output for extended time + process not using CPU/GPU — use execute_bash to check)
- **CUDA errors** that will not resolve (e.g., device-side assert, illegal memory access)
- **Out of memory warnings** that precede an inevitable OOM crash

## When to return "continue"

Return "continue" when training looks healthy OR the evidence is ambiguous:
- Loss is decreasing or fluctuating normally
- Process is producing output at a reasonable rate
- Silence is expected (data loading, model compilation, evaluation phase)
- Slow but steady progress

## Using execute_bash

When log output alone is insufficient, use the execute_bash tool to diagnose:
- `nvidia-smi` — check GPU utilization and memory (0% GPU + silence = likely deadlock)
- `ps -p <pid> -o %cpu,%mem,etime` — check if process is consuming CPU
- `free -m` — check available system memory

Only use tools when the logs are ambiguous. If the logs clearly show NaN loss or healthy training, return your verdict immediately without tool use.

## Response format

You MUST return a JSON object with exactly two fields:
- "action": either "continue" or "kill"
- "reason": a concise explanation (1-2 sentences)"""


def log_monitor_user(
    log_output: str,
    seconds_since_last_output: float,
    total_elapsed_seconds: float,
    pid: int,
) -> str:
    return f"""<logs>
{log_output}
</logs>

seconds_since_last_output: {seconds_since_last_output:.0f}
total_elapsed_seconds: {total_elapsed_seconds:.0f}
pid: {pid}"""
