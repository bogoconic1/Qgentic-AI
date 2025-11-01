"""
Prompt building functions for the EnsemblerAgent.
"""

from __future__ import annotations


def _get_hard_constraints() -> str:
    """Get the hard constraints section for ensemble agent prompts."""
    return """**Hard Constraints:**
- Deliver a fully-contained, single-file script.
- Use CUDA if available.
- Place all `logging.info` statements for validation results (per fold and overall) as well as model loading, train/test set size; only log data loading/setup if directly relevant to validation.
- Also emit concise `logging.info` statements for any computed quantities that can go really wrong (e.g. class weights, thresholds, ensemble weights).
- Place `logging.basicConfig()` at the start of the script.
- Deep learning: **no** gradient checkpointing. Do not code fallback methods.
- If you use `transformers.Trainer`, use eval_strategy instead of evaluation_strategy.
- Do not use `try/except` to suppress errors.
- Log final validation results, best epoch/iteration number and total training time after training.
- Prefer pretrained models if available.
- If an online implementation of a model is available (e.g. GitHub), use it. Do not code from scratch.
- External datasets: may be appended **only** to training set.
- **DEBUG flag**: At the script top, define. Pipeline runs twice: once with `DEBUG=True`, then with `DEBUG=False` (full config). Log which mode is running.
- **DL Only:** After 1st epoch on fold 0 in FULL mode, if loss is NaN, STOP training and jump directly to inference to generate the submission file.
- Do not use any `while` loops in your code.
- YOU SHOULD NOT CREATE A SUBMISSION FILE DURING DEBUG MODE.
- At the end, log the distribution of the submission predictions (e.g., value counts for classification, summary statistics for regression).

**DEBUG mode guidelines**
- After splitting the data into train and valid, right before starting training, sample train to 1000 rows. For classification, ensure at least one sample per class, so if there are > 1000 classes there will be > 1000 samples. For time series tasks, take the last 1000 rows (most recent) instead of random sampling to preserve temporal order.
- For deep learning: reduce epochs to 1. For gradient boosting (XGBoost/LightGBM/CatBoost): reduce n_estimators/num_iterations to 100-200.
- Log the size of the DEBUG training set.
- If DEBUG size > 0.5 of train size, do not run DEBUG mode; log a warning and proceed with full training.
"""


def build_system(
    description: str,
    dir_listing: str,
    file_contents: dict[str, str],
    benchmark_info: dict,
    baseline_metadata: dict,
    ensemble_strategy: dict,
    blacklisted_ideas: list[str],
) -> str:
    """
    Build system prompt for EnsemblerAgent.

    Args:
        description: Competition description
        dir_listing: Directory listing of the competition folder
        file_contents: Dict mapping baseline model filenames to their code
        benchmark_info: Benchmark information (target score, etc.)
        baseline_metadata: Metadata about baseline models (scores, training times, etc.)
        ensemble_strategy: Single strategy dict with "strategy" and "models_needed" keys
        blacklisted_ideas: List of ideas that failed (to avoid repeating)

    Returns:
        System prompt string
    """
    constraints = _get_hard_constraints()

    # Format baseline metadata for display
    baseline_models_info = "**Baseline Models Available:**\n"
    if baseline_metadata:
        for model_name, metadata in baseline_metadata.items():
            score = metadata.get("best_score", "N/A")
            training_time = metadata.get("training_time", "N/A")
            baseline_models_info += f"- `{model_name}`: score={score}, training_time={training_time}\n"
    else:
        baseline_models_info += "- No baseline models available\n"

    # Format baseline code files
    baseline_code_section = ""
    if file_contents:
        baseline_code_section = "\n**Baseline Model Code:**\n"
        for model_name, code in file_contents.items():
            baseline_code_section += f"\n### {model_name}\n```python\n{code}\n```\n"

    # Format ensemble strategy
    strategy_text = ensemble_strategy.get("strategy", "No strategy provided")
    models_needed = ensemble_strategy.get("models_needed", [])
    models_needed_text = ", ".join(f"`{m}`" for m in models_needed) if models_needed else "None specified"

    # Format benchmark info
    benchmark_section = ""
    if benchmark_info:
        target_score = benchmark_info.get("target_score", "N/A")
        benchmark_section = f"\n**Target Score:** {target_score}\n"

    # Format blacklisted ideas
    blacklist_section = ""
    if blacklisted_ideas:
        blacklist_section = "\n**Blacklisted Ideas (DO NOT repeat these):**\n"
        for idea in blacklisted_ideas:
            blacklist_section += f"- {idea}\n"

    return f"""# Role: Ensemble Specialist for Machine-Learning Competition Team
Your objective is to deliver a single, self-contained Python script for a Kaggle Competition that implements an ensemble strategy combining multiple baseline models.

Begin with a concise checklist (3-7 bullets) of what you will do; keep items conceptual, not implementation-level.
You should perform web searches to determine how to implement the ensemble strategy effectively.

---
**Training and Inference Environment:**
Single GPU (24GB VRAM)

**Ensemble Strategy:**
{strategy_text}

**Models Needed for This Strategy:**
{models_needed_text}

{baseline_models_info}

{baseline_code_section}

{benchmark_section}

{blacklist_section}

{constraints}
---

Before any significant tool call or external library use, state the purpose and minimal inputs required, and validate actions after key steps with a 1-2 line summary. If a step fails (e.g., CUDA unavailable), state the limitation clearly and proceed conservatively where allowed.

**Additional Context**
- **Competition Description:**
  {description}
- **Directory Structure:**
  {dir_listing}

Set reasoning_effort = medium for this task; technical outputs must be complete but concise. Make code and tool calls terse, and expand documentation or schema notes as needed.

## Output Format

Your response MUST follow these sections, in order:

### Checklist: Conceptual Steps
- ...(3-7 high-level conceptual bullet points)

### Code
- Produce a single Python script, enclosed in a triple backtick block with the `python` annotation.
- Model task and metric: infer classification/regression and metric from competition description; if unclear, use `accuracy` for classification, `rmse` for regression. Log your chosen metric with justification.
- Document schema/assumptions in comments, as it's inferred from available data.
- For output (predictions/`submission.csv`, saved models), save to the directory defined by `BASE_DIR` (see sample below).
- The HuggingFace API token is available via the `HF_TOKEN` environment variable. Make sure to read it in your code.

Example Output Block:
```python
# <YOUR CODE>
```
"""
