"""
Prompt building functions for the EnsemblerAgent.
"""

from __future__ import annotations

from prompts.shared_constraints import get_hard_constraints


def _get_hard_constraints() -> str:
    """
    Get the hard constraints section for ensemble agent prompts.

    This is a wrapper around the shared get_hard_constraints function that
    maintains backward compatibility with the existing ensembler agent interface.

    Returns:
        Formatted hard constraints string with ensemble-specific directives
    """
    return get_hard_constraints(
        model_name=None,
        allow_multi_fold=False,
        include_ensemble_copy_directive=True,
    )


def build_system(
    description: str,
    dir_listing: str,
    file_contents: dict[str, str],
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
        baseline_metadata: Metadata about baseline models (scores, training times, etc.)
        ensemble_strategy: Single strategy dict with "strategy" and "models_needed" keys
        blacklisted_ideas: List of ideas that failed (to avoid repeating)

    Returns:
        System prompt string
    """
    constraints = _get_hard_constraints()

    # Get models needed for this strategy
    models_needed = ensemble_strategy.get("models_needed", [])

    # Format baseline metadata for display (only models needed for this strategy)
    baseline_models_info = "**Baseline Models Available:**\n"
    if baseline_metadata and models_needed:
        for model_name in models_needed:
            metadata = baseline_metadata.get(model_name, {})
            if not metadata:  # Skip if model not found in metadata
                continue
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
    models_needed_text = ", ".join(f"`{m}`" for m in models_needed) if models_needed else "None specified"

    # Format blacklisted ideas
    blacklist_section = ""
    if blacklisted_ideas:
        blacklist_section = "\n**Blacklisted Ideas (DO NOT repeat these):**\n"
        for idea in blacklisted_ideas:
            blacklist_section += f"- {idea}\n"

    return f"""# Role: Ensemble Specialist for Machine-Learning Competition Team
Your objective is to deliver a single, self-contained Python script for a Kaggle Competition that implements an ensemble strategy combining multiple baseline models.

Begin with a concise checklist (3-7 bullets) of what you will do; keep items conceptual, not implementation-level.
You should perform web searches to determine how to implement the ensemble strategy effectively, or setting up any models marked as "NEW:".

---
**Training and Inference Environment:**
Single GPU (40GB VRAM)

**Ensemble Strategy:**
{strategy_text}

**Models Needed for This Strategy:**
{models_needed_text}

{baseline_models_info}

{baseline_code_section}

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

def ensembler_code_summarize_system_prompt():
    return """You are a **Kaggle Competitions Grandmaster**.

Your task is to analyze a complete Python implementation and its execution logs for a specific model crafted for a particular Kaggle competition, and generate a comprehensive technical summary.

Begin with a concise checklist (3-7 bullets) summarizing your planned review steps for the provided Python code and execution logs. Use this to structure your analysis and ensure clarity across sections.

## Inputs
- `<competition_description>`
- `<model_name>`
- `<code>`
- `<execution_logs>`

## Output Format
Format the summary in Markdown and use the following sections. Present details in the order they appear in the code and logs, grouping items by their function or topic for clarity. 

### Preprocessing
- List all preprocessing steps taken, grouped by type (e.g., data cleaning, feature engineering, normalization).

### Loss Function
- Describe the loss function(s) used, including type and parameters (if any).
- Explain why this loss is likely suitable for the competition.

### Model Architecture and Hyperparameters
- Detail the model's structure: layers, activation functions, and regularization methods.
- List hyperparameters with their values.

### Inference
- Describe the inference process: input handling, prediction method, output post-processing.
- Note batch sizes, ensembling, or inference-time customizations if present.
"""

def ensembler_code_summarize_user_prompt(description: str, model_name: str, code: str, logs: str):
    return f"""<competition_description>
{description}
</competition_description>

<model_name>
{model_name}
</model_name>

<code>
```python
{code}
```
</code>

<execution_logs>
{logs}
</execution_logs>
"""
