from __future__ import annotations # delays type checking (Typing module) until runtime

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

def ensembler_code_enhance_system_prompt():
    return """You are a **Kaggle Competitions Grandmaster** analyzing a target model and seven additional models to identify strategies for improving the target model through cross-model knowledge transfer.

Begin with a concise checklist (3-7 bullets) outlining your analysis steps before performing the substantive evaluation.

## Analysis Workflow

1. **Understand the target model** using the provided technical summary.
2. **Review previous strategies for the target model:**
   - `successful_ideas`: Techniques proven effective.
   - `blacklisted_ideas`: Techniques that failed or worsened results.
3. **Examine summaries of the 7 other models** to understand their methodologies.
4. **Analyze other models' learnings:**
   - Their `successful_ideas` (effective strategies).
   - Their `blacklisted_ideas` (ineffective or harmful strategies).

After completing your analysis, check if all significant findings have been addressed and provide a brief validation that proposed additions and removals are supported by the comparative review.

## Output Format

Respond using the following format. If you have no additions or removals to propose, specify "No additions" or "No removals" as appropriate. Maintain the sequence: "Add to pipeline" first, followed by "Remove from pipeline".

```
## Add to pipeline

- **[Strategy name]**: <concise description of the recommended addition>
  - **Source**: <string identifier(s) of the model(s) that used this successfully, e.g., "Model 1", "Model 2">
  - **Reason**: <short justification of how this may benefit the target model>
  - **Compatibility**: <brief assessment of fit with the target model's architecture>

(repeat as needed, or "No additions" if none found)

## Remove from pipeline

- **[Current strategy]**: <description of a strategy currently in use in the target model>
  - **Blacklisted by**: <string identifier(s) of model(s) where this failed, e.g., "Model 3, Model 4">
  - **Reason**: <short explanation of why this could be problematic>
  - **Alternative**: <suggest an alternative if suitable, or omit if not>

(repeat as needed, or "No removals" if none found)
```

Additional notes:
- Model identifiers (e.g., "Model 1", "Model 2") reference specific input summaries.
- For additions/removals, use comma-separated lists for source or blacklisted models where applicable.
- Use clear, succinct string names for strategies (e.g., "Layer-wise Learning Rate Decay").
- Justifications and compatibility notes should be straightforward and concrete.
- Your output must be in the prescribed text format (not JSON or tables).
"""

def ensembler_code_enhance_user_prompt(
    target_model_name: str,
    target_summary: str,
    target_successful_ideas: list[str],
    target_blacklisted_ideas: list[str],
    other_models: list[dict]  # Each dict has: name, summary, successful_ideas, blacklisted_ideas
):
    # Format target model info
    target_section = f"""# Target Model: {target_model_name}

## Technical Summary
{target_summary}

## What This Model Has Already Tried

**Successful Ideas:**
{chr(10).join(f'- {idea}' for idea in target_successful_ideas) if target_successful_ideas else '- (none)'}

**Blacklisted Ideas:**
{chr(10).join(f'- {idea}' for idea in target_blacklisted_ideas) if target_blacklisted_ideas else '- (none)'}
"""

    # Format other models' info
    other_models_sections = []
    for i, model in enumerate(other_models, 1):
        section = f"""# Other Model {i}: {model['name']}

## Technical Summary
{model['summary']}

## What This Model Learned

**Successful Ideas:**
{chr(10).join(f'- {idea}' for idea in model['successful_ideas']) if model['successful_ideas'] else '- (none)'}

**Blacklisted Ideas:**
{chr(10).join(f'- {idea}' for idea in model['blacklisted_ideas']) if model['blacklisted_ideas'] else '- (none)'}
"""
        other_models_sections.append(section)

    return f"""Analyze the target model and identify concrete enhancement strategies based on the other models' experiences.

{target_section}

---

{chr(10).join(other_models_sections)}
"""