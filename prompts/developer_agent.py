from __future__ import annotations

from pathlib import Path


def build_system(description: str, directory_listing: str, model_name: str, example_details: str, researcher_data_driven_recommendations: str, slug: str) -> str:
    return f"""# Role: Lead Developer for Machine-Learning Competition Team
Your objective is to deliver a single, self-contained Python script for a Kaggle Competition using **only** the specified model `{model_name}`.

Begin with a concise checklist (3-7 bullets) of what you will do; keep items conceptual, not implementation-level.
You should perform web searches to determine how to set up and configure `{model_name}` in Python.

---

**Model Name:**
`{model_name}`

**Your Previous Recommendations**
{example_details}

**Researcher Data-Driven Recommendations**
{researcher_data_driven_recommendations}

**Hard Constraints:**
- Use ONLY `{model_name}` (no substitutions or fallback models).
- Deliver a fully-contained, single-file script.
- Use CUDA whenever available.
- Place all `logging.info` statements for validation results only (per fold and overall); only log data loading/setup if directly relevant to validation.
- Place `logging.basicConfig()` at the start of the script.
- Deep learning: always use `bfloat16`, **no** gradient checkpointing. Do not code fallback methods.
- LightGBM (if used): **CPU only**.
- If you use `transformers.Trainer`, use eval_strategy instead of evaluation_strategy.
- Do not use `try/except` to suppress errors.
- Log final validation results, best epoch number and total training time after training.
- Modular pipeline: update preprocessing/postprocessing or hyperparameters, but do not swap out `{model_name}`.
- Prefer pretrained models if available.
- External datasets: may be appended **only** to training set.
- **DEBUG flag**: At the script top, define. Pipeline runs twice: once with `DEBUG=True` (subset of data, e.g., 1000 samples, 1 epoch), then with `DEBUG=False` (full config). Log which mode is running.
- **DL Only:** After 1st epoch on fold 0, if loss is NaN, raise Exception to halt.
- Split: 80% train, 20% validation. **No K-Fold** methods.

---

Before any significant tool call or external library use, state the purpose and minimal inputs required, and validate actions after key steps with a 1-2 line summary. If a step fails (e.g., CUDA unavailable), state the limitation clearly and proceed conservatively where allowed.

**Additional Context**
- **Competition Description:**
  {description}
- **Directory Structure for `{Path('task') / slug}`:**
  {directory_listing}

Set reasoning_effort = medium for this task; technical outputs must be complete but concise. Make code and tool calls terse, and expand documentation or schema notes as needed.

## Output Format

Your response MUST follow these sections, in order:

### Checklist: Conceptual Steps
- ...(3-7 high-level conceptual bullet points)

### How would I update my previous recommendations to incorporate the researcher data-driven recommendations?
- ...(if anything in your previous recommendation is not the best choice for this task, explain why and what is the best choice)
- ...(explain the changes you make to your previous recommendations based on the researcher data-driven recommendations)

### My new strategy
- ...(explain the data/preprocessing/feature engineering step, why it is the best choice for this task)
- ...(explain the model/training/evaluation step, why it is the best choice for this task)
- ...(explain the metric and loss function used, why it is the best choice for this task)
- ...(explain the training regime used, why it is the best choice for this task)
- ...(explain the inference/postprocessing step, why it is the best choice for this task)

### Code
- Produce a single Python script, enclosed in a triple backtick block with the `python` annotation.
- Model task and metric: infer classification/regression and metric from `{description}`; if unclear, use `accuracy` for classification, `rmse` for regression. Log your chosen metric with justification.
- Document schema/assumptions in comments, as it's inferred from available data.
- For output (predictions/`submission.csv`, saved models), save to the directory defined by `BASE_DIR` (see sample below).

Example Output Block:
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
BASE_DIR = "task/{slug}" if not os.getenv('KAGGLE_KERNEL_RUN_TYPE') else "/kaggle/input/{slug}"
# <YOUR CODE>
```
"""


def build_user(
    base_dir: str | Path,
    outputs_dir: str | Path,
    log_path: str | Path,
    submission_path: str | Path,
    threshold_directive: str = "",
) -> str:
    base = f"""
Project structure:
- Base data dir: {base_dir}
- Outputs dir: {outputs_dir}
- The logs should be written to a file named {log_path}
- Required output: {submission_path}
"""
    base += (
        "\nReturn the complete Python script that, when run, writes logs to "
        f"{log_path} "
        "and produces a submission CSV at "
        f"{submission_path}."
    )
    if threshold_directive:
        base += f"\n{threshold_directive}"
    return base


def patch_mode_directive(base_filename: str) -> str:
    return (
        f"""**IMPORTANT**: Please write a git diff (patch) within ```diff to fix the above issues!
- Produce a unified diff (apply patch format) that updates {base_filename}. Do not include any prefixes in the diff other than {base_filename}.
- Return only the diff enclosed in ```diff fences; do not resend the full script.
- Ensure the diff applies cleanly with the `patch` utility using the file names above.
- Use standard hunk headers with explicit line numbers, e.g. @@ -12,7 +12,9 @@.
- Refer to the <previous_code_with_line_numbers> section above when calculating line numbers.

Like this
```diff
--- {base_filename}
+++ {base_filename}
@@ -1,3 +1,3 @@
 start
-first change
+new first change
 middle
@@ -7,4 +7,4 @@
 some content
-second change
+new second change
 more content
 end
 ```"""
    ).strip()


def guardrail_fix_suffix(next_log_path: str | Path, next_submission_path: str | Path) -> str:
    return (
        "\nPlease regenerate the script addressing the above guardrail issues. "
        f"Write logs to {next_log_path} "
        f"and produce {next_submission_path}."
    )


def execution_failure_suffix(next_log_path: str | Path, next_submission_path: str | Path) -> str:
    return (
        "\nPlease modify your code to fix the error!\n\n"
        "Remember:\n"
        f"- write logs to {next_log_path}\n"
        f"- and produce the next submission at {next_submission_path}"
    )

