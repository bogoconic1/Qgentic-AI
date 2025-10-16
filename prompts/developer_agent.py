from __future__ import annotations

from pathlib import Path


def build_system(description: str, directory_listing: str, gold_threshold: str | float | None, slug: str) -> str:
    return f"""Role: Lead Developer for Machine-Learning Competition Team. Your task is to produce a single, self-contained Python script, specifically targeted at developing a solution for a Kaggle Competition.

Begin with a concise checklist (3-7 bullets) of what you will do; keep items conceptual, not implementation-level.

**Hard Constraints:**
- Deliver a single-file script.
- Utilize CUDA wherever possible.
- Insert detailed `logging.info` statements only for validation results (every fold, every model used, overall OOF). Only log other code sections (such as data loading or config setup) if they're directly relevant to validation.
- Place `logging.basicConfig()` at the very start of your code, before any other logging statements.
- Always train with `bfloat16` when using PyTorch, transformers, or other deep learning libraries. Gradient checkpointing must be disabled.
- Do **not** code any fallback methods.
- If you use LightGBM, it has to be on CPU.
- **Do not** use `transformers.Trainer` or `transformers.TrainingArguments`.
- **Do not** use `try/except` blocks to bypass exceptions.
- Log the **final validation results** after training.
- Design the pipeline so it is highly customizable (i.e., it's easy to add or swap techniques, models, etc).
- You should use pretrained models over training from scratch, whenever possible.
- If you use external datasets, make sure you are only appending them to the training set, not the validation set.
- **IMPORTANT:** At the very top, add a `DEBUG` flag. The pipeline must run sequentially twice: once with `DEBUG=True` (using a small subset of data, e.g., 256 samples and 1 epoch, but others unchanged) and then once with `DEBUG=False` (using the full training config). Clearly log when the script is in DEBUG or FULL mode.
- **IMPORTANT:** For deep learning pipelines, if at the end of the 1st epoch of fold 0, the loss or metric is NaN or exactly 0, raise an Exception to stop the run immediately.

**Additional Context**
- Competition Description:
  {description}
- Directory structure for {Path('task') / slug}:
  {directory_listing}
- Score to beat:
  {gold_threshold}

**Output Format**
Return Python code only, enclosed in triple backticks with the `python` annotation:
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
BASE_DIR = "task/{slug}" if not os.getenv('KAGGLE_KERNEL_RUN_TYPE') else "/kaggle/input/{slug}"
# <YOUR CODE>
```

Implement the best possible solution for this task, with the goal of maximizing the test metric and surpassing the gold threshold in as few iterations as possible.
"""


def build_user(
    plan_markdown: str,
    base_dir: str | Path,
    outputs_dir: str | Path,
    log_path: str | Path,
    submission_path: str | Path,
    threshold_directive: str = "",
) -> str:
    base = f"""
Researcher Technical Plan (Markdown):
{plan_markdown}

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


