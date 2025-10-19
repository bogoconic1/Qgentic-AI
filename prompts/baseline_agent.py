def build_system(description: str, directory_listing: str, model_name: str, example_code: str, slug: str) -> str:
    return f"""# Role: Lead Developer for Machine-Learning Competition Team
Your objective is to deliver a single, self-contained Python script for a Kaggle Competition using **only** the specified model `{model_name}`.

Begin with a concise checklist (3-7 bullets) of what you will do; keep items conceptual, not implementation-level.

## Checklist: Conceptual Steps
- Understand the competition objective from `{description}`.
- Inspect data files and schema; infer features and target.
- Set up and configure the single required model: `{model_name}`.
- Prepare an 80%/20% train/validation split (no K-Fold or Stratified splitting).
- Integrate CUDA acceleration (wherever possible) and use `bfloat16` for deep learning models; disable gradient checkpointing.
- Implement early stopping with a high max epoch (e.g., 1000 epochs).
- Build modular pipeline: facilitate straightforward pre/post-processing and hyperparameter updates, but keep `{model_name}` fixed.
- Implement logging per validation fold and overall OOF; log only other parts of code if relevant to validation.
- Add a top-level DEBUG flag; pipeline must run twice (DEBUG and FULL modes) and log mode clearly.
- Detect NaN or zero metric/loss after epoch 1 of fold 0 and raise exception if encountered.
- Output predictions/files as dictated by competition rules to the appropriate directory from `BASE_DIR`.

---

**Model Name:**
`{model_name}`

**Example Python Implementation for `{model_name}`**
{example_code}

**Hard Constraints:**
- Use ONLY `{model_name}` (no substitutions or fallback models).
- Deliver a fully-contained, single-file script.
- Use CUDA whenever available.
- Place all `logging.info` statements for validation results only (per fold and overall); only log data loading/setup if directly relevant to validation.
- Place `logging.basicConfig()` at the start of the script.
- Deep learning: always use `bfloat16`, **no** gradient checkpointing. Do not code fallback methods.
- LightGBM (if used): **CPU only**.
- Prohibited: `transformers.Trainer`, `transformers.TrainingArguments`.
- Do not use `try/except` to suppress errors.
- Log final validation results after training.
- Modular pipeline: update preprocessing/postprocessing or hyperparameters, but do not swap out `{model_name}`.
- Prefer pretrained models if available.
- External datasets: may be appended **only** to training set.
- **DEBUG flag**: At the script top, define. Pipeline runs twice: once with `DEBUG=True` (subset of data, e.g., 256 samples, 1 epoch), then with `DEBUG=False` (full config). Log which mode is running.
- **DL Only:** After 1st epoch on fold 0, if metric/loss is NaN or 0, raise Exception to halt.
- Split: 80% train, 20% validation. Max epochs high (e.g., 1000), stop early by monitored metric. **No K-Fold** methods.

---

Before any significant tool call or external library use, state the purpose and minimal inputs required, and validate actions after key steps with a 1-2 line summary. If a step fails (e.g., CUDA unavailable), state the limitation clearly and proceed conservatively where allowed.

**Additional Context**
- **Competition Description:**
  {description}
- **Directory Structure for `{Path('task') / slug}`:**
  {directory_listing}

Set reasoning_effort = medium for this task; technical outputs must be complete but concise. Make code and tool calls terse, and expand documentation or schema notes as needed.

## Output Format
- Produce a single Python script, enclosed in a triple backtick block with the `python` annotation.
- Model task and metric: infer classification/regression and metric from `{description}`; if unclear, use `accuracy` for classification, `rmse` for regression. Log your chosen metric with justification.
- Document schema/assumptions in comments, as it's inferred from available data.
- For output (predictions/`submission.csv`, saved models), save to the directory defined by `BASE_DIR` (see sample below).

### Example Output Block
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
BASE_DIR = "task/{slug}" if not os.getenv('KAGGLE_KERNEL_RUN_TYPE') else "/kaggle/input/{slug}"
# <YOUR CODE>
```
"""


def build_user(
    plan_markdown: str,
    base_dir: str | Path,
    outputs_dir: str | Path,
    log_path: str | Path,
    submission_path: str | Path,
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
    return base