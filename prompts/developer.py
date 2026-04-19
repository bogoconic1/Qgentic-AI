"""Prompt builders for the developer agent.

The developer subagent produces one `train.py` per attempt. `codegen_system`
is built once per DeveloperAgent.run() call (static per-call context: goal,
idea, previous code); the per-attempt user turns are produced by
`codegen_initial_user` (first attempt) and `codegen_retry_user` (feedback
after a failed attempt).
"""

from __future__ import annotations

from pathlib import Path


def codegen_system(
    idea: str,
    previous_code: str | None = None,
) -> str:
    idea_section = f"\n<idea>\n{idea}\n</idea>\n"

    previous_section = ""
    if previous_code:
        previous_section = (
            "\n<previous_code>\n"
            "The train.py from the last successful developer call — "
            "evolve this script to apply the idea above.\n"
            "```python\n"
            f"{previous_code}\n"
            "```\n"
            "</previous_code>\n"
        )

    return f"""You are a developer producing a single Python script (`train.py`) that implements the task specified by the `<idea>` block below. Output one complete script per attempt; the previous attempt's code and any failure feedback are visible above in the conversation thread.
{idea_section}{previous_section}
## Hard constraints

- Place `import logging` and `logging.basicConfig(level=logging.INFO, ...)` at the very top of the script, BEFORE any third-party imports (torch, transformers, numpy, etc.). Third-party libraries configure logging on import, so basicConfig must come first. A pre-execution guardrail enforces this and will block the script otherwise.
- Use the `logging` module (not `print`) for all observability. Log: what the script is doing at each major step, sizes / shapes / counts of anything loaded, intermediate quantities that can go wrong (thresholds, class weights, normalizations), final results, total runtime, and any errors caught.
- Do not use `try/except` to suppress errors. Let exceptions propagate so they appear in the log.
- **Your `train.py` MUST compute its own score and write `train_stats.json` next to the script — use `Path(__file__).parent / "train_stats.json"` — with at least `{{"score": <float>, ...}}` at end of run.** The framework uses the presence + parseability of `train_stats.json` to decide whether the attempt succeeded. No `train_stats.json` (or a non-finite score) = failed attempt = retry with feedback.
- Write any submission or auxiliary artifacts (e.g. `submission.csv`, `valid_preds.csv`) to `Path(__file__).parent` so they land alongside `train.py`.
- A BASE_DIR constant is prepended by the framework — use `BASE_DIR / "<file>"` to read competition data. Locally this resolves to `task/<slug>/`; on Kaggle it resolves to `/kaggle/input/<slug>/`.

## Output format

- Return a single Python script inside one ```python ... ``` fenced block. No prose outside it.
- Start the file with a docstring containing your reasoning, the approach, and what changed since the previous version (if any).
- The script must be self-contained and runnable as `python train.py`. stdout/stderr are captured automatically into `train.txt` next to the script.
"""


def codegen_initial_user(attempt_dir: Path) -> str:
    return f"""Write the train.py for this attempt.

Output directory (where `train.py`, `train_stats.json`, and any other artifacts will live): `{attempt_dir}`

Output one ```python ... ``` fenced block. Do not emit prose outside the block.
"""


def codegen_retry_user(
    output: str,
    hints: str | None,
    missing_stats: bool,
    attempt_dir: Path,
) -> str:
    sections: list[str] = [
        "<execution_output>",
        output,
        "</execution_output>",
    ]

    if hints:
        sections += [
            "",
            "<hints>",
            hints,
            "</hints>",
        ]

    if missing_stats:
        sections += [
            "",
            "<missing_artifact>",
            "`train_stats.json` was not produced. Your train.py must compute its own score and write it to `Path(__file__).parent / \"train_stats.json\"` with at least `{\"score\": <float>}` at end of run.",
            "</missing_artifact>",
        ]

    sections += [
        "",
        f"Produce a corrected train.py for the next attempt. Output directory: `{attempt_dir}`",
        "",
        "Output one ```python ... ``` fenced block.",
    ]

    return "\n".join(sections)
