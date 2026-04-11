"""Prompt builders for the standalone goal-mode agent.

Five pure template-string functions. The agent (`agents/goal_developer.py`)
threads the codegen calls through a single growing message list — these
builders just substitute strings into triple-quoted templates and have no
knowledge of history, truncation, or schemas.
"""

from __future__ import annotations

from pathlib import Path

from schemas.goal_developer import GoalReview


def codegen_system(goal_text: str) -> str:
    return f"""You write a single Python script (`train.py`) that iterates toward the goal stated in `<goal>` below. Each iteration produces one complete script; the previous attempt is visible above in the conversation thread, along with the reviewer's verdict.

<goal>
{goal_text}
</goal>

**Hard Constraints:**
- Place `import logging` and `logging.basicConfig(level=logging.INFO, ...)` at the very top of the script, BEFORE any third-party imports (torch, transformers, numpy, etc.). Third-party libraries configure logging on import, so basicConfig must come first. A pre-execution guardrail enforces this and will block the script otherwise.
- Use the `logging` module (not `print`) for all observability. Log: what the script is doing at each major step, sizes / shapes / counts of anything loaded, intermediate quantities that can go wrong (thresholds, class weights, normalizations), final results, total runtime, and any errors caught.
- Do not use `try/except` to suppress errors. Let exceptions propagate so they appear in the log and the reviewer can act on them.
- Treat any constraints listed in `<goal>` as inviolable. The reviewer will flag violations and the loop will refuse to record the version as progress.

## Output
- Return a single Python script inside one ```python ... ``` fenced block. No prose outside it.
- Start the file with a docstring containing your reasoning, the approach, and what changed since the previous version (if any).
- Save every artifact the script produces inside the version directory printed in the user message.
- The script must be self-contained and runnable as `python train.py`. stdout/stderr are captured automatically into `train.txt` next to the script.

When the user message includes a previous review with a `next_step`, evolve your previous attempt in that direction. The previous code is visible in the conversation thread above.
"""


def initial_codegen_user(version_dir: Path) -> str:
    return f"""<version_dir>{version_dir}</version_dir>

This is iteration 1. Write the first attempt from scratch.

Output one ```python ... ``` fenced block. Save all artifacts under {version_dir}/.
"""


def next_codegen_user(output: str, review: GoalReview, version_dir: Path) -> str:
    return f"""<execution_output>
{output}
</execution_output>

<review>
reasoning: {review.reasoning}
is_valid: {review.is_valid}
violations: {review.violations}
score: {review.score}
done: {review.done}
next_step: {review.next_step}
</review>

<version_dir>{version_dir}</version_dir>

Apply the reviewer's `next_step` to your previous attempt. Output one ```python ... ``` fenced block. Save all artifacts under {version_dir}/.
"""


def review_system() -> str:
    return """You are reviewing one execution of a candidate Python script against a stated goal. Return a structured GoalReview JSON.

## Rules
- Be strict about violations of any hard constraints listed in `<goal>`. Examples: forbidden imports, forbidden function calls, reading files the goal said not to read, exceeding stated budgets.
- The `score` field is a lower-is-better progress signal toward the goal. Choose whatever scalar best reflects "distance to done" for this goal (e.g. worst per-tensor relative_err for a clone task, 1 - accuracy for a classifier, mean L2 error for a regression). Use the same metric consistently across iterations so progress is comparable.
- Set `done=True` ONLY if the goal is fully achieved. Partial progress is not done.
- `next_step` should be the single most impactful change to try next. Use web search to find better approaches, techniques, or library APIs that could help — don't rely solely on what you already know. If `done=True`, leave `next_step` empty.
- `is_valid` is True iff the run completed without an unhandled exception and produced the artifacts the script claims to have produced.
"""


def review_user(goal_text: str, code: str, output: str) -> str:
    return f"""<goal>
{goal_text}
</goal>

<candidate_code>
```python
{code}
```
</candidate_code>

<execution_output>
{output}
</execution_output>

Return a GoalReview JSON with: reasoning, is_valid, violations, score, done, next_step.
"""
