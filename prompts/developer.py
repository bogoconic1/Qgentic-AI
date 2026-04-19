"""Prompt builders for the standalone developer agent.

Pure template-string functions. The agent (`agents/developer.py`) threads the
codegen calls through a single growing message list — these builders just
substitute strings into triple-quoted templates.
"""

from __future__ import annotations

from pathlib import Path


def codegen_system(goal_text: str) -> str:
    return f"""You write a single Python script (`train.py`) that iterates toward the goal stated in `<goal>` below. Each iteration produces one complete script; the previous attempt is visible above in the conversation thread.

<goal>
{goal_text}
</goal>

**Hard Constraints:**
- Place `import logging` and `logging.basicConfig(level=logging.INFO, ...)` at the very top of the script, BEFORE any third-party imports (torch, transformers, numpy, etc.). Third-party libraries configure logging on import, so basicConfig must come first. A pre-execution guardrail enforces this and will block the script otherwise.
- Use the `logging` module (not `print`) for all observability. Log: what the script is doing at each major step, sizes / shapes / counts of anything loaded, intermediate quantities that can go wrong (thresholds, class weights, normalizations), final results, total runtime, and any errors caught.
- Do not use `try/except` to suppress errors. Let exceptions propagate so they appear in the log.
- Treat any constraints listed in `<goal>` as inviolable.

## Output
- Return a single Python script inside one ```python ... ``` fenced block. No prose outside it.
- Start the file with a docstring containing your reasoning, the approach, and what changed since the previous version (if any).
- Save every artifact the script produces inside the version directory printed in the user message.
- The script must be self-contained and runnable as `python train.py`. stdout/stderr are captured automatically into `train.txt` next to the script.
"""


def initial_codegen_user(version_dir: Path) -> str:
    return f"""<version_dir>{version_dir}</version_dir>

This is iteration 1. Write the first attempt from scratch.

Output one ```python ... ``` fenced block. Save all artifacts under {version_dir}/.
"""


def next_codegen_user(output: str, version_dir: Path) -> str:
    return f"""<execution_output>
{output}
</execution_output>

<version_dir>{version_dir}</version_dir>

Evolve your previous attempt based on the execution output above. Output one ```python ... ``` fenced block. Save all artifacts under {version_dir}/.
"""
