import os
import re
import json
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from tools.helpers import call_llm_with_retry
from tools.developer import execute_code


def _safe_read(path: str) -> str:
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception:
        return ""


class DeveloperAgent:
    """Turns the Researcher plan into a runnable single-file solution.

    - Generates a single python file: code_{iteration}_v{version}.py
    - Executes it and iterates on failures up to max_tries
    - Success condition: writes submission.csv at
      task/<slug>/outputs/<iteration>/submission.csv
    - Uses OpenRouter (qwen/qwen3-coder) for code generation
    - Ensures Torch MPS fallback flags for Apple Silicon
    """

    def __init__(self, slug: str, iteration: int):
        load_dotenv()
        self.slug = slug
        self.iteration = iteration

        self.base_dir = Path("task") / slug
        self.outputs_dir = self.base_dir / "outputs" / str(iteration)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        # File targets
        self.submission_path = self.outputs_dir / "submission.csv"
        self.plan_path = self.outputs_dir / "plan.md"

        # OpenRouter client
        self.client = OpenAI(api_key=os.environ.get("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")

        # Encourage safe Torch usage on Apple Silicon
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    def _compose_system(self) -> str:
        description = _safe_read(str(self.base_dir / "description.md"))
        return f"""
You are a meticulous Kaggle Developer. Produce a single, self-contained Python script that reads data only from task/{self.slug} and writes a submission to task/{self.slug}/outputs/{self.iteration}/submission.csv.

Hard constraints:
- Single file script. Name must be code_{self.iteration}_v{{version}}.py
- No network access; do not download anything.
- Support Apple Silicon MPS: select device = 'mps' if available, else CPU. Set torch backends safely.
- Create parent directories as needed.
- Always write the final CSV to the exact path above.
- Keep logging informative but concise.

Environment context:
{description}

Deliver only Python. If using code fences, use ```python.
"""

    def _build_user_prompt(self, plan_markdown: str, prior_error: Optional[str] = None, prior_code: Optional[str] = None) -> str:
        base = f"""
Researcher Technical Plan (Markdown):
{plan_markdown}

Project structure:
- Base data dir: task/{self.slug}
- Outputs dir: task/{self.slug}/outputs/{self.iteration}
- Required output: task/{self.slug}/outputs/{self.iteration}/submission.csv

Implementation requirements:
- Single file script. Ensure deterministic paths using pathlib.
- If PyTorch is used, pick device by:
    device = 'cuda' if torch.cuda.is_available() else 'mps' if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available() else 'cpu'
- If using tokenizers/transformers, set TOKENIZERS_PARALLELISM=false and keep batch sizes modest.
- Do not exceed RAM by loading huge data unnecessarily.
- Print a brief run log with key steps and final submission shape.
"""
        if prior_error:
            base += "\nPrevious run failed. Here is the traceback and analysis to fix:\n" + prior_error[:8000]
        if prior_code:
            base += "\nPrevious script (for reference):\n" + prior_code[:8000]
        base += "\nReturn the complete Python script that, when run, writes the submission.csv."
        return base

    def _extract_code(self, content: str) -> str:
        pattern = r"```python\s*(.*?)\s*```"
        m = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return content.strip()

    def _generate_code(self, plan_markdown: str, prior_error: Optional[str], prior_code: Optional[str]) -> str:
        system_prompt = self._compose_system()
        user_prompt = self._build_user_prompt(plan_markdown, prior_error, prior_code)
        completion = call_llm_with_retry(
            self.client,
            model="qwen/qwen3-coder",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        msg = completion.choices[0].message
        content = msg.content or ""
        return self._extract_code(content)

    def _write_code(self, code: str, version: int) -> Path:
        code_path = self.outputs_dir / f"code_{self.iteration}_v{version}.py"
        with open(code_path, "w") as f:
            f.write(code)
        return code_path

    def run(self, plan_markdown: str, max_tries: int = 3) -> bool:
        try:
            with open(self.plan_path, "w") as f:
                f.write(plan_markdown)
        except Exception:
            pass

        prior_error: Optional[str] = None
        prior_code: Optional[str] = None

        for attempt in range(1, max_tries + 1):
            version = attempt
            code = self._generate_code(plan_markdown, prior_error, prior_code)
            code_path = self._write_code(code, version)

            # Execute the code (inherits current env with MPS flags)
            output = execute_code(str(code_path))

            # Save run log
            try:
                with open(self.outputs_dir / f"run_v{version}.log", "w") as f:
                    f.write(output or "")
            except Exception:
                pass

            # Success condition
            if self.submission_path.exists():
                return True

            # Prepare for next attempt
            prior_error = output
            prior_code = code

        return False

