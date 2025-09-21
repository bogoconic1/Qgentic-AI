import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from tools.developer import execute_code
from tools.helpers import call_llm_with_retry


logger = logging.getLogger(__name__)


def _safe_read(path: str) -> str:
    try:
        with open(path, "r") as f:
            content = f.read()
            logger.debug("Successfully read file: %s", path)
            return content
    except Exception:
        logger.exception("Failed to read file: %s", path)
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
        self.developer_log_path = self.outputs_dir / "developer.txt"
        self._configure_logger()

        # File targets
        self.plan_path = self.outputs_dir / "plan.md"
        self.messages = []
        self.latest_submission_path: Optional[Path] = None

        # OpenRouter client
        self.client = OpenAI(api_key=os.environ.get("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        logger.info(
            "Initialized DeveloperAgent for slug=%s iteration=%s", self.slug, self.iteration
        )
        logger.debug("Outputs directory resolved to: %s", self.outputs_dir)

    def _configure_logger(self) -> None:
        # Avoid duplicate handlers pointing to same file
        existing_paths = {
            getattr(handler, "baseFilename", None)
            for handler in logger.handlers
        }
        if str(self.developer_log_path) not in existing_paths:
            file_handler = logging.FileHandler(self.developer_log_path)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
            )
            logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)

    def _build_directory_listing(self) -> str:
        lines: list[str] = []
        for current_root, dirs, files in os.walk(self.base_dir):
            dirs[:] = sorted(d for d in dirs if d != "outputs")
            rel_root = os.path.relpath(current_root, self.base_dir)
            depth = 0 if rel_root in (".", "") else rel_root.count(os.sep) + 1
            indent = "    " * depth
            folder_display = "." if rel_root in (".", "") else rel_root
            lines.append(f"{indent}{folder_display}/")
            for name in sorted(files):
                lines.append(f"{indent}    {name}")
        return "\n".join(lines)

    def _compose_system(self) -> str:
        logger.debug("Composing system prompt for slug=%s", self.slug)
        description = _safe_read(str(self.base_dir / "description.md"))
        logger.debug("Description length: %s characters", len(description))
        directory_listing = self._build_directory_listing()
        logger.debug(
            "Directory listing prepared for %s (length=%s)", self.base_dir, len(directory_listing)
        )
        return f"""
You are a expert Python developer with 10 years of experience. Produce a single, self-contained Python script.

Hard constraints:
- Single file script.
- Use CUDA everywhere where possible.
- Write logging.info statements everywhere where possible in your code. 
- Always train with fp16 if using PyTorch/transformers or deep learning.
- Do not code any fallback methods.
- Do not try to bypass any potential exceptions by writing your code in try/except blocks.
- Make sure you log the final validation results.
- You should make your pipeline as customizable as possible (i.e. easy to add new techniques, models, etc).
- If possible, DO NOT train from scratch. Use pretrained models.

Environment context:
{description}

Directory structure for task/{self.slug}:
{directory_listing}

Deliver only Python. Your code should be between ```python backticks, like this:
```python 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
<YOUR CODE>
```
"""

    def _build_user_prompt(self, plan_markdown: str, version: int) -> str:
        logger.debug("Building user prompt")
        base = f"""
Researcher Technical Plan (Markdown):
{plan_markdown}

Project structure:
- Base data dir: task/{self.slug}
- Outputs dir: task/{self.slug}/outputs/{self.iteration}
- The logs should be written to a file named task/{self.slug}/outputs/{self.iteration}/code_{self.iteration}_v{version}.txt
- Required output: task/{self.slug}/outputs/{self.iteration}/submission_{version}.csv
"""
        base += (
            "\nReturn the complete Python script that, when run, writes logs to "
            f"task/{self.slug}/outputs/{self.iteration}/code_{self.iteration}_v{version}.txt "
            "and produces a submission CSV at "
            f"task/{self.slug}/outputs/{self.iteration}/submission_{version}.csv."
        )
        return base

    def _extract_code(self, content: str) -> str:
        logger.debug("Extracting code from completion content. Content length: %s", len(content))
        pattern = r"```python\s*(.*?)\s*```"
        m = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if m:
            logger.debug("Python fenced block located in completion output.")
            return m.group(1).strip()
        logger.debug("No fenced block detected; returning raw content.")
        return content.strip()

    def _generate_code(self, messages: list[dict[str, str]]) -> str:
        logger.info("Requesting code generation from model for iteration %s", self.iteration)
        for msg in messages:
            logger.debug("============================")
            logger.debug("Message role: %s, content length: %s start: %s", msg['role'], len(msg['content']), msg['content'][:100])
            logger.debug("============================")
        
        content = ""
        while content == "":
            completion = call_llm_with_retry(
                self.client,
                model="openai/gpt-5",
                messages=messages,
            )
            msg = completion.choices[0].message
            content = msg.content or ""

        logger.info("Model response received for iteration %s", self.iteration)
        logger.debug("Completion content length: %s", len(content))
        return self._extract_code(content)

    def _write_code(self, code: str, version: int) -> Path:
        code_path = self.outputs_dir / f"code_{self.iteration}_v{version}.py"
        logger.info("Writing generated code to %s", code_path)
        with open(code_path, "w") as f:
            f.write(code)
        logger.debug("Written code size: %s characters", len(code))
        return code_path

    def run(self, plan_markdown: str, max_tries: int = 20) -> bool:
        logger.info(
            "Starting developer run for slug=%s iteration=%s with max_tries=%s",
            self.slug,
            self.iteration,
            max_tries,
        )
        try:
            with open(self.plan_path, "w") as f:
                f.write(plan_markdown)
            logger.debug("Plan markdown persisted to %s", self.plan_path)
        except Exception:
            logger.exception("Failed to persist plan markdown to %s", self.plan_path)

        system_prompt = self._compose_system()
        user_prompt = self._build_user_prompt(plan_markdown, version=1)
        self.messages.append({"role": "system", "content": system_prompt})
        self.messages.append({"role": "user", "content": user_prompt})

        for attempt in range(1, max_tries + 1):
            # keep at most 3 assistant/user messages
            if len(self.messages) > 8:
                self.messages = self.messages[:2] + self.messages[-6:]

            logger.info("Attempt %s/%s for developer run", attempt, max_tries)
            version = attempt
            code = self._generate_code(self.messages)
            code_path = self._write_code(code, version)

            # Execute the code (inherits current env with MPS flags)
            output = execute_code(str(code_path))
            logger.info("Execution output captured for version v%s", version)
            logger.debug("Execution output: %s", output)

            self.messages.append({'role': 'assistant', 'content': code})

            try:
                with open(self.outputs_dir / f"run_v{version}.log", "w") as f:
                    f.write(output or "")
                logger.debug("Run log written for version v%s", version)
            except Exception:
                logger.exception("Failed to write run log for version v%s", version)

            log_path = self.outputs_dir / f"code_{self.iteration}_v{version}.txt"
            log_content = ""
            try:
                if log_path.exists():
                    log_content = log_path.read_text()
                    logger.debug(
                        "Loaded execution log from %s (length=%s)",
                        log_path,
                        len(log_content),
                    )
            except Exception:
                logger.exception("Failed to read execution log at %s", log_path)

            submission_path = self.outputs_dir / f"submission_{version}.csv"
            feedback_parts = []
            if log_content:
                feedback_parts.append("Execution log:\n" + log_content.strip())
            if output:
                feedback_parts.append("Program stdout/stderr:\n" + output.strip())

            if submission_path.exists():
                self.latest_submission_path = submission_path
                logger.info(
                    "Submission detected at %s after attempt %s", submission_path, attempt
                )
                grade_feedback = ""
                try:
                    grade_cmd = [
                        "mlebench",
                        "grade-sample",
                        str(submission_path),
                        self.slug,
                    ]
                    grade_result = subprocess.run(
                        grade_cmd,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    grade_feedback = (grade_result.stdout or "").strip()
                    if grade_result.stderr:
                        if grade_feedback:
                            grade_feedback += "\n"
                        grade_feedback += grade_result.stderr.strip()
                    logger.info("mlebench grade-sample output:\n%s", grade_feedback)
                except Exception as exc:
                    grade_feedback = f"Failed to run grading tool: {exc}"
                    logger.exception("Grading command failed for version %s", version)

                if grade_feedback:
                    feedback_parts.append("Grader report:\n" + grade_feedback)

                feedback = "\n\n".join(part for part in feedback_parts if part) or "Submission generated successfully."
                next_instr = (
                    f"\nAbove are the logs. Please study them and try to think of ways to improve the validation score. In your refined solution, continue writing logs to "
                    f"task/{self.slug}/outputs/{self.iteration}/code_{self.iteration}_v{version+1}.txt "
                    f"and produce the next submission at task/{self.slug}/outputs/{self.iteration}/submission_{version+1}.csv."
                )
                self.messages.append({'role': 'user', 'content': feedback + next_instr})
            else:
                feedback = "\n\n".join(part for part in feedback_parts if part) or "Run did not produce a submission."
                next_instr = (
                    f"\nIn your revised script, ensure logs write to task/{self.slug}/outputs/{self.iteration}/code_{self.iteration}_v{version+1}.txt "
                    f"and output submission_{version+1}.csv in the same directory."
                )
                self.messages.append({'role': 'user', 'content': feedback + next_instr})

        logger.warning(
            "Developer run exhausted all attempts without creating submission: %s",
            self.outputs_dir / f"submission_{max_tries}.csv",
        )
        return False
