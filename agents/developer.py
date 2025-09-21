import json
import logging
import os
import re
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
        self.submission_path = self.outputs_dir / "submission.csv"
        self.plan_path = self.outputs_dir / "plan.md"
        self.messages = []

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

    def _compose_system(self) -> str:
        logger.debug("Composing system prompt for slug=%s", self.slug)
        description = _safe_read(str(self.base_dir / "description.md"))
        logger.debug("Description length: %s characters", len(description))
        return f"""
You are a expert Python developer with 10 years of experience. Produce a single, self-contained Python script that reads data only from task/{self.slug} and writes a submission to task/{self.slug}/outputs/{self.iteration}/submission.csv.

Hard constraints:
- Single file script.
- No network access; do not download anything.
- Use CUDA everywhere where possible.
- Write logging.info statements everywhere where possible in your code. 
- Always train with fp16 if using PyTorch/transformers or deep learning.
- Do not code any fallback methods.
- Do not try to bypass any potential exceptions by writing your code in try/except blocks.
- Make sure you log the final validation results.
- Always write the final CSV to the exact path above.

Environment context:
{description}

Deliver only Python. If using code fences, use ```python.
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
- Required output: task/{self.slug}/outputs/{self.iteration}/submission.csv
"""
        base += "\nReturn the complete Python script that, when run, writes the submission.csv."
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

            if self.submission_path.exists():
                logger.info(
                    "Submission detected at %s after attempt %s", self.submission_path, attempt
                )
                feedback = log_content or "Submission generated successfully."
                self.messages.append({'role': 'assistant', 'content': code})
                self.messages.append({'role': 'user', 
                                      'content': feedback + f"\nAbove are the logs. Please study them and try to think of ways to improve the validation score. In your new code - The logs should be written to a file named task/{self.slug}/outputs/{self.iteration}/code_{self.iteration}_v{version+1}.txt"})
            else:
                feedback = log_content + output
                self.messages.append({'role': 'assistant', 'content': code})
                self.messages.append({'role': 'user', 'content': feedback + f"In your new code - The logs should be written to a file named task/{self.slug}/outputs/{self.iteration}/code_{self.iteration}_v{version+1}.txt"})

        logger.warning(
            "Developer run exhausted all attempts without creating submission: %s",
            self.submission_path,
        )
        return True
