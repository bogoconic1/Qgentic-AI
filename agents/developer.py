import os
import re
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from tools.developer import execute_code
from tools.helpers import call_llm_with_retry


def _safe_read(path: str) -> str:
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception:
        return ""


class DeveloperAgent:
    """Developer/Coder

    Consumes the Researcher plan, generates runnable Python code that writes
    a submission.csv to task/<slug>/outputs/<iteration>/, and iterates until success.
    """

    def __init__(self, slug: str, iteration: int):
        load_dotenv()
        self.slug = slug
        self.iteration = int(iteration)
        os.environ["TASK_SLUG"] = slug
        self.outputs_dir = os.path.join("task", slug, "outputs", str(self.iteration))
        os.makedirs(self.outputs_dir, exist_ok=True)
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.code_try = 1

    def _read_docs(self) -> str:
        base_dir = os.path.join("task", self.slug)
        overview = _safe_read(os.path.join(base_dir, "overview.md"))
        data_description = _safe_read(os.path.join(base_dir, "data_description.md"))
        return f"Overview:\n{overview}\n\nData Description:\n{data_description}"

    def _extract_code_from_text(self, text: str) -> str:
        # Prefer ```python fenced blocks
        match = re.search(r"```python\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # Fallback to any fenced block
        match_any = re.search(r"```\s*(.*?)```", text, flags=re.DOTALL)
        if match_any:
            return match_any.group(1).strip()
        # As last resort, return the whole text
        return text.strip()

    def _current_code_path(self) -> str:
        return os.path.join(self.outputs_dir, f"code_{self.iteration}_v{self.code_try}.py")

    def _submission_path(self) -> str:
        return os.path.join(self.outputs_dir, "submission.csv")

    def _compose_system(self) -> str:
        base_dir = os.path.join("task", self.slug)
        overview = _safe_read(os.path.join(base_dir, "overview.md"))
        data_description = _safe_read(os.path.join(base_dir, "data_description.md"))
        return f"""
You are an experienced Kaggle Competitions Grandmaster with 10 years of coding experience in Python. You have top-notch coding ability in machine learning and data science pipelines.

Competition Overview:
{overview}

Data Description:
{data_description}

Your code must adhere to these guidelines
- Begin with importing os and defining DATA_DIR as follows:
    ```python
    import os
    DATA_DIR = '/kaggle/input/{self.slug}' if os.getenv('KAGGLE_KERNEL_RUN_TYPE') else os.path.join('task', '{self.slug}')
    ```
- End with 
    ```python
    submission.to_csv("submission.csv", index=False)
    ```     
- You will be given a plan/instruction to implement. Do not deviate from the plan.
- YOU MUST ONLY implement one code block within ```python backticks. Do not write any other text outside the code block.
"""

    def _generate_code_text(self, plan_markdown: str) -> str:
        system_prompt = self._compose_system()
        self.messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Instruction: {plan_markdown}\n\n"
            },
        ]
        completion = call_llm_with_retry(
            self.client,
            model="gpt-5",
            messages=self.messages,
            max_completion_tokens=16384,
        )
        print(completion)
        message = completion.choices[0].message
        return message.content or ""

    def _write_code(self, code_text: str) -> str:
        code_only = self._extract_code_from_text(code_text)
        path = self._current_code_path()
        with open(path, "w") as f:
            f.write(code_only + "\n")
        return path

    def run(self, plan_markdown: str, max_tries: int = 3) -> bool:
        previous_feedback: Optional[str] = None
        while self.code_try <= max_tries:
            print("--"*50)
            print(f"\n[Try {self.code_try}/{max_tries}]")
            raw = self._generate_code_text(plan_markdown, previous_feedback)
            code_path = self._write_code(raw)

            output = execute_code(code_path)
            submission_path = self._submission_path()

            if os.path.exists(submission_path) and os.path.getsize(submission_path) > 0:
                return True

            # Prepare feedback and iterate
            if not output:
                output = "No output captured. Submission not found."
            if not os.path.exists(submission_path):
                output += f"\nSubmission not found at: {submission_path}"

            previous_feedback = output
            self.code_try += 1

        return False

