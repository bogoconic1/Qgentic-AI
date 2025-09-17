import os
import re
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from tools.developer import execute_code


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

    def _compose_messages(self, plan_markdown: str, previous_feedback: Optional[str]) -> list:
        docs = self._read_docs()
        system_prompt = (
            "You are a senior Kaggle developer. Based on the competition docs and the \n"
            "Researcher plan, write a single Python script that:\n"
            f"- Saves a submission file to: {self._submission_path()}\n"
            "- Uses only local files under the repository.\n"
            "- Is self-contained and runnable with `python script.py`.\n"
            "- Avoids interactive inputs and external credentials.\n"
            "- Prints progress with clear logs.\n"
            "Return ONLY one fenced python code block."
        )

        user_prompt = (
            f"Competition Docs:\n{docs}\n\n"
            f"Researcher Strategic Plan:\n{plan_markdown}\n\n"
            f"Outputs Directory: {self.outputs_dir}\n"
            f"Submission Path: {self._submission_path()}\n"
            f"Slug: {self.slug}\n"
            "Implement a baseline solution that adheres to the data description and\n"
            "submission format. If target/metric are unclear, implement a safe stub that\n"
            "creates the correct submission schema with placeholder predictions, derived\n"
            "from available IDs per docs.\n\n"
            "Environment-aware data directory requirements:\n"
            "- Define `import os` and compute a `DATA_DIR` variable as follows:\n"
            f"  `DATA_DIR = '/kaggle/input/{self.slug}' if os.getenv('KAGGLE_KERNEL_RUN_TYPE') else os.path.join('task', '{self.slug}')`\n"
            "- Always load input files from `DATA_DIR`.\n"
            f"- Create the outputs directory with `os.makedirs('{self.outputs_dir}', exist_ok=True)` before writing.\n"
            f"- Write the final submission to exactly '{self._submission_path()}'."
        )

        if previous_feedback:
            user_prompt += ("\n\nPrevious Attempt Feedback (errors/output):\n" + previous_feedback[:8000])

        messages = [
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return messages

    def _generate_code_text(self, plan_markdown: str, previous_feedback: Optional[str]) -> str:
        messages = self._compose_messages(plan_markdown, previous_feedback)
        response = self.client.responses.create(
            model="gpt-5",
            input=messages,
        )
        return response.output[0].content[0].text

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


