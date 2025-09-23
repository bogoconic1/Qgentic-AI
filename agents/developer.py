import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Optional
import json

from dotenv import load_dotenv
from openai import OpenAI

from tools.developer import (
    execute_code,
    search_sota_suggestions,
)
from guardrails.developer import (
    check_logging_basicconfig_order,
    llm_leakage_review,
)
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
        self.benchmark_info: Optional[dict] = None
        self.threshold_directive: str = ""

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

    def _load_benchmark_info(self) -> None:
        self.benchmark_info = None
        self.threshold_directive = ""
        sample_submission = self.base_dir / "sample_submission.csv"
        if not sample_submission.exists():
            logger.warning(
                "Sample submission not found at %s; skipping baseline grading",
                sample_submission,
            )
            return

        grade_cmd = [
            "mlebench",
            "grade-sample",
            str(sample_submission),
            self.slug,
        ]

        logger.info("Fetching grading baseline via: %s", " ".join(grade_cmd))
        try:
            result = subprocess.run(
                grade_cmd,
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception:
            logger.exception("Failed to execute mlebench grade-sample for baseline")
            return

        stdout = (result.stdout or "").strip()
        if result.returncode != 0:
            logger.warning(
                "Baseline grading command returned non-zero exit (%s). stderr=\n%s",
                result.returncode,
                (result.stderr or "").strip(),
            )
            return

        try:
            start = stdout.index("{")
            end = stdout.rindex("}") + 1
            stdout = stdout[start:end]
            logger.debug("stdout JSON snippet: %s", stdout)
            info = json.loads(stdout)
        except json.JSONDecodeError:
            logger.warning("Baseline grading output was not valid JSON: %s", stdout[:500])
            return

        self.benchmark_info = info
        median = info.get("median_threshold")
        is_lower_better = info.get("is_lower_better")

        if median is None or is_lower_better is None:
            logger.debug(
                "Insufficient benchmark data to build threshold directive: %s",
                info,
            )
            return

        comparator = "greater than" if is_lower_better else "less than"
        self.threshold_directive = (
            "Remember: you must include this in your code. Performance baseline: After the first validation fold, if the score is "
            f"{comparator} {median}, break out of the loop after fold 0 instead of completing all folds. For deep learning pipelines, if at the end of the 1st epoch of fold 0, the loss or metric is NaN, also break out of the loop. If you are using more than 1 model, the stopping condition should be computed based on the ensembled result, not the individual model result. Also add a logging message which says CRITICAL: the performance is too weak and you need to investigate."
        )
        logger.info(
            "Baseline grading info cached: median=%s, is_lower_better=%s",
            median,
            is_lower_better,
        )
        logger.info("Threshold directive: %s", self.threshold_directive)

    def _compose_system(self) -> str:
        logger.debug("Composing system prompt for slug=%s", self.slug)
        self.description = _safe_read(str(self.base_dir / "description.md"))
        logger.debug("Description length: %s characters", len(self.description))
        directory_listing = self._build_directory_listing()
        logger.debug(
            "Directory listing prepared for %s (length=%s)", self.base_dir, len(directory_listing)
        )
        threshold_text = f"\n- {self.threshold_directive}" if self.threshold_directive else ""
        return f"""
You are a expert Python developer with 10 years of experience. Produce a single, self-contained Python script.

Hard constraints:
- Single file script.
- Use CUDA everywhere where possible.
- Write logging.info statements everywhere where possible in your code. 
- MAKE SURE you call logging.basicConfig() at the beginning of your code before any other logging statements.
- Always train with bfloat16 if using PyTorch/transformers or deep learning.
- Do not code any fallback methods.
- Do not use LightGBM as it is very slow.
- Do not use transformers.Trainer or transformers.TrainingArguments.
- Do not try to bypass any potential exceptions by writing your code in try/except blocks.
- Make sure you log the final validation results.
- You should make your pipeline as customizable as possible (i.e. easy to add new techniques, models, etc).
- If possible, DO NOT train from scratch. Use pretrained models.
- NOTE: your code will be run on an H100 SXM5 80GB GPU.
{threshold_text}

Environment context:
{self.description}

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
        if self.threshold_directive:
            base += f"\n{self.threshold_directive}"
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
        
        content = ""
        while content == "":
            completion = call_llm_with_retry(
                self.client,
                model="openai/gpt-5:online",
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

        self._load_benchmark_info()
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

            # ---------------------------
            # Pre-exec guardrail checks
            # ---------------------------
            guard_report = {"logging_check": {}, "leakage_check": {}, "decision": "proceed"}

            # 1 AST logging.basicConfig order
            try:
                log_check = check_logging_basicconfig_order(str(code_path))
            except Exception:
                logger.exception("Logging AST check failed")
                log_check = {"status": "error", "error": "exception in logging check"}
            guard_report["logging_check"] = log_check
            try:
                logger.info(
                    "Guardrail[logging] v%s: status=%s, violations=%s, basic_line=%s",
                    version,
                    log_check.get("status"),
                    len(log_check.get("violations", [])),
                    log_check.get("basicConfig_line"),
                )
            except Exception:
                logger.debug("Failed to log logging guardrail status for v%s", version)
            if log_check.get("status") == "fail":
                guard_report["decision"] = "block"

            # 2 LLM-based leakage review (only if not already blocked)
            if guard_report["decision"] != "block":
                try:
                    leakage_json_text = llm_leakage_review(self.description, _safe_read(str(code_path)))
                except Exception:
                    logger.exception("LLM leakage review call failed")
                    leakage_json_text = '{"severity":"warn","findings":[{"rule_id":"llm_error","snippet":"N/A","rationale":"LLM call failed","suggestion":"Proceed with caution"}]}'
                guard_report["leakage_check"] = leakage_json_text
                try:
                    parsed = json.loads(leakage_json_text)
                    try:
                        logger.info(
                            "Guardrail[leakage] v%s: severity=%s, findings=%s",
                            version,
                            parsed.get("severity"),
                            len(parsed.get("findings", [])),
                        )
                    except Exception:
                        logger.debug("Failed to log leakage guardrail status for v%s", version)
                    if parsed.get("severity") == "block":
                        guard_report["decision"] = "block"
                except Exception:
                    # If JSON malformed, treat as warn but proceed
                    logger.warning("Guardrail[leakage] v%s: malformed JSON from reviewer; proceeding as warn", version)

            # Persist guard report
            try:
                with open(self.outputs_dir / f"guardrail_v{version}.json", "w") as f:
                    if isinstance(guard_report.get("leakage_check"), str):
                        # Keep raw string for leakage_check if not JSON parsed
                        json.dump({**guard_report, "_note": "leakage_check is raw string if model returned non-JSON"}, f)
                    else:
                        json.dump(guard_report, f)
            except Exception:
                logger.exception("Failed to write guardrail report for v%s", version)

            try:
                logger.info("Guardrail decision v%s: %s", version, guard_report.get("decision"))
            except Exception:
                logger.debug("Failed to log final guardrail decision for v%s", version)

            if guard_report.get("decision") == "block":
                # Build feedback and ask for a corrected script without executing
                summary = ["Guardrail checks failed:"]
                if log_check.get("status") == "fail":
                    summary.append("- logging.basicConfig must be called before any top-level logging usage.")
                try:
                    raw_leak = guard_report.get("leakage_check", "{}")
                    parsed = json.loads(raw_leak.strip()) if isinstance(raw_leak, str) else (raw_leak or {})
                    sev = parsed.get("severity")
                    if sev == "block":
                        summary.append("- Potential data leakage risks detected. Please fix as suggested.")
                        findings = parsed.get("findings", [])
                        if findings:
                            summary.append("\nLeakage reviewer findings:")
                            for idx, f in enumerate(findings, start=1):
                                rule_id = f.get("rule_id", "unknown")
                                snippet = f.get("snippet", "")
                                rationale = f.get("rationale", "")
                                suggestion = f.get("suggestion", "")
                                summary.append(
                                    f"{idx}. rule_id={rule_id}\n   - snippet: {snippet}\n   - rationale: {rationale}\n   - suggestion: {suggestion}"
                                )
                except Exception:
                    # Could not parse JSON; include raw reviewer text for context
                    try:
                        summary.append("- Data leakage reviewer returned non-JSON content:")
                        summary.append(str(guard_report.get("leakage_check")))
                    except Exception:
                        pass
                fix_instr = (
                    "\nPlease regenerate the script addressing the above guardrail issues. "
                    f"Write logs to task/{self.slug}/outputs/{self.iteration}/code_{self.iteration}_v{version+1}.txt "
                    f"and produce submission_{version+1}.csv."
                )
                self.messages.append({"role": "user", "content": "\n".join(summary) + fix_instr})
                logger.info("User prompt with guardrail feedback: %s", "\n".join(summary) + fix_instr)
                # continue to next attempt without execution
                continue

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

                sota_context = code + "\n\n".join(part for part in feedback_parts if part)
                try:
                    sota_suggestions = search_sota_suggestions(self.description, sota_context)
                except Exception:
                    logger.exception("Failed to fetch SOTA suggestions for attempt %s", attempt)
                    sota_suggestions = ""
                if sota_suggestions:
                    feedback_parts.append("Improvment advices:\n" + sota_suggestions)

                feedback = "\n\n".join(part for part in feedback_parts if part) or "Submission generated successfully."
                next_instr = (
                    f"\nAbove are the logs and advices. You must incorporate the advices in your refined solution. Remember you must make the test metric as high as possible. You want to cross the gold threshold in as few iterations as possible. In the refined solution, write logs to "
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
        return True
