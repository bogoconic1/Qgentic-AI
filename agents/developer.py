import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Optional
import json

from dotenv import load_dotenv
from openai import OpenAI
from sortedcontainers import SortedList
import weave
import wandb

from tools.developer import (
    execute_code,
    search_sota_suggestions,
)
from guardrails.developer import (
    check_logging_basicconfig_order,
    llm_debug_sequence_review,
    llm_leakage_review,
)
from tools.helpers import call_llm_with_retry, _build_directory_listing


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

# NOTE: I am aware that this code will not work if self.is_lower_better is True (to be fixed)
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
        self._load_benchmark_info()

        # File targets
        self.plan_path = self.outputs_dir / "plan.md"
        self.messages = []
        self.latest_submission_path: Optional[Path] = None
        self.benchmark_info: Optional[dict] = None
        self.threshold_directive: str = ""
        self.previous_runs = [("Initial State", float("inf") if self.is_lower_better else float("-inf"), [""])]

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

    @staticmethod
    def _get_grade_report_json(stdout: str) -> Optional[dict]:
        try:
            start = stdout.index("{")
            end = stdout.rindex("}") + 1
            stdout = stdout[start:end]
            logger.debug("stdout JSON snippet: %s", stdout)
            info = json.loads(stdout)
        except json.JSONDecodeError:
            logger.warning("Baseline grading output was not valid JSON: %s", stdout[:500])
            return

        return info

    def _load_benchmark_info(self) -> None:
        self.benchmark_info = None
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

        info = self._get_grade_report_json(stdout)
        self.benchmark_info = info
        self.gold_threshold = info.get("gold_threshold")
        self.is_lower_better = info.get("is_lower_better")

    def _compose_system(self) -> str:
        logger.debug("Composing system prompt for slug=%s", self.slug)
        self.description = _safe_read(str(self.base_dir / "description.md"))
        logger.debug("Description length: %s characters", len(self.description))
        directory_listing = _build_directory_listing(self.base_dir)
        logger.debug(
            "Directory listing prepared for %s (length=%s)", self.base_dir, len(directory_listing)
        )
        return f"""Role: Lead Developer for Machine-Learning Competition Team. Your task is to produce a single, self-contained Python script, specifically targeted at developing a solution for a Kaggle Competition.

Begin with a concise checklist (3-7 bullets) of what you will do; keep items conceptual, not implementation-level.

**Hard Constraints:**
- Deliver a single-file script.
- Utilize CUDA wherever possible.
- Insert detailed `logging.info` statements covering all aspects of training and validation (e.g., losses, learning rates, timings, evaluation metrics). Only log other code sections (such as data loading or config setup) if they're directly relevant to training or validation.
- Place `logging.basicConfig()` at the very start of your code, before any other logging statements.
- Always train with `bfloat16` when using PyTorch, transformers, or other deep learning libraries. Gradient checkpointing must be disabled.
- Do **not** code any fallback methods.
- **Do not** use LightGBM (it's very slow). For gradient boosting, use XGBoost or CatBoost instead.
- **Do not** use `transformers.Trainer` or `transformers.TrainingArguments`.
- **Do not** use `try/except` blocks to bypass exceptions.
- Log the **final validation results** after training.
- Design the pipeline so it is highly customizable (i.e., it's easy to add or swap techniques, models, etc).
- Prefer pretrained models over training from scratch, whenever possible.
- **IMPORTANT:** For deep learning pipelines, if at the end of the 1st epoch of fold 0, the loss or metric is NaN, raise an Exception to stop the run immediately.
- Run your experiments with just validation on fold 0 UNLESS EXPLICITLY INSTRUCTED OTHERWISE.

**Additional Context**
- Competition Description:
  {self.description}
- Directory structure for task/{self.slug}:
  {directory_listing}
- Score to beat:
  {self.gold_threshold}
- Environment that your code will run in:
  A single A100 80GB GPU

**Output Format**
Return Python code only, enclosed in triple backticks with the `python` annotation:
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# <YOUR CODE>
```

Implement the best possible solution for this task, with the goal of maximizing the test metric and surpassing the gold threshold in as few iterations as possible.
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

    @weave.op()
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

    def _log_attempt_score(self, attempt: int, score: Optional[float]) -> None:
        """Send attempt/score metrics to wandb while guarding against logging errors."""
        try:
            wandb.log({"attempt": attempt, "score": score})
            logger.debug("Logged attempt %s with score %s to wandb", attempt, score)
        except Exception:
            logger.exception("Failed to log attempt %s metrics to wandb", attempt)

    @weave.op()
    def run(self, plan_markdown: str, max_tries: int = 50) -> bool:
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

        run_score = 0

        system_prompt = self._compose_system()
        user_prompt = self._build_user_prompt(plan_markdown, version=1)
        self.messages.append({"role": "system", "content": system_prompt})
        self.messages.append({"role": "user", "content": user_prompt})

        for attempt in range(1, max_tries + 1):

            artifact = wandb.Artifact(f'{self.iteration}-{self.slug}', type='files')

            if len(self.messages) > 6:
                self.messages = self.messages[:2] + self.messages[-4:]

            logger.info("Attempt %s/%s for developer run", attempt, max_tries)
            version = attempt
            code = self._generate_code(self.messages)
            code_path = self._write_code(code, version)

            # ---------------------------
            # Pre-exec guardrail checks
            # ---------------------------
            guard_report = {
                "logging_check": {},
                "debug_sequence_check": {},
                "leakage_check": {},
                "decision": "proceed",
            }

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

            # 2 DEBUG sequencing + NaN guard review (LLM)
            if guard_report["decision"] != "block":
                try:
                    debug_json_text = llm_debug_sequence_review(_safe_read(str(code_path)))
                except Exception:
                    logger.exception("DEBUG sequencing guardrail call failed")
                    debug_json_text = '{"severity":"warn","findings":[{"rule_id":"llm_error","snippet":"N/A","rationale":"LLM call failed","suggestion":"Manually ensure NaN raises exceptions."}]}'
                guard_report["debug_sequence_check"] = debug_json_text
                try:
                    parsed = json.loads(debug_json_text)
                    try:
                        logger.info(
                            "Guardrail[debug_seq] v%s: severity=%s, findings=%s",
                            version,
                            parsed.get("severity"),
                            len(parsed.get("findings", [])),
                        )
                    except Exception:
                        logger.debug("Failed to log debug sequencing guardrail status for v%s", version)
                    if parsed.get("severity") == "block":
                        guard_report["decision"] = "block"
                except Exception:
                    logger.warning(
                        "Guardrail[debug_seq] v%s: malformed JSON from reviewer; proceeding as warn",
                        version,
                    )

            # 3 LLM-based leakage review (only if not already blocked)
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
                    serialized = dict(guard_report)
                    notes = []
                    if isinstance(serialized.get("debug_sequence_check"), str):
                        notes.append("debug_sequence_check is raw string if model returned non-JSON")
                    if isinstance(serialized.get("leakage_check"), str):
                        notes.append("leakage_check is raw string if model returned non-JSON")
                    if notes:
                        serialized["_note"] = " | ".join(notes)
                    json.dump(serialized, f)
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
                    raw_debug = guard_report.get("debug_sequence_check", "{}")
                    parsed_debug = json.loads(raw_debug.strip()) if isinstance(raw_debug, str) else (raw_debug or {})
                    if parsed_debug.get("severity") == "block":
                        summary.append(
                            "- Ensure the script runs DEBUG mode before FULL mode and raises an Exception when loss/metric is NaN."
                        )
                        findings = parsed_debug.get("findings", [])
                        if findings:
                            summary.append("\nDEBUG sequencing findings:")
                            for idx, finding in enumerate(findings, start=1):
                                summary.append(
                                    f"{idx}. rule_id={finding.get('rule_id', 'unknown')}\n   - snippet: {finding.get('snippet', '')}\n   - rationale: {finding.get('rationale', '')}\n   - suggestion: {finding.get('suggestion', '')}"
                                )
                except Exception:
                    try:
                        summary.append("- DEBUG sequencing reviewer returned non-JSON content:")
                        summary.append(str(guard_report.get("debug_sequence_check")))
                    except Exception:
                        pass
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

            log_path = self.outputs_dir / f"code_{self.iteration}_v{version}.txt"
            log_content = ""
            try:
                if log_path.exists():
                    log_content = log_path.read_text().strip()
                    logger.debug(
                        "Loaded execution log from %s (length=%s)",
                        log_path,
                        len(log_content),
                    )
            except Exception:
                logger.exception("Failed to read execution log at %s", log_path)

            submission_path = self.outputs_dir / f"submission_{version}.csv"
            code += "\n\n" + log_content[-30000:] # to avoid token limit issues

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
                    if grade_result.returncode != 0:
                        logger.warning(
                            "Grading command returned non-zero exit (%s). stderr=\n%s",
                            grade_result.returncode,
                            (grade_result.stderr or "").strip(),
                        )
                    else:
                        logger.info("Grading command completed successfully for version %s", version)
                        info = self._get_grade_report_json(grade_feedback)
                        run_score = info.get('score')
                        self._log_attempt_score(attempt, run_score)
                        logger.info("Your result on the test set is %s", run_score)
                except Exception as exc:
                    grade_feedback = f"Failed to run grading tool: {exc}"
                    logger.exception("Grading command failed for version %s", version)

                # improvement
                if (run_score < self.previous_runs[-1][1]) == self.is_lower_better:
                    logger.info(
                        "New best score achieved: %s (previous best was %s)",
                        run_score,
                        self.previous_runs[-1][1],
                    )
                    prev_suggestions = ""
                    self.previous_runs.append((code, run_score, []))
                else:
                    # discard
                    logger.info(
                        "No improvement over previous best score of %s; current score is %s",
                        self.previous_runs[-1][1],
                        run_score,
                    )

                    # rollback
                    code = self.previous_runs[-1][0]

                    # prev sota
                    prev_suggestions = "\n------------------------------".join(self.previous_runs[-1][2])

                if len(self.previous_runs[-1][2]) >= 8 and "Please scale up the number of folds in your training." not in self.previous_runs[-1][2]:
                    # we may have pleataued
                    sota_suggestions = "Please scale up the number of folds in your training."
                else:
                    use_sota = True if len(self.previous_runs[-1][2]) >= 4 else False
                    try:
                        sota_suggestions = search_sota_suggestions(self.description, code, prev_suggestions, use_sota)
                    except Exception:
                        logger.exception("Failed to fetch SOTA suggestions for attempt %s", attempt)
                        sota_suggestions = ""

                logger.info("SOTA suggestion: %s", sota_suggestions)

                self.previous_runs[-1][2].append(sota_suggestions)

                next_instr = f"""
                Please modify your code to ONLY incorporate this advice (keep everything else the same):
                {sota_suggestions}

                Remember:
                - write logs to task/{self.slug}/outputs/{self.iteration}/code_{self.iteration}_v{version+1}.txt
                - and produce the next submission at task/{self.slug}/outputs/{self.iteration}/submission_{version+1}.csv."
                """
                self.messages = self.messages[:2]
                self.messages.append({'role': 'assistant', 'content': code})
                self.messages.append({'role': 'user', 'content': next_instr})

            else:
                next_instr = f"""
                Your code FAILED during execution!
                This is the stack trace and advice on how to fix the error:
                {output}

                Please modify your code to fix the error!

                Remember:
                - write logs to task/{self.slug}/outputs/{self.iteration}/code_{self.iteration}_v{version+1}.txt
                - and produce the next submission at task/{self.slug}/outputs/{self.iteration}/submission_{version+1}.csv."
                """
                self.messages.append({'role': 'assistant', 'content': code})
                self.messages.append({'role': 'user', 'content': next_instr})

            logger.info("previous runs count: %s", len(self.previous_runs))
            logger.info("previous runs list: %s", str(self.previous_runs))

            for path in self.outputs_dir.iterdir():
                if path.is_file():
                    artifact.add_file(str(path), overwrite=True)
                else:
                    logger.debug("Skipping non-file path when logging artifact: %s", path)

            artifact.save()

        logger.warning(
            "Developer run exhausted all attempts without creating submission: %s",
            self.outputs_dir / f"submission_{max_tries}.csv",
        )
        return True
