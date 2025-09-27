import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Optional
import json

from dotenv import load_dotenv
from openai import OpenAI
from project_config import get_config
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


_CONFIG = get_config()
_LLM_CFG = _CONFIG.get("llm", {}) if isinstance(_CONFIG, dict) else {}
_PATH_CFG = _CONFIG.get("paths", {}) if isinstance(_CONFIG, dict) else {}
_RUNTIME_CFG = _CONFIG.get("runtime", {}) if isinstance(_CONFIG, dict) else {}
_HARDWARE_CFG = _CONFIG.get("hardware", {}) if isinstance(_CONFIG, dict) else {}
_GUARDRAIL_CFG = _CONFIG.get("guardrails", {}) if isinstance(_CONFIG, dict) else {}

_ENABLE_LOGGING_GUARD = bool(_GUARDRAIL_CFG.get("logging_basicconfig_order", True))
_ENABLE_NAN_GUARD = bool(_GUARDRAIL_CFG.get("nan_guard", True))
_ENABLE_LEAKAGE_GUARD = bool(_GUARDRAIL_CFG.get("leakage_review", True))

_BASE_URL = _LLM_CFG.get("base_url", "https://openrouter.ai/api/v1")
_API_KEY_ENV = _LLM_CFG.get("api_key_env", "OPENROUTER_API_KEY")
_ONLINE_MODEL = _LLM_CFG.get("online_model", "openai/gpt-5:online")

_TASK_ROOT = Path(_PATH_CFG.get("task_root", "task"))
_OUTPUTS_DIRNAME = _PATH_CFG.get("outputs_dirname", "outputs")
_CODE_TEMPLATE = _PATH_CFG.get("code_filename_template", "code_{iteration}_v{version}.py")
_LOG_TEMPLATE = _PATH_CFG.get("log_filename_template", "code_{iteration}_v{version}.txt")
_SUBMISSION_TEMPLATE = _PATH_CFG.get("submission_filename_template", "submission_{version}.csv")
_HARDWARE_DESCRIPTION = _HARDWARE_CFG.get("description", "A single A100 80GB GPU")
_DEFAULT_MAX_TRIES = _RUNTIME_CFG.get("developer_max_tries", 50)

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
      <task_root>/<slug>/<outputs_dir>/<iteration>/submission.csv
    - Ensures Torch MPS fallback flags for Apple Silicon
    """

    def __init__(self, slug: str, iteration: int):
        load_dotenv()
        self.slug = slug
        self.iteration = iteration

        self.task_root = _TASK_ROOT
        self.outputs_dirname = _OUTPUTS_DIRNAME
        self.base_dir = self.task_root / slug
        self.outputs_dir = self.base_dir / self.outputs_dirname / str(iteration)
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
        self.client = OpenAI(api_key=os.environ.get(_API_KEY_ENV), base_url=_BASE_URL)
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

    def _code_filename(self, version: int) -> str:
        return _CODE_TEMPLATE.format(iteration=self.iteration, version=version)

    def _log_filename(self, version: int) -> str:
        return _LOG_TEMPLATE.format(iteration=self.iteration, version=version)

    def _submission_filename(self, version: int) -> str:
        return _SUBMISSION_TEMPLATE.format(iteration=self.iteration, version=version)

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
- **Do not** use `transformers.Trainer` or `transformers.TrainingArguments`.
- **Do not** use `try/except` blocks to bypass exceptions.
- Log the **final validation results** after training.
- Design the pipeline so it is highly customizable (i.e., it's easy to add or swap techniques, models, etc).
- Prefer pretrained models over training from scratch, whenever possible.
- **IMPORTANT:** At the very top, add a `DEBUG` flag. The pipeline must run sequentially twice: once with `DEBUG=True` (using a small subset of data, e.g., 256 samples and 1 epoch, but otherwise unchanged) and then once with `DEBUG=False` (using the full training config). Clearly log when the script is in DEBUG or FULL mode.
- **IMPORTANT:** For deep learning pipelines, if at the end of the 1st epoch of fold 0, the loss or metric is NaN, raise an Exception to stop the run immediately.

**Additional Context**
- Competition Description:
  {self.description}
- Directory structure for {self.base_dir}:
  {directory_listing}
- Score to beat:
  {self.gold_threshold}

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
        base_dir_display = self.base_dir
        outputs_dir_display = self.outputs_dir
        log_path_display = self.outputs_dir / self._log_filename(version)
        submission_path_display = self.outputs_dir / self._submission_filename(version)
        base = f"""
Researcher Technical Plan (Markdown):
{plan_markdown}

Project structure:
- Base data dir: {base_dir_display}
- Outputs dir: {outputs_dir_display}
- The logs should be written to a file named {log_path_display}
- Required output: {submission_path_display}
"""
        base += (
            "\nReturn the complete Python script that, when run, writes logs to "
            f"{log_path_display} "
            "and produces a submission CSV at "
            f"{submission_path_display}."
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
                model=_ONLINE_MODEL,
                messages=messages,
            )
            msg = completion.choices[0].message
            content = msg.content or ""

        logger.info("Model response received for iteration %s", self.iteration)
        logger.debug("Completion content length: %s", len(content))
        return self._extract_code(content)

    def _write_code(self, code: str, version: int) -> Path:
        code_path = self.outputs_dir / self._code_filename(version)
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
    def run(self, plan_markdown: str, max_tries: Optional[int] = None) -> bool:
        max_tries = max_tries or _DEFAULT_MAX_TRIES
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
            log_check = {"status": "skipped", "reason": "disabled in config"}
            if _ENABLE_LOGGING_GUARD:
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
            else:
                guard_report["logging_check"] = log_check
                try:
                    logger.info(
                        "Guardrail[logging] v%s: skipped (disabled in config)",
                        version,
                    )
                except Exception:
                    logger.debug("Failed to log logging guardrail skip status for v%s", version)

            # 2 DEBUG sequencing + NaN guard review (LLM)
            if guard_report["decision"] != "block":
                if _ENABLE_NAN_GUARD:
                    try:
                        debug_json_text = llm_debug_sequence_review(_safe_read(str(code_path)))
                    except Exception:
                        logger.exception("DEBUG sequencing guardrail call failed")
                        debug_json_text = '{"severity":"warn","findings":[{"rule_id":"llm_error","snippet":"N/A","rationale":"LLM call failed","suggestion":"Manually ensure DEBUG runs before FULL and NaN raises exceptions."}]}'
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
                            logger.debug("Failed to log DEBUG sequencing guardrail status for v%s", version)
                        if parsed.get("severity") == "block":
                            guard_report["decision"] = "block"
                    except Exception:
                        logger.warning(
                            "Guardrail[debug_seq] v%s: malformed JSON from reviewer; proceeding as warn",
                            version,
                        )
                else:
                    guard_report["debug_sequence_check"] = {
                        "status": "skipped",
                        "reason": "disabled in config",
                    }
                    try:
                        logger.info(
                            "Guardrail[debug_seq] v%s: skipped (disabled in config)",
                            version,
                        )
                    except Exception:
                        logger.debug("Failed to log DEBUG sequencing guardrail skip status for v%s", version)

            # 3 LLM-based leakage review (only if not already blocked)
            if guard_report["decision"] != "block":
                if _ENABLE_LEAKAGE_GUARD:
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
                        logger.warning(
                            "Guardrail[leakage] v%s: malformed JSON from reviewer; proceeding as warn",
                            version,
                        )
                else:
                    guard_report["leakage_check"] = {
                        "status": "skipped",
                        "reason": "disabled in config",
                    }
                    try:
                        logger.info(
                            "Guardrail[leakage] v%s: skipped (disabled in config)",
                            version,
                        )
                    except Exception:
                        logger.debug("Failed to log leakage guardrail skip status for v%s", version)

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
                next_log_path = self.outputs_dir / self._log_filename(version + 1)
                next_submission_path = self.outputs_dir / self._submission_filename(version + 1)
                fix_instr = (
                    "\nPlease regenerate the script addressing the above guardrail issues. "
                    f"Write logs to {next_log_path} "
                    f"and produce {next_submission_path}."
                )
                self.messages.append({"role": "user", "content": "\n".join(summary) + fix_instr})
                logger.info("User prompt with guardrail feedback: %s", "\n".join(summary) + fix_instr)
                # continue to next attempt without execution
                continue

            # Execute the code (inherits current env with MPS flags)
            output = execute_code(str(code_path))
            logger.info("Execution output captured for version v%s", version)
            logger.debug("Execution output: %s", output)

            log_path = self.outputs_dir / self._log_filename(version)
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

            submission_path = self.outputs_dir / self._submission_filename(version)
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
                    prev_suggestions = "\n-".join(self.previous_runs[-1][2])

                #if len(self.previous_runs[-1][2]) >= 5:
                    #self.messages[0]['content'] = self.messages[0]['content'].replace("- Run your experiments with just validation on fold 0 UNLESS EXPLICITLY INSTRUCTED OTHERWISE.", "")
                #else:
                try:
                    sota_suggestions = search_sota_suggestions(self.description, code, prev_suggestions)
                except Exception:
                    logger.exception("Failed to fetch SOTA suggestions for attempt %s", attempt)
                    sota_suggestions = ""

                logger.info("SOTA suggestion: %s", sota_suggestions)

                # extract json
                sota_suggestion_short_summary = ""
                try:
                    m = re.search(r"```json\s*(\{.*?\})\s*```", sota_suggestions, re.DOTALL | re.IGNORECASE)
                    if m:
                        json_text = m.group(1)
                        logger.debug("Extracted JSON snippet from SOTA suggestions: %s", json_text)
                        parsed = json.loads(json_text)
                        sota_suggestion_short_summary = parsed.get("suggestion", "")
                    else: 
                        logger.debug("No JSON snippet found in SOTA suggestions; using full text.")
                except Exception:
                    logger.exception("Failed to parse JSON from SOTA suggestions; using full text.")


                self.previous_runs[-1][2].append(sota_suggestion_short_summary)
                logger.info("Summary of SOTA suggestion: %s", sota_suggestion_short_summary)

                next_log_path = self.outputs_dir / self._log_filename(version + 1)
                next_submission_path = self.outputs_dir / self._submission_filename(version + 1)
                next_instr = f"""
                Please modify your code to ONLY incorporate this advice (keep everything else the same):
                {sota_suggestions}

                Remember:
                - write logs to {next_log_path}
                - and produce the next submission at {next_submission_path}"
                """
                self.messages = self.messages[:2]
                self.messages.append({'role': 'assistant', 'content': code})
                self.messages.append({'role': 'user', 'content': next_instr})

            else:
                next_log_path = self.outputs_dir / self._log_filename(version + 1)
                next_submission_path = self.outputs_dir / self._submission_filename(version + 1)
                next_instr = f"""
                Your code FAILED during execution!
                This is the stack trace and advice on how to fix the error:
                {output}

                Please modify your code to fix the error!

                Remember:
                - write logs to {next_log_path}
                - and produce the next submission at {next_submission_path}"
                """
                self.messages.append({'role': 'assistant', 'content': code})
                self.messages.append({'role': 'user', 'content': next_instr})

            logger.info("previous runs count: %s", len(self.previous_runs))

            for path in self.outputs_dir.iterdir():
                if path.is_file():
                    artifact.add_file(str(path), overwrite=True)
                else:
                    logger.debug("Skipping non-file path when logging artifact: %s", path)

            artifact.save()

        logger.warning(
            "Developer run exhausted all attempts without creating submission: %s",
            self.outputs_dir / self._submission_filename(max_tries),
        )
        return True