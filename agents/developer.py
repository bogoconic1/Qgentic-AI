import difflib
import logging
import math
import os
import re
import subprocess
import textwrap
import time
from pathlib import Path
from typing import Optional
import json

from dotenv import load_dotenv
from openai import OpenAI
from project_config import get_config
import weave
import wandb

from tools.developer import (
    execute_code,
    search_sota_suggestions,
    ablation_summarize_baseline,
    ablation_summarize_batch,
)
from guardrails.developer import (
    check_logging_basicconfig_order,
    llm_debug_sequence_review,
    llm_leakage_review,
)
from utils.guardrails import evaluate_guardrails, build_block_summary
from tools.helpers import call_llm_with_retry, _build_directory_listing
from utils.diffs import (
    extract_diff_block,
    normalize_diff_payload,
    apply_patch as util_apply_patch,
)
from utils.grade import run_grade, parse_grade_output
from prompts.developer_agent import (
    build_system as prompt_build_system,
    build_user as prompt_build_user,
    patch_mode_directive as prompt_patch_mode_directive,
    guardrail_fix_suffix as prompt_guardrail_fix_suffix,
    execution_failure_suffix as prompt_execution_failure_suffix,
)


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

_PATCH_MODE_ENABLED = bool(_RUNTIME_CFG.get("patch_mode_enabled", False))

_BASE_URL = _LLM_CFG.get("base_url", "https://openrouter.ai/api/v1")
_API_KEY_ENV = _LLM_CFG.get("api_key_env", "OPENROUTER_API_KEY")
_DEVELOPER_MODEL = _LLM_CFG.get("developer_model", "google/gemini-2.5-pro")

_TASK_ROOT = Path(_PATH_CFG.get("task_root", "task"))
_OUTPUTS_DIRNAME = _PATH_CFG.get("outputs_dirname", "outputs")
_CODE_TEMPLATE = "code_v{attempt}_{suggestion_type}_{sub_attempt}.py"
_LOG_TEMPLATE = "code_v{attempt}_{suggestion_type}_{sub_attempt}.txt"
_SUBMISSION_TEMPLATE = "submission_v{attempt}_{suggestion_type}_{sub_attempt}.csv"
_HARDWARE_DESCRIPTION = _HARDWARE_CFG.get("description", "A single A100 80GB GPU")

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

    - Generates a single python file: code_v{attempt}_{suggestion_type}_{sub_attempt}.py
    - Executes it and iterates while within a time budget
    - Success condition: writes submission.csv at
      <task_root>/<slug>/<outputs_dir>/<iteration>/submission.csv
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
        # Default suggestion metadata used for filename decoration
        self.current_suggestion_type = "baseline"
        self.current_sub_attempt = 1
        self._configure_logger()

        # Metric-related defaults; overwritten once benchmark info is available
        self.gold_threshold: Optional[float] = None
        self.is_lower_better: bool = False
        self.best_score: float = float("-inf")

        # Iteration state
        self.previous_runs: list[tuple[str, float]] = []
        self.blacklisted_ideas: list[str] = []
        self.last_suggestion: Optional[str] = None
        self.last_suggestion_code: Optional[str] = None
        self.best_code: Optional[str] = None
        self.best_attempt: Optional[int] = None
        self.next_patch_base_version: Optional[int] = None

        self._load_benchmark_info()
        self.best_score = float("inf") if self.is_lower_better else float("-inf")

        # File targets
        self.plan_path = self.outputs_dir / "plan.md"
        self.messages = []
        self.latest_submission_path: Optional[Path] = None
        self.benchmark_info: Optional[dict] = None
        self.threshold_directive: str = ""

        # OpenRouter client
        self.client = OpenAI(api_key=os.environ.get(_API_KEY_ENV), base_url=_BASE_URL)
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        # Patch mode configuration
        self.patch_mode_enabled = _PATCH_MODE_ENABLED

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

    def _code_filename(self, attempt: int, suggestion_type: str, sub_attempt: int) -> str:
        return _CODE_TEMPLATE.format(
            attempt=attempt,
            suggestion_type=suggestion_type,
            sub_attempt=sub_attempt,
        )

    def _log_filename(self, attempt: int, suggestion_type: str, sub_attempt: int) -> str:
        return _LOG_TEMPLATE.format(
            attempt=attempt,
            suggestion_type=suggestion_type,
            sub_attempt=sub_attempt,
        )
        
    def _submission_filename(self, attempt: int, suggestion_type: str, sub_attempt: int) -> str:
        return _SUBMISSION_TEMPLATE.format(
            attempt=attempt,
            suggestion_type=suggestion_type,
            sub_attempt=sub_attempt,
        )

    

    def _load_benchmark_info(self) -> None:
        self.benchmark_info = None
        sample_submission = self.base_dir / "sample_submission.csv"
        if not sample_submission.exists():
            logger.warning(
                "Sample submission not found at %s; skipping baseline grading",
                sample_submission,
            )
            return

        info, stdout, returncode, stderr = run_grade(str(sample_submission), self.slug)
        try:
            logger.info("Baseline grade feedback: %s", stdout)
        except Exception:
            logger.debug("Failed to log baseline grade feedback")
        if returncode != 0:
            logger.warning(
                "Baseline grading command returned non-zero exit (%s). stderr=\n%s",
                returncode,
                stderr,
            )
            return
        self.benchmark_info = info
        self.gold_threshold = info.get("gold_threshold")
        self.is_lower_better = info.get("is_lower_better")
        logger.info("is_lower_better=%s", self.is_lower_better)

        if self.is_lower_better:
            logger.info("is_lower_better=True")
        else:
            logger.info("is_lower_better=False")

    def _compose_system(self) -> str:
        logger.debug("Composing system prompt for slug=%s", self.slug)
        self.description = _safe_read(str(self.base_dir / "description.md"))
        logger.debug("Description length: %s characters", len(self.description))
        directory_listing = _build_directory_listing(self.base_dir)
        logger.debug(
            "Directory listing prepared for %s (length=%s)", self.base_dir, len(directory_listing)
        )
        return prompt_build_system(
            description=self.description,
            directory_listing=directory_listing,
            gold_threshold=self.gold_threshold,
            slug=self.slug,
        )

    def _build_user_prompt(self, plan_markdown: str, attempt: int, suggestion_type: str, sub_attempt: int) -> str:
        logger.debug("Building user prompt")
        base_dir_display = self.base_dir
        outputs_dir_display = self.outputs_dir
        log_path_display = self.outputs_dir / self._log_filename(attempt, suggestion_type, sub_attempt)
        submission_path_display = self.outputs_dir / self._submission_filename(attempt, suggestion_type, sub_attempt)
        return prompt_build_user(
            plan_markdown=plan_markdown,
            base_dir=base_dir_display,
            outputs_dir=outputs_dir_display,
            log_path=log_path_display,
            submission_path=submission_path_display,
            threshold_directive=self.threshold_directive,
        )

    def _extract_code(self, content: str) -> str:
        logger.debug("Extracting code from completion content. Content length: %s", len(content))
        pattern = r"```python\s*(.*?)\s*```"
        m = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if m:
            logger.debug("Python fenced block located in completion output.")
            return m.group(1).strip()
        logger.debug("No fenced block detected; returning raw content.")
        return content.strip()

    @staticmethod
    def _format_with_line_numbers(code: str) -> str:
        lines = code.splitlines()
        return "\n".join(f"{idx:04d}: {line}" for idx, line in enumerate(lines, start=1))

    def _build_assistant_payload(self, code: str, include_line_numbers: bool) -> str:
        if include_line_numbers:
            return (
                "<previous_code_with_line_numbers>\n"
                f"{self._format_with_line_numbers(code)}\n"
                "</previous_code_with_line_numbers>"
            )
        return "<previous_code>\n" + code + "\n</previous_code>"

    @staticmethod
    def _extract_diff_block(content: str) -> str:
        return extract_diff_block(content)

    @weave.op()
    def _generate_code(self, messages: list[dict[str, str]], expect_patch: bool = False) -> str:
        logger.info("Requesting code generation from model for iteration %s", self.iteration)
        content = ""
        while content == "":
            completion = call_llm_with_retry(
                self.client,
                model=_DEVELOPER_MODEL,
                messages=messages,
            )
            try:
                msg = completion.choices[0].message
                content = msg.content or ""
            except Exception:
                msg = ""
                content = ""

        logger.info("Model response received for iteration %s", self.iteration)
        logger.debug("Completion content length: %s", len(content))
        if expect_patch:
            return content.strip()
        return self._extract_code(content)

    @staticmethod
    def _normalize_diff_payload(base_path: Path, diff_text: str) -> Optional[str]:
        return normalize_diff_payload(base_path, diff_text)

    def _apply_patch(self, base_attempt: int, diff_payload: str, target_attempt: int, base_sub_attempt: int, suggestion_type: str, target_sub_attempt: int) -> Optional[str]:
        """Apply diff payload to previous file (explicit base/target sub-attempts)."""
        if base_attempt <= 0:
            logger.warning("Patch requested but base attempt is invalid: %s", base_attempt)
            return None

        # Compute filenames for base and output using provided sub-attempts
        restore_sub = self.current_sub_attempt
        self.current_sub_attempt = base_sub_attempt
        base_filename = self._code_filename(base_attempt, suggestion_type, base_sub_attempt)
        base_path = self.outputs_dir / base_filename
        self.current_sub_attempt = target_sub_attempt
        output_filename = self._code_filename(target_attempt, suggestion_type, target_sub_attempt)
        self.current_sub_attempt = restore_sub

        if not base_path.exists():
            logger.warning("Patch requested but base file does not exist: %s", base_path)
            return None

        diff_text = self._extract_diff_block(diff_payload)
        if not diff_text:
            logger.warning("Patch payload was empty after extraction.")
            return None

        output_path = self.outputs_dir / output_filename
        if output_path.exists():
            try:
                output_path.unlink()
            except Exception:
                logger.exception(
                    "Failed to remove existing output file before applying patch: %s",
                    output_path,
                )
                return None

        attempts: list[tuple[str, str]] = []
        normalized_payload = self._normalize_diff_payload(base_path, diff_text)
        attempts.append(("normalized", normalized_payload))

        for label, payload in attempts:
            logger.debug("Attempting to apply %s diff for attempt %s", label, target_attempt)
            logger.debug("Payload: %s", payload)
            updated_code = util_apply_patch(
                outputs_dir=self.outputs_dir,
                base_filename=base_filename,
                output_filename=output_filename,
                payload=payload,
            )
            if updated_code is not None:
                logger.info(
                    "Successfully applied %s diff to generate %s from base %s",
                    label,
                    output_filename,
                    base_filename,
                )
                return updated_code

        logger.warning("All patch attempts failed for %s", output_filename)
        return None

    # Removed old _apply_patch variant (DRY)

    def _append_patch_directive(self, instruction: str, attempt: int, suggestion_type: str, sub_attempt: int) -> str:
        if not self.patch_mode_enabled:
            return instruction
        instruction = instruction.replace("Please modify your code to fix the error!", "Please write a git diff within ```diff to fix the error!")
        instruction = instruction.replace("Please regenerate the script addressing the above guardrail issues.", "Please write a git diff within ```diff to fix the above issues.")
        base_filename = self._code_filename(attempt, suggestion_type, sub_attempt)
        directive = prompt_patch_mode_directive(base_filename)
        return f"{instruction}\n\n{directive}"

    def _write_code(self, code: str, attempt: int, suggestion_type: str, sub_attempt: int) -> Path:
        code_path = self.outputs_dir / self._code_filename(attempt, suggestion_type, sub_attempt)
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

    def _is_improvement(self, score: Optional[float], best_score: float) -> bool:
        """Return True when the provided score beats the current best."""
        if score is None:
            return False
        try:
            if math.isnan(score):
                return False
        except TypeError:
            return False

        if self.is_lower_better:
            if math.isinf(best_score):
                return not math.isinf(score)
            return score < best_score

        if math.isinf(best_score):
            return not math.isinf(score)
        return score > best_score

    @staticmethod
    def _format_score_value(value: Optional[float]) -> str:
        """Format score values for human-readable logging/messages."""
        if value is None:
            return "N/A"
        try:
            if math.isnan(value) or math.isinf(value):
                return "N/A"
        except TypeError:
            return "N/A"
        return f"{value}"

    def _parse_sota_response(self, raw: str) -> tuple[str, str, list[str]]:
        """Extract new suggestion, code snippet, and list of blacklisted ideas."""
        suggestion_text = ""
        code_snippet = ""
        blacklist_list: list[str] = []

        if not raw:
            return suggestion_text, code_snippet, blacklist_list

        json_blocks = []
        try:
            json_blocks = re.findall(r"```json\s*(\{.*?\})\s*```", raw, re.DOTALL | re.IGNORECASE)
        except Exception:
            logger.debug("Unable to locate JSON blocks in SOTA suggestions output.")

        decision_payload = {}
        suggestion_payload = {}

        if json_blocks:
            try:
                decision_payload = json.loads(json_blocks[0])
            except Exception:
                logger.debug("Failed to parse blacklist decision JSON block.")
        if len(json_blocks) >= 2:
            try:
                suggestion_payload = json.loads(json_blocks[1])
            except Exception:
                logger.debug("Failed to parse suggestion JSON block.")

        try:
            if isinstance(decision_payload.get("blacklist"), list):
                blacklist_list = [str(x).strip() for x in decision_payload.get("blacklist") if str(x).strip()]
        except Exception:
            pass
        suggestion_text = (suggestion_payload.get("suggestion") or "").strip()
        if not suggestion_text:
            suggestion_text = (decision_payload.get("suggestion") or "").strip()

        code_match = re.search(r"```python\s*(.*?)\s*```", raw, re.DOTALL | re.IGNORECASE)
        if code_match:
            code_snippet = code_match.group(1).strip()

        return suggestion_text, code_snippet, blacklist_list

    def _register_blacklist(self, suggestion: str, reason: str | None = None) -> None:
        if not suggestion:
            return
        entry = suggestion
        if reason:
            entry = f"{suggestion} -- {reason}"
        if entry not in self.blacklisted_ideas:
            self.blacklisted_ideas.append(entry)
        if len(self.blacklisted_ideas) > 10:
            self.blacklisted_ideas = self.blacklisted_ideas[-10:]

    @weave.op()
    def run(self, plan_markdown: str, max_time_seconds: Optional[int] = 6 * 3600) -> bool:
        logger.info(
            "Starting developer run for slug=%s iteration=%s",
            self.slug,
            self.iteration,
        )
        start_time = time.time()
        deadline = start_time + (max_time_seconds or 6 * 3600)
        try:
            with open(self.plan_path, "w") as f:
                f.write(plan_markdown)
            logger.debug("Plan markdown persisted to %s", self.plan_path)
        except Exception:
            logger.exception("Failed to persist plan markdown to %s", self.plan_path)

        # Compose initial conversation for baseline
        system_prompt = self._compose_system()
        user_prompt = self._build_user_prompt(plan_markdown, attempt=1, suggestion_type="baseline", sub_attempt=1)
        self.messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

        # Suggestions schedule: start with baseline only
        suggestions: list[tuple[str, Optional[str], Optional[str]]] = [("baseline", None, None)]
        attempt = 0
        baseline_result: dict = {}
        batch_index = 0

        while True:
            now = time.time()
            if max_time_seconds is not None and now >= deadline:
                logger.info("Time budget exhausted (%.2f minutes)", (deadline - start_time) / 60.0)
                break

            attempt += 1
            # attempt index for filenames and patch anchors
            logger.info("Starting attempt(batch) %s with %s suggestion(s)", attempt, len(suggestions))
            artifact = wandb.Artifact(f'{self.iteration}-{self.slug}', type='files')

            # scores for ablation after this batch
            batch_scores: list[dict] = []

            for (suggestion_type, suggestion_text, suggestion_code) in suggestions:
                # Manage time budget
                now = time.time()
                if max_time_seconds is not None and now >= deadline:
                    logger.info("Time budget exhausted before suggestion=%s", suggestion_type)
                    break

                # Set filename decorations for this suggestion
                self.current_suggestion_type = suggestion_type
                self.current_sub_attempt = 0

                graded = False
                while True:
                    now = time.time()
                    if max_time_seconds is not None and now >= deadline:
                        logger.info("Time budget exhausted before sub-attempt for %s", suggestion_type)
                        break

                    self.current_sub_attempt += 1
                    sub_attempt = self.current_sub_attempt
                    logger.info("Attempt %s | %s sub_attempt %s", attempt, suggestion_type, sub_attempt)

                    # Trim conversation if too large
                    if len(self.messages) > 6:
                        self.messages = self.messages[:2] + self.messages[-4:]

                    # Build next instruction if we have a specific suggestion to implement
                    if suggestion_text:
                        next_log_path = self.outputs_dir / self._log_filename(attempt, suggestion_type, sub_attempt)
                        next_submission_path = self.outputs_dir / self._submission_filename(attempt, suggestion_type, sub_attempt)
                        instr = (
                            f"Implement the following idea for this batch: \n<suggestion>\n{suggestion_text}\n</suggestion>\n"
                        )
                        if suggestion_code:
                            instr += "Suggested code snippet:\n```python\n" + suggestion_code + "\n```\n"
                        instr += (
                            f"Remember:\n- write logs to {next_log_path}\n- and produce the next submission at {next_submission_path}"
                        )
                        # Only append patch directive for sub_attempt > 1
                        if self.patch_mode_enabled and sub_attempt > 1:
                            patch_instr = self._append_patch_directive(instr, attempt=attempt, suggestion_type=suggestion_type, sub_attempt=sub_attempt)
                        else:
                            patch_instr = None
                        # Reset conversation to system+user, then add assistant base and user instruction
                        self.messages = self.messages[:2]
                        if self.best_code:
                            base_code_payload = self.best_code
                        else:
                            base_code_payload = ""
                        assistant_payload = self._build_assistant_payload(
                            base_code_payload,
                            include_line_numbers=self.patch_mode_enabled,
                        )
                        self.messages.append({'role': 'assistant', 'content': assistant_payload})
                        self.messages.append({'role': 'user', 'content': patch_instr or instr})

                    # Generate code (patch for sub_attempt>1)
                    expect_patch = self.patch_mode_enabled and sub_attempt > 1
                    generated = self._generate_code(self.messages, expect_patch=expect_patch)
                    if expect_patch:
                        # Apply patch from base sub_attempt = current_sub_attempt-1 to target sub_attempt = current_sub_attempt
                        base_sub_attempt = max(1, self.current_sub_attempt - 1)
                        base_attempt = attempt
                        logger.info(
                            "Applying patch relative to base attempt=%s %s sub_attempt=%s -> target sub_attempt=%s",
                            base_attempt,
                            suggestion_type,
                            base_sub_attempt,
                            self.current_sub_attempt,
                        )
                        code = self._apply_patch(base_attempt, generated, attempt, base_sub_attempt, suggestion_type, self.current_sub_attempt)
                        if code is None:
                            logger.warning("Patch application failed; requesting full script next.")
                            if self.messages and self.messages[-1].get("role") == "user":
                                self.messages[-1]["content"] = instr
                            generated = self._generate_code(self.messages, expect_patch=False)

                    code = generated

                    # Write code and guardrails
                    code_path = self._write_code(code, attempt, suggestion_type, sub_attempt)
                    guard_report = evaluate_guardrails(
                        description=self.description,
                        code_text=_safe_read(str(code_path)),
                        enable_logging_guard=_ENABLE_LOGGING_GUARD,
                        enable_nan_guard=_ENABLE_NAN_GUARD,
                        enable_leakage_guard=_ENABLE_LEAKAGE_GUARD,
                    )
                    if guard_report.get("decision") == "block":
                        summary_text = build_block_summary(guard_report)
                        next_log_path = self.outputs_dir / self._log_filename(attempt, suggestion_type, sub_attempt + 1)
                        next_submission_path = self.outputs_dir / self._submission_filename(attempt, suggestion_type, sub_attempt + 1)
                        fix_instr = prompt_guardrail_fix_suffix(next_log_path, next_submission_path)
                        guardrail_prompt = summary_text + fix_instr
                        guardrail_prompt = self._append_patch_directive(guardrail_prompt, attempt=attempt, suggestion_type=suggestion_type, sub_attempt=sub_attempt)
                        assistant_payload = self._build_assistant_payload(
                            code,
                            include_line_numbers=self.patch_mode_enabled,
                        )
                        self.messages.append({"role": "assistant", "content": assistant_payload})
                        self.messages.append({"role": "user", "content": guardrail_prompt})
                        logger.info("Guardrail blocked; continuing to next sub-attempt for %s", suggestion_type)
                        continue

                    # Execute
                    output = execute_code(str(code_path))
                    log_path = self.outputs_dir / self._log_filename(attempt, suggestion_type, sub_attempt)
                    log_content = ""
                    try:
                        if log_path.exists():
                            log_content = log_path.read_text().strip()
                    except Exception:
                        logger.exception("Failed to read execution log at %s", log_path)

                    submission_path = self.outputs_dir / self._submission_filename(attempt, suggestion_type, sub_attempt)
                    run_score = None
                    if submission_path.exists():
                        self.latest_submission_path = submission_path
                        try:
                            info, grade_feedback, returncode, stderr = run_grade(str(submission_path), self.slug)
                            if returncode == 0:
                                run_score = info.get('score') if info else None
                                self._log_attempt_score(attempt, run_score)
                                logger.info("Score for %s/%s sub_attempt %s: %s", attempt, suggestion_type, sub_attempt, run_score)
                            else:
                                logger.warning("Grader returned non-zero: %s\n%s", returncode, stderr)
                        except Exception as exc:
                            logger.exception("Grading failed: %s", exc)

                    # If graded, record and break out of this suggestion
                    if run_score is not None:
                        graded = True
                        # Track best
                        if self._is_improvement(run_score, self.best_score):
                            self.best_score = run_score
                            self.best_attempt = attempt
                            self.best_code = code
                        # Append ablation row
                        batch_scores.append({
                            "suggestion_type": suggestion_type,
                            "idea": suggestion_text or "baseline",
                            "score": run_score,
                        })
                        break

                    # Otherwise, construct next-instruction for failure path and continue sub-attempts
                    next_log_path = self.outputs_dir / self._log_filename(attempt, suggestion_type, sub_attempt + 1)
                    next_submission_path = self.outputs_dir / self._submission_filename(attempt, suggestion_type, sub_attempt + 1)
                    next_instr = f"""
                    Your code FAILED during execution!
                    This is the stack trace and advice on how to fix the error:
                    {output}

                    {prompt_execution_failure_suffix(next_log_path, next_submission_path)}
                    """
                    # Only add patch directive for sub-attempts > 1
                    if self.patch_mode_enabled and sub_attempt > 1:
                        next_instr = self._append_patch_directive(next_instr, attempt=attempt, suggestion_type=suggestion_type, sub_attempt=sub_attempt)
                    assistant_payload = self._build_assistant_payload(
                        code,
                        include_line_numbers=self.patch_mode_enabled,
                    )
                    self.messages.append({'role': 'assistant', 'content': assistant_payload})
                    self.messages.append({'role': 'user', 'content': next_instr})
                    # continue to next sub_attempt

                # End while sub-attempts
                if not graded:
                    logger.info("Suggestion %s ended without a graded submission", suggestion_type)

                # Log artifacts for this suggestion
                for path in self.outputs_dir.iterdir():
                    if path.is_file():
                        artifact.add_file(str(path), overwrite=True)
                artifact.save()

            # End for suggestions in batch

            # Run ablation
            if attempt == 1:
                # Baseline ablation
                baseline_result = batch_scores[0] if batch_scores else {"suggestion_type": "baseline", "score": None}
                try:
                    summary_text = ablation_summarize_baseline(baseline_result)
                except Exception:
                    logger.exception("Baseline ablation summarize failed")
                    summary_text = ""
                self.latest_ablation_summary = summary_text
                try:
                    ablation_dir = self.outputs_dir / "ablation"
                    ablation_dir.mkdir(parents=True, exist_ok=True)
                    (ablation_dir / "baseline.json").write_text(json.dumps({"summary": summary_text, "baseline": baseline_result}, ensure_ascii=False, indent=2))
                except Exception:
                    logger.exception("Failed to persist baseline ablation summary")
            else:
                try:
                    summary_text = ablation_summarize_batch(baseline_result, batch_scores)
                except Exception:
                    logger.exception("Batch ablation summarize failed")
                    summary_text = ""
                self.latest_ablation_summary = summary_text
                try:
                    batch_index += 1
                    ablation_dir = self.outputs_dir / "ablation"
                    ablation_dir.mkdir(parents=True, exist_ok=True)
                    (ablation_dir / f"batch_{batch_index}.json").write_text(json.dumps({"summary": summary_text, "results": batch_scores}, ensure_ascii=False, indent=2))
                except Exception:
                    logger.exception("Failed to persist batch ablation summary")

            # Prepare next suggestions via SOTA (skip after baseline if no time)
            now = time.time()
            if max_time_seconds is not None and now >= deadline:
                logger.info("Time exhausted after ablation")
                break

            # Collect plans
            plan_texts: list[str] = []
            try:
                if self.plan_path.exists():
                    plan_texts.append(_safe_read(str(self.plan_path)))
            except Exception:
                pass
            try:
                for extra_plan_path in sorted(self.outputs_dir.glob("plan_*.md")):
                    plan_texts.append(_safe_read(str(extra_plan_path)))
            except Exception:
                pass

            # Fetch multi-suggestions
            try:
                raw = search_sota_suggestions(
                    self.description,
                    failed_ideas=self.blacklisted_ideas,
                    plans=plan_texts,
                    ablation_summary=self.latest_ablation_summary if hasattr(self, 'latest_ablation_summary') else None,
                )
            except Exception:
                logger.exception("SOTA search failed; ending run")
                break

            # Parse JSON with four categories
            parsed = None
            try:
                parsed = json.loads(raw)
            except Exception:
                try:
                    m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", raw, re.IGNORECASE)
                    if m:
                        parsed = json.loads(m.group(1))
                except Exception:
                    parsed = None

            next_suggestions: list[tuple[str, Optional[str], Optional[str]]] = []
            if isinstance(parsed, dict):
                for key in ["data", "architecture", "ensembling", "sota"]:
                    block = parsed.get(key)
                    if isinstance(block, dict):
                        idea = block.get("idea") or ""
                        code_snippet = block.get("code") or None
                        next_suggestions.append((key, idea, code_snippet))

                # Register blacklist if present
                bl = parsed.get("blacklist") if isinstance(parsed, dict) else None
                if isinstance(bl, list):
                    for idea in bl:
                        self._register_blacklist(str(idea), None)

            if not next_suggestions:
                logger.info("No valid multi-suggestions parsed; ending run")
                break

            suggestions = next_suggestions

        logger.info("Developer run finished for iteration=%s", self.iteration)
        return True
