import logging
import math
import os
import re
import time
from pathlib import Path
from typing import Optional
import json

from dotenv import load_dotenv
from project_config import get_config
import weave
import wandb

from tools.developer import (
    execute_code,
    search_sota_suggestions,
)
from utils.guardrails import evaluate_guardrails, build_block_summary
from tools.helpers import call_llm_with_retry, _build_directory_listing
from utils.diffs import (
    extract_diff_block,
    normalize_diff_payload,
    apply_patch as util_apply_patch,
)
from utils.grade import run_grade
from prompts.developer_agent import (
    build_system as prompt_build_system,
    build_user as prompt_build_user,
    patch_mode_directive as prompt_patch_mode_directive,
    guardrail_fix_suffix as prompt_guardrail_fix_suffix,
    execution_failure_suffix as prompt_execution_failure_suffix,
)


logger = logging.getLogger(__name__)


_CONFIG = get_config()
_LLM_CFG = _CONFIG.get("llm")
_PATH_CFG = _CONFIG.get("paths")
_RUNTIME_CFG = _CONFIG.get("runtime")
_GUARDRAIL_CFG = _CONFIG.get("guardrails")

_ENABLE_LOGGING_GUARD = bool(_GUARDRAIL_CFG.get("logging_basicconfig_order"))
_ENABLE_NAN_GUARD = bool(_GUARDRAIL_CFG.get("nan_guard"))
_ENABLE_LEAKAGE_GUARD = bool(_GUARDRAIL_CFG.get("leakage_review"))

_PATCH_MODE_ENABLED = bool(_RUNTIME_CFG.get("patch_mode_enabled"))

_DEVELOPER_MODEL = _LLM_CFG.get("developer_model")

_TASK_ROOT = Path(_PATH_CFG.get("task_root"))
_OUTPUTS_DIRNAME = _PATH_CFG.get("outputs_dirname")
_CODE_TEMPLATE = "code_{iteration}_v{version}.py"
_LOG_TEMPLATE = "code_{iteration}_v{version}.txt"
_SUBMISSION_TEMPLATE = "submission_{version}.csv"

class DeveloperAgent:
    """Turns the Researcher plan into a runnable single-file solution.

    - Generates a single python file: code_{iteration}_v{version}.py
    - Executes it and iterates while within a time budget
    - Success condition: writes submission.csv at
      <task_root>/<slug>/<outputs_dir>/<iteration>/submission.csv
    """

    def __init__(self, slug: str, iteration: int, model_name: Optional[str] = None, example_code: Optional[str] = None):
        load_dotenv()
        self.slug = slug
        self.iteration = iteration

        self.task_root = _TASK_ROOT
        self.outputs_dirname = _OUTPUTS_DIRNAME
        self.base_dir = self.task_root / slug
        self.outputs_dir = self.base_dir / self.outputs_dirname / str(iteration)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.developer_log_path = self.outputs_dir / f"developer_{iteration}.txt"
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
        self.best_version: Optional[int] = None
        self.next_patch_base_version: Optional[int] = None

        self._load_benchmark_info()
        self.best_score = float("inf") if self.is_lower_better else float("-inf")

        # File targets
        self.plan_path = self.outputs_dir / "plan.md"
        self.latest_submission_path: Optional[Path] = None
        self.benchmark_info: Optional[dict] = None
        self.threshold_directive: str = ""

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        # Patch mode configuration
        self.patch_mode_enabled = _PATCH_MODE_ENABLED

        # Optional model constraints for the developer system prompt
        self.model_name: Optional[str] = model_name
        self.example_code: Optional[str] = example_code

        assert self.model_name is not None and self.example_code is not None, "Both model_name and example_code must be provided"

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

    def _compose_system(self) -> str:
        logger.debug("Composing system prompt for slug=%s", self.slug)
        self.description = open(self.base_dir / "description.md", "r").read()
        logger.debug("Description length: %s characters", len(self.description))
        directory_listing = _build_directory_listing(self.base_dir)
        logger.debug(
            "Directory listing prepared for %s (length=%s)", self.base_dir, len(directory_listing)
        )
        return prompt_build_system(
            description=self.description,
            directory_listing=directory_listing,
            model_name=self.model_name,
            example_code=self.example_code,
            slug=self.slug,
        )

    def _build_user_prompt(self, plan_markdown: str, version: int) -> str:
        logger.debug("Building user prompt")
        base_dir_display = self.base_dir
        outputs_dir_display = self.outputs_dir
        log_path_display = self.outputs_dir / self._log_filename(version)
        submission_path_display = self.outputs_dir / self._submission_filename(version)
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

    @staticmethod
    def _extract_diff_block(content: str) -> str:
        return extract_diff_block(content)

    @weave.op()
    def _generate_code(self, instructions: str, messages: list[dict[str, str]], expect_patch: bool = False) -> str:
        logger.info("Requesting code generation from model for iteration %s", self.iteration)
        response = call_llm_with_retry(
            model=_DEVELOPER_MODEL,
            instructions=instructions,
            tools=[],
            messages=messages,
            web_search_enabled=True
        )

        content = response.output_text

        logger.info("Model response received for iteration %s", self.iteration)
        logger.debug("Completion content length: %s", len(content))
        if expect_patch:
            return content.strip()
        return response.output, self._extract_code(content)

    @staticmethod
    def _normalize_diff_payload(base_path: Path, diff_text: str) -> Optional[str]:
        return normalize_diff_payload(base_path, diff_text)

    def _apply_patch(self, base_version: int, diff_payload: str, target_version: int) -> Optional[str]:
        """Apply diff payload to the previous source file and return updated code."""
        if base_version <= 0:
            logger.warning("Patch requested but base version is invalid: %s", base_version)
            return None

        base_filename = self._code_filename(base_version)
        base_path = self.outputs_dir / base_filename
        if not base_path.exists():
            logger.warning("Patch requested but base file does not exist: %s", base_path)
            return None

        diff_text = self._extract_diff_block(diff_payload)
        if not diff_text:
            logger.warning("Patch payload was empty after extraction.")
            return None

        output_filename = self._code_filename(target_version)
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
            logger.debug("Attempting to apply %s diff for version %s", label, target_version)
            logger.debug("Payload: %s", payload)
            updated_code = util_apply_patch(
                outputs_dir=self.outputs_dir,
                base_filename=base_filename,
                output_filename=output_filename,
                payload=payload,
            )
            if updated_code is not None:
                logger.info(
                    "Successfully applied %s diff to generate version %s from base %s",
                    label,
                    target_version,
                    base_version,
                )
                return updated_code

        logger.warning("All patch attempts failed for target version %s", target_version)
        return None

    def _append_patch_directive(self, instruction: str, version: int) -> str:
        if not self.patch_mode_enabled:
            return instruction
        instruction = instruction.replace("Please modify your code to fix the error!", "Please write a git diff within ```diff to fix the error!")
        instruction = instruction.replace("Please regenerate the script addressing the above guardrail issues.", "Please write a git diff within ```diff to fix the above issues.")
        base_filename = self._code_filename(version)
        directive = prompt_patch_mode_directive(base_filename)
        return f"{instruction}\n\n{directive}"

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

    def _parse_sota_response(self, raw: str) -> tuple[str, str, bool, str]:
        """Extract new suggestion, code snippet, blacklist decision, and rationale."""
        # ok to fallback if LLM cannot give response
        suggestion_text = ""
        code_snippet = ""
        blacklist_flag = False
        blacklist_reason = ""

        if not raw:
            return suggestion_text, code_snippet, blacklist_flag, blacklist_reason

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

        blacklist_flag = bool(decision_payload.get("blacklist", False))
        blacklist_reason = (decision_payload.get("reason") or "").strip()
        suggestion_text = (suggestion_payload.get("suggestion") or "").strip()
        if not suggestion_text:
            suggestion_text = (decision_payload.get("suggestion") or "").strip()

        code_match = re.search(r"```python\s*(.*?)\s*```", raw, re.DOTALL | re.IGNORECASE)
        if code_match:
            code_snippet = code_match.group(1).strip()

        return suggestion_text, code_snippet, blacklist_flag, blacklist_reason

    def _register_blacklist(self, suggestion: str, reason: str | None = None) -> None:
        if not suggestion:
            return
        entry = suggestion
        if reason:
            entry = f"{suggestion} -- {reason}"
        if entry not in self.blacklisted_ideas:
            self.blacklisted_ideas.append(entry)

    @weave.op()
    def run(self, plan_markdown: str, max_time_seconds: int = 6 * 3600) -> bool:
        logger.info(
            "Starting developer run for slug=%s iteration=%s",
            self.slug,
            self.iteration,
        )
        start_time = time.time()
        deadline = start_time + max_time_seconds
        # you only keep the part within ```plan tags
        start_index = plan_markdown.find("## External Data Recommendations")
        if start_index != -1: plan_markdown = plan_markdown[start_index:].strip()
        try:
            with open(self.plan_path, "w") as f:
                f.write(plan_markdown)
            logger.debug("Plan markdown persisted to %s", self.plan_path)
        except Exception:
            logger.exception("Failed to persist plan markdown to %s", self.plan_path)

        run_score = 0

        system_prompt = self._compose_system()
        user_prompt = self._build_user_prompt(plan_markdown, version=1)
        input_list = [{"role": "user", "content": user_prompt}]
        
        attempt = 0
        while True:
            now = time.time()
            if max_time_seconds is not None and now >= deadline:
                logger.info("Time budget exhausted (%.2f minutes)", (deadline - start_time) / 60.0)
                break

            attempt += 1

            artifact = wandb.Artifact(f'{self.iteration}-{self.slug}', type='files')

            # if len(input_list) > 6:s
            #     input_list = input_list[:1] + input_list[-5:]

            minutes_left = ((deadline - now) / 60.0) if max_time_seconds is not None else float('inf')
            try:
                logger.info("Attempt %s (time left ~%.1f min)", attempt, minutes_left)
            except Exception:
                logger.info("Attempt %s", attempt)
            version = attempt
            expect_patch = self.patch_mode_enabled and attempt > 1
            while True:
                response_output, generated = self._generate_code(instructions=system_prompt, messages=input_list, expect_patch=expect_patch)
                input_list += response_output
                if expect_patch:
                    # Choose the correct base version for patch application.
                    # Prefer an explicitly set base from the previous step; fallback to previous attempt.
                    preferred_base = self.next_patch_base_version if self.next_patch_base_version else (version - 1)
                    base_candidate_path = self.outputs_dir / self._code_filename(preferred_base)
                    if not base_candidate_path.exists():
                        logger.warning(
                            "Configured patch base v%s not found; falling back to previous attempt v%s",
                            preferred_base,
                            version - 1,
                        )
                        preferred_base = version - 1
                    base_version = preferred_base
                    logger.info("Applying patch relative to base v%s -> target v%s", base_version, version)
                    code = self._apply_patch(base_version, generated, version)
                    if code is not None:
                        break
                    logger.warning(
                        "Patch generation failed for attempt %s; requesting full script instead.",
                        attempt,
                    )
                    input_list.append({"role": "user", "content": "Patch application failed. Ignore the diff request and return the complete updated script enclosed within ```python backticks."})
                    expect_patch = False
                    continue
                else:
                    code = generated
                    break

            code_path = self._write_code(code, version)

            # ---------------------------
            # Pre-exec guardrail checks
            # ---------------------------
            guard_report = evaluate_guardrails(
                description=self.description,
                code_text=open(str(code_path), "r").read(),
                enable_logging_guard=_ENABLE_LOGGING_GUARD,
                enable_nan_guard=_ENABLE_NAN_GUARD,
                enable_leakage_guard=_ENABLE_LEAKAGE_GUARD,
            )

            try:
                logger.info("Guardrail decision v%s: %s", version, guard_report.get("decision"))
            except Exception:
                logger.debug("Failed to log final guardrail decision for v%s", version)

            if guard_report.get("decision") == "block":
                # Build feedback and ask for a corrected script without executing
                summary_text = build_block_summary(guard_report)
                next_log_path = self.outputs_dir / self._log_filename(version + 1)
                next_submission_path = self.outputs_dir / self._submission_filename(version + 1)
                fix_instr = prompt_guardrail_fix_suffix(next_log_path, next_submission_path)
                guardrail_prompt = summary_text + fix_instr
                base_version_for_next_patch = version
                guardrail_prompt = self._append_patch_directive(guardrail_prompt, base_version_for_next_patch)
                input_list.append({"role": "user", "content": guardrail_prompt})
                self.next_patch_base_version = base_version_for_next_patch
                logger.info("Next patch will be based on v%s due to guardrail block", base_version_for_next_patch)
                logger.info("User prompt with guardrail feedback: %s", guardrail_prompt)
                # continue to next attempt without execution
                continue

            # Execute the code
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
            code_with_logs = "<code>\n" + code + "\n</code>\n"
            if log_content:
                code_with_logs += f"<validation_log>\n{log_content[-30000:]}\n</validation_log>\n"  # to avoid token limit issues

            run_score = float("inf") if self.is_lower_better else float("-inf")

            if submission_path.exists():
                self.latest_submission_path = submission_path
                logger.info(
                    "Submission detected at %s after attempt %s", submission_path, attempt
                )
                grade_feedback = ""
                try:
                    info, grade_feedback, returncode, stderr = run_grade(str(submission_path), self.slug)
                    try:
                        logger.info("Grade feedback: %s", grade_feedback)
                    except Exception:
                        logger.debug("Failed to log grade feedback for version %s", version)
                    if returncode != 0:
                        logger.warning(
                            "Grading command returned non-zero exit (%s). stderr=\n%s",
                            returncode,
                            stderr,
                        )
                    else:
                        logger.info("Grading command completed successfully for version %s", version)
                        run_score = info.get('score') if info else None
                        self._log_attempt_score(attempt, run_score)
                        logger.info("Your result on the test set is %s", run_score)
                except Exception as exc:
                    grade_feedback = f"Failed to run grading tool: {exc}"
                    logger.exception("Grading command failed for version %s", version)

                code_with_logs += f"<leaderboard_score>\n{run_score}\n</leaderboard_score>\n"

                previous_best = self.best_score
                run_score_display = self._format_score_value(run_score)
                previous_best_display = self._format_score_value(previous_best)
                improvement = self._is_improvement(run_score, previous_best)

                if improvement:
                    logger.info(
                        "New best score achieved: %s (previous best was %s)",
                        run_score,
                        previous_best,
                    )
                    self.best_score = run_score
                    self.best_version = version
                    if self.gold_threshold is not None:
                        target_text = f"Let's push further to reach {self.gold_threshold}."
                    else:
                        target_text = "Let's push further to reach an even stronger result."

                    analysis_msg = f"Nice work! The score has improved to {run_score_display}. {target_text}"
                else:
                    logger.info(
                        "No improvement over previous best score of %s; current score is %s",
                        previous_best,
                        run_score,
                    )
                    if previous_best_display != "N/A" and run_score_display != "N/A":
                        analysis_msg = f"The latest run scored {run_score_display}, but the best remains {previous_best_display}. Please investigate the regression before proceeding."
                    else:
                        analysis_msg = "The latest run did not improve the benchmark. Please investigate the reasons before continuing."

                code_with_logs += f"<analysis>\n{analysis_msg}\n</analysis>\n"
                if improvement:
                    self.best_code = code
                self.previous_runs.append((code, run_score))

                try:
                    # Collect all researcher plans in outputs dir: plan.md, plan_*.md
                    # not an issue: no plan is acceptable
                    plan_texts: list[str] = []
                    try:
                        if self.plan_path.exists():
                            plan_texts.append(open(self.plan_path, "r").read())
                    except Exception:
                        pass
                    try:
                        for extra_plan_path in sorted(self.outputs_dir.glob("plan_*.md")):
                            plan_texts.append(open(extra_plan_path, "r").read())
                    except Exception:
                        pass

                    sota_suggestions = search_sota_suggestions(
                        self.description,
                        code_with_logs,
                        executed_suggestion=self.last_suggestion,
                        failed_to_improve_score=not improvement,
                        failed_ideas=self.blacklisted_ideas,
                        executed_code=self.last_suggestion_code,
                        plans=plan_texts,
                    )
                except Exception:
                    logger.exception("Failed to fetch SOTA suggestions for attempt %s", attempt)
                    sota_suggestions = ""

                logger.info("SOTA suggestion: %s", sota_suggestions)

                suggestion_text, code_snippet, blacklist_flag, blacklist_reason = self._parse_sota_response(sota_suggestions)

                if blacklist_flag and self.last_suggestion:
                    self._register_blacklist(self.last_suggestion, blacklist_reason)
                    logger.info(
                        "Previous suggestion marked as blacklisted: %s (reason: %s)",
                        self.last_suggestion,
                        blacklist_reason or "N/A",
                    )

                if suggestion_text:
                    logger.info("Summary of SOTA suggestion: %s", suggestion_text)
                else:
                    logger.info("SOTA response did not include a new suggestion summary.")

                if suggestion_text:
                    self.last_suggestion = suggestion_text
                elif blacklist_flag:
                    # if suggestion missing but blacklist decision made, reset last suggestion
                    self.last_suggestion = None

                self.last_suggestion_code = code_snippet if code_snippet else None

                next_log_path = self.outputs_dir / self._log_filename(version + 1)
                next_submission_path = self.outputs_dir / self._submission_filename(version + 1)

                suggestion_block = ""
                if suggestion_text:
                    suggestion_block += f"<suggestion>\n{suggestion_text}\n</suggestion>\n"
                else:
                    suggestion_block += "<suggestion>\nNo suggestion provided.\n</suggestion>\n"

                if code_snippet:
                    suggestion_block += "Suggested code snippet:\n```python\n" + code_snippet + "\n```\n"
                else:
                    suggestion_block += "Suggested code snippet: No code provided.\n"

                if improvement:
                    summary_line = "The latest attempt improved the score; refine the approach with the guidance below."
                else:
                    summary_line = "The latest attempt did not improve the score; address the issues flagged below."

                if previous_best_display != "N/A" and run_score_display != "N/A":
                    summary_line += (
                        f" Previous best: {previous_best_display}. Current score: {run_score_display}."
                    )

                next_instr = (
                    f"{summary_line}\n\n"
                    f"{suggestion_block}\n"
                    f"Remember:\n- write logs to {next_log_path}\n- and produce the next submission at {next_submission_path}"
                )

                # Choose consistent base for the next patch: use best when blacklisted, else current version.
                if blacklist_flag and self.best_code:
                    base_version_for_next_patch = self.best_version if self.best_version is not None else version
                else:
                    base_version_for_next_patch = version
                next_instr = self._append_patch_directive(next_instr, base_version_for_next_patch)

                if blacklist_flag and self.best_code:
                    input_list.append({'role': 'user', 'content': 'The previous code has been blacklisted. Here is the best known working version for your reference (please start work from this version): \n' + self.best_code})

                input_list.append({'role': 'user', 'content': next_instr})
                self.next_patch_base_version = base_version_for_next_patch
                logger.info("Next patch will be based on v%s (blacklist=%s)", base_version_for_next_patch, blacklist_flag)

            else:
                next_log_path = self.outputs_dir / self._log_filename(version + 1)
                next_submission_path = self.outputs_dir / self._submission_filename(version + 1)
                next_instr = f"""
                Your code FAILED during execution!
                This is the stack trace and advice on how to fix the error:
                {output}

                {prompt_execution_failure_suffix(next_log_path, next_submission_path)}
                """
                base_version_for_next_patch = version
                next_instr = self._append_patch_directive(next_instr, base_version_for_next_patch)
                input_list.append({'role': 'user', 'content': next_instr})
                self.next_patch_base_version = base_version_for_next_patch
                logger.info("Next patch will be based on v%s due to execution failure", base_version_for_next_patch)

            logger.info("previous runs count: %s", len(self.previous_runs))

            for path in self.outputs_dir.iterdir():
                if path.is_file():
                    artifact.add_file(str(path), overwrite=True)
                else:
                    logger.debug("Skipping non-file path when logging artifact: %s", path)

            artifact.save()

        return self.best_score, self.best_code, self.blacklisted_ideas