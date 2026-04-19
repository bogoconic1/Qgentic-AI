import logging
import math
import os
import shutil
from pathlib import Path
import json

from dotenv import load_dotenv
from project_config import get_config, get_instructions
import weave
import wandb

from tools.developer import (
    _build_resource_header,
    execute_code,
    execute_with_monitor,
    _LOG_MONITOR_INTERVAL,
    _BASELINE_CODE_TIMEOUT,
)
from utils.guardrails import evaluate_guardrails, build_block_summary
from tools.helpers import call_llm, _build_directory_listing
from google.genai import types
from tools.explore import explore_codebase
from utils.llm_utils import append_message, get_developer_tools
from utils.checkpoint import (
    create_db as _create_checkpoint_db,
    save_checkpoint,
    load_latest_checkpoint,
)
from utils.code_utils import strip_header_from_code, extract_python_code
from prompts.developer_agent import (
    build_system as prompt_build_system,
    build_user as prompt_build_user,
)


_CONFIG = get_config()
_LLM_CFG = _CONFIG["llm"]
_PATH_CFG = _CONFIG["paths"]
_RUNTIME_CFG = _CONFIG["runtime"]
_GUARDRAIL_CFG = _CONFIG["guardrails"]

_ENABLE_LOGGING_GUARD = bool(_GUARDRAIL_CFG["logging_basicconfig_order"])
_ENABLE_LEAKAGE_GUARD = bool(_GUARDRAIL_CFG["leakage_review"])
_ENABLE_CODE_SAFETY = bool(_GUARDRAIL_CFG["enable_code_safety"])


_DEVELOPER_MODEL = _LLM_CFG["developer_model"]
_HITL_INSTRUCTIONS = get_instructions()["# Developer Instructions"]

_TASK_ROOT = Path(_PATH_CFG["task_root"])


class DeveloperAgent:
    """Turns the Researcher plan into a runnable training script.

    - Generates train.py in version folder: task/{slug}/{run_id}/{iteration}/{version}/train.py
    - Executes it and iterates while within a time budget
    - Success condition: writes submission.csv at task/{slug}/{run_id}/{iteration}/{version}/submission.csv
    """

    def __init__(
        self,
        slug: str,
        run_id: str,
        iteration: int | str,
        external_data_listing: str | None = None,
        plan_content: str | None = None,
    ):
        load_dotenv()
        self.slug = slug
        self.run_id = run_id
        self.iteration = iteration  # Per-strategy suffix (e.g. "1", "2", "3")

        self.task_root = _TASK_ROOT
        self.base_dir = self.task_root / slug

        # Fail fast if required helper files are missing
        cv_splits_path = self.base_dir / "cv_splits.json"
        metric_path = self.base_dir / "metric.py"
        if not cv_splits_path.exists():
            raise FileNotFoundError(f"Required file missing: {cv_splits_path}")
        if not metric_path.exists():
            raise FileNotFoundError(f"Required file missing: {metric_path}")

        self.outputs_dir = self.base_dir / self.run_id / str(iteration)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.developer_log_path = self.outputs_dir / f"developer_{iteration}.txt"
        self._configure_logger()

        # Metric direction and target from config
        _is_lower_better = _RUNTIME_CFG["is_lower_better"]
        if _is_lower_better not in (True, False):
            raise ValueError(
                f"config runtime.is_lower_better must be true or false, got: {_is_lower_better!r}"
            )
        self.is_lower_better: bool = _is_lower_better
        self.gold_threshold: float | None = _RUNTIME_CFG["gold_threshold"]
        self.best_score: float = float("inf") if self.is_lower_better else float("-inf")

        self.previous_runs: list[tuple[str, float]] = []
        self.successful_versions: set[int] = set()
        self.version_scores: dict[int, float] = {}
        self.best_code: str | None = None
        self.best_code_file: str | None = None
        self.best_version: int | None = None
        self.last_successful_version: int | None = None

        self.latest_submission_path: Path | None = None

        self.threshold_directive: str = ""
        self._checkpoint_conn = None

        self.external_data_listing: str = (
            external_data_listing or "No external data directories found."
        )
        self.plan_content: str = plan_content or "No plan.md found."

        # Use parent iteration folder (e.g., outputs/17 for iteration 17_2)
        parent_iteration_folder = self._get_parent_iteration_folder()
        self.external_dir = parent_iteration_folder / "external_data"
        self.external_dir.mkdir(parents=True, exist_ok=True)
        os.environ["EXTERNAL_DATA_DIR"] = str(self.external_dir)

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        self.logger.info(
            "Initialized DeveloperAgent for slug=%s iteration=%s",
            self.slug,
            self.iteration,
        )
        self.logger.debug("Outputs directory resolved to: %s", self.outputs_dir)

    def _configure_logger(self) -> None:
        self.logger = logging.getLogger(f"{__name__}.{self.slug}.{self.iteration}")

        self.logger.handlers = []

        file_handler = logging.FileHandler(self.developer_log_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        )
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.DEBUG)

        # Prevent propagation to parent loggers to avoid duplicate logs
        self.logger.propagate = False

    def _compose_system(self) -> str:
        self.logger.debug("Composing system prompt for slug=%s", self.slug)
        with open(self.base_dir / "description.md", "r") as f:
            self.description = f.read()
        self.logger.debug("Description length: %s characters", len(self.description))
        directory_listing = _build_directory_listing(self.base_dir)
        self.logger.debug(
            "Directory listing prepared for %s (length=%s)",
            self.base_dir,
            len(directory_listing),
        )
        return prompt_build_system(
            description=self.description,
            directory_listing=directory_listing,
            slug=self.slug,
            hitl_instructions=_HITL_INSTRUCTIONS,
        )

    def _build_user_prompt(self, version: int) -> str:
        self.logger.debug("Building user prompt")
        base_dir_display = self.base_dir
        outputs_dir_display = self.outputs_dir
        log_path_display = self.outputs_dir / f"{version}/train.txt"
        submission_path_display = self.outputs_dir / f"{version}/submission.csv"
        return prompt_build_user(
            base_dir=base_dir_display,
            outputs_dir=outputs_dir_display,
            log_path=log_path_display,
            submission_path=submission_path_display,
            threshold_directive=self.threshold_directive,
            version=version,
        )

    @weave.op()
    def _generate_code(
        self, instructions: str, messages: list[dict[str, str]], version: int
    ) -> tuple[list[dict], str, int | None]:
        """Generate code via markdown-fenced output.

        The model can call explore_codebase to read project source files and
        installed package source, or execute_python to run snippets in a
        subprocess, before producing its final code output. Up to 100 tool
        rounds are allowed before the model must emit code.

        Returns:
            Tuple of (message_history, train_py_code, input_tokens)
        """
        self.logger.info(
            "Requesting code generation from model for iteration %s", self.iteration
        )

        tools = get_developer_tools()
        max_tool_steps = 100
        input_tokens = None

        for step in range(max_tool_steps + 1):
            is_last_step = step == max_tool_steps

            response, step_tokens = call_llm(
                model=_DEVELOPER_MODEL,
                system_instruction=instructions,
                messages=messages,
                function_declarations=tools if not is_last_step else [],
                enable_google_search=True,
                include_usage=True,
            )
            input_tokens = step_tokens

            parts = response.candidates[0].content.parts
            has_function_calls = any(
                part.function_call
                for part in parts
                if hasattr(part, "function_call")
            )

            if not has_function_calls:
                self.logger.info(
                    "Model response received for iteration %s (step %d)",
                    self.iteration,
                    step + 1,
                )
                raw_content = response.text
                code = extract_python_code(raw_content)
                if not code:
                    raise ValueError(
                        "No ```python code block found in model response"
                    )

                content = f"```python\n{code}\n```"
                output = [append_message("assistant", content)]

                if input_tokens:
                    self.logger.info("API call used %d input tokens", input_tokens)

                return output, code, input_tokens

            # Execute tool calls (explore_codebase / execute_python)
            function_responses = []
            for call_idx, part in enumerate(parts, start=1):
                if hasattr(part, "function_call") and part.function_call:
                    result_str = self._execute_developer_tool_call(
                        part.function_call,
                        version=version,
                        step=step + 1,
                        call_idx=call_idx,
                    )
                    function_responses.append(
                        types.Part.from_function_response(
                            name=part.function_call.name,
                            response={"result": result_str},
                        )
                    )

            messages.append(response.candidates[0].content.model_dump(mode="json", exclude_none=True))
            if function_responses:
                messages.append(
                    types.Content(role="function", parts=function_responses).model_dump(mode="json", exclude_none=True)
                )

        # Should not reach here (last step has no tools so model must produce text)
        raise RuntimeError("Code generation exhausted tool steps without producing code")

    def _execute_developer_tool_call(
        self,
        function_call,
        *,
        version: int,
        step: int,
        call_idx: int,
    ) -> str:
        """Dispatch a tool call made during code generation."""
        args = dict(function_call.args)

        if function_call.name == "explore_codebase":
            query = args["query"]
            self.logger.info("explore_codebase called: %s", query[:100])
            return explore_codebase(query)

        if function_call.name == "execute_python":
            code = args["code"]
            timeout = args.get("timeout_seconds", 300)
            self.logger.info(
                "execute_python called (v%s step %s call %s, %d bytes, timeout=%ds)",
                version,
                step,
                call_idx,
                len(code),
                timeout,
            )
            work_dir = self.outputs_dir / str(version) / "codegen_snippets"
            work_dir.mkdir(parents=True, exist_ok=True)
            script_file = work_dir / f"{step}_{call_idx}.py"
            script_file.write_text(_build_resource_header() + code)
            job = execute_code(str(script_file), timeout_seconds=timeout)
            return json.dumps({"output": job.result()})

        return json.dumps({"error": f"Unknown tool: {function_call.name}"})

    def _postprocess_code(self, code: str) -> tuple[str, int]:
        """Insert BASE_DIR setup at the top of generated code.

        Returns:
            Tuple of (postprocessed_code, num_header_lines_added)
        """
        lines = [
            "import os",
            f'BASE_DIR = "task/{self.slug}" if not os.getenv(\'KAGGLE_KERNEL_RUN_TYPE\') else "/kaggle/input/{self.slug}"',
            "",
        ]
        header = "\n".join(lines)
        return header + "\n" + code, len(lines)

    def _get_parent_iteration_folder(self) -> Path:
        """Get parent run folder for shared per-run artifacts (external_data, etc.).

        Returns:
            Path to the run container (e.g., task/csiro-biomass/<run_id>/)
        """
        return self.base_dir / self.run_id

    def _write_code(self, train_py: str, version: int) -> Path:
        """Write train.py and supporting files to a version folder.

        Args:
            train_py: Python training script source code
            version: Version number

        Returns:
            Path to the version folder
        """
        version_folder = self.outputs_dir / str(version)
        version_folder.mkdir(parents=True, exist_ok=True)
        self.logger.info("Writing code to version folder: %s", version_folder)
        postprocessed_code, num_header_lines = self._postprocess_code(train_py)
        train_path = version_folder / "train.py"
        train_path.write_text(postprocessed_code)
        self.logger.debug("Written train.py (%d characters)", len(postprocessed_code))

        metadata_path = version_folder / "train.json"
        metadata_path.write_text(json.dumps({"num_header_lines": num_header_lines}))
        self.logger.debug("Written train.json with %d header lines", num_header_lines)

        # These files are REQUIRED - throw error if not found
        task_root = self.base_dir

        cv_splits_src = task_root / "cv_splits.json"
        if not cv_splits_src.exists():
            raise FileNotFoundError(
                f"cv_splits.json not found in task root: {cv_splits_src}. This file is required."
            )
        cv_splits_dst = version_folder / "cv_splits.json"
        shutil.copy2(cv_splits_src, cv_splits_dst)
        self.logger.debug("Copied cv_splits.json from %s", cv_splits_src)

        metric_src = task_root / "metric.py"
        if not metric_src.exists():
            raise FileNotFoundError(
                f"metric.py not found in task root: {metric_src}. This file is required."
            )
        metric_dst = version_folder / "metric.py"
        shutil.copy2(metric_src, metric_dst)
        self.logger.debug("Copied metric.py from %s", metric_src)

        return version_folder

    def _log_attempt_score(self, attempt: int, score: float | None) -> None:
        """Send attempt/score metrics to wandb while guarding against logging errors."""
        try:
            wandb.log({"attempt": attempt, "score": score})
            self.logger.debug(
                "Logged attempt %s with score %s to wandb", attempt, score
            )
        except Exception:
            self.logger.exception("Failed to log attempt %s metrics to wandb", attempt)

    def _is_improvement(self, score: float | None, best_score: float) -> bool:
        """Return True when the provided score beats the current best."""
        if score is None:
            return False
        if math.isnan(score):
            return False

        if self.is_lower_better:
            if math.isinf(best_score):
                return not math.isinf(score)
            return score < best_score

        if math.isinf(best_score):
            return not math.isinf(score)
        return score > best_score

    @staticmethod
    def _format_score_value(value: float | None) -> str:
        """Format score values for human-readable logging/messages."""
        if value is None:
            return "N/A"
        if math.isnan(value) or math.isinf(value):
            return "N/A"
        return f"{value}"

    def _execute_code(self, code_path: Path, version: int) -> str:
        """Execute code with LLM-based log monitoring.

        Thin wrapper around ``tools.developer.execute_with_monitor`` so the
        same launch/monitor loop is shared between the existing developer
        flow and the standalone goal-mode agent.
        """
        output = execute_with_monitor(
            code_path,
            timeout_seconds=_BASELINE_CODE_TIMEOUT,
            log_monitor_interval=_LOG_MONITOR_INTERVAL,
            logger=self.logger,
        )
        self.logger.debug("Execution output for v%s: %s", version, output)
        return output


    def _evaluate_submission(
        self, code_clean: str, version: int, attempt: int
    ) -> tuple[str, float, int | None, float | None, bool]:
        """Evaluate submission and build enriched code context.

        Args:
            code_clean: Clean code without markdown fences
            version: Version number
            attempt: Attempt number

        Returns:
            Tuple of (code_context, run_score, previous_successful_version, base_score, submission_exists)
            - code_context: Code with score and analysis
            - run_score: Score from grading (or inf/-inf if no submission)
            - previous_successful_version: Most recent successful version before this one
            - base_score: Score of the previous successful version
            - submission_exists: True if submission file exists for THIS version
        """
        submission_path = self.outputs_dir / f"{version}/submission.csv"
        code_context = "<code>\n" + code_clean + "\n</code>\n"

        run_score = float("inf") if self.is_lower_better else float("-inf")

        previous_successful_version: int | None = None
        base_score: float | None = None
        submission_exists = submission_path.exists()

        if submission_exists:
            self.latest_submission_path = submission_path
            self.successful_versions.add(version)
            self.logger.info(
                "Submission detected at %s after attempt %s (marking v%s as successful)",
                submission_path,
                attempt,
                version,
            )
            previous_successful_version = (
                self.last_successful_version
            )  # Initialize before try

            self.logger.info("Reading validation score from train_stats.json")
            try:
                version_folder = self.outputs_dir / str(version)
                train_stats_path = version_folder / "train_stats.json"

                if train_stats_path.exists():
                    with open(train_stats_path, "r") as f:
                        train_stats = json.load(f)

                    # Extract cv_worst as the validation score (worst fold score)
                    run_score = train_stats.get("cv_worst")

                    if run_score is not None:
                        self._log_attempt_score(attempt, run_score)
                        self.logger.info(
                            "Your validation score (cv_worst) is %s", run_score
                        )

                        self.version_scores[version] = run_score
                        self.last_successful_version = version
                    else:
                        self.logger.warning(
                            "cv_worst not found in train_stats.json for version %s",
                            version,
                        )
                else:
                    self.logger.warning(
                        "train_stats.json not found for version %s", version
                    )
            except Exception as exc:
                self.logger.exception("Failed to read train_stats.json: %s", exc)

            code_context += (
                f"<validation_score>\n{run_score}\n</validation_score>\n"
            )

            if previous_successful_version is not None:
                base_version = previous_successful_version
                base_score = self.version_scores[base_version]
            else:
                # This is the first version with a score, no comparison base
                base_version = None
                base_score = None

            run_score_display = self._format_score_value(run_score)
            improvement = (
                self._is_improvement(run_score, base_score)
                if base_score is not None
                else False
            )

            # Always check and update global best, regardless of base comparison
            if self._is_improvement(run_score, self.best_score):
                self.best_score = run_score
                self.best_version = version
                self.best_code = code_clean
                self.best_code_file = f"{version}/train.py"
                self.logger.info(
                    "New global best achieved: %s (version %s)", run_score, version
                )

            if improvement:
                self.logger.info(
                    "Score improved from base v%s: %s -> %s (global best: %s)",
                    base_version,
                    base_score,
                    run_score,
                    self.best_score,
                )
            else:
                self.logger.info(
                    "No improvement from base v%s: %s (current score: %s, global best: %s)",
                    base_version,
                    base_score,
                    run_score,
                    self.best_score,
                )

            analysis_msg = f"The current score is {run_score_display}."

            if base_score is not None:
                base_score_display = self._format_score_value(base_score)
                analysis_msg += f" Your score before implementing this suggestion was {base_score_display}."

            if self.gold_threshold is not None and run_score is not None:
                if self.is_lower_better:
                    if run_score <= self.gold_threshold:
                        analysis_msg += f" You have reached/exceeded the TARGET of {self.gold_threshold}! Focus on further incremental improvements."
                    else:
                        gap = run_score - self.gold_threshold
                        analysis_msg += f" Current gap to TARGET ({self.gold_threshold}): {gap:.6f}. Let's close this gap."
                else:
                    if run_score >= self.gold_threshold:
                        analysis_msg += f" You have reached/exceeded the TARGET of {self.gold_threshold}! Focus on further incremental improvements."
                    else:
                        gap = self.gold_threshold - run_score
                        analysis_msg += f" Current gap to TARGET ({self.gold_threshold}): {gap:.6f}. Let's close this gap."
            elif run_score is None:
                analysis_msg += " The submission failed to produce a valid score. Please review the error logs and fix any issues."
            else:
                analysis_msg += " Let's push further to reach an even stronger result."

            code_context += f"<analysis>\n{analysis_msg}\n</analysis>\n"
            self.previous_runs.append((code_clean, run_score))

        return (
            code_context,
            run_score,
            previous_successful_version,
            base_score,
            submission_exists,
        )

    def _get_checkpoint_db(self):
        """Open (or return cached) checkpoint database connection."""
        if self._checkpoint_conn is None:
            self._checkpoint_conn = _create_checkpoint_db()
        return self._checkpoint_conn

    def _save_checkpoint(self, version, input_list, last_input_tokens):
        """Save a per-version checkpoint to SQLite after step 5 (evaluate submission)."""
        best_score = self.best_score
        if best_score is not None and math.isinf(best_score):
            best_score = None

        state = {
            "best_score": best_score,
            "best_version": self.best_version,
            "best_code_file": self.best_code_file,
            "version_scores": {str(k): v for k, v in self.version_scores.items()},
            "successful_versions": list(self.successful_versions),
            "input_list": input_list,
            "last_input_tokens": last_input_tokens,
        }
        conn = self._get_checkpoint_db()
        save_checkpoint(conn, self.slug, str(self.iteration), version, state)
        self.logger.info("Checkpoint saved for v%s", version)

    def _load_latest_checkpoint(self):
        """Load the most recent checkpoint and restore instance state.

        Returns dict with run() locals (input_list, last_input_tokens, version)
        or None if no checkpoint exists.
        """
        conn = self._get_checkpoint_db()
        row = load_latest_checkpoint(conn, self.slug, str(self.iteration))
        if row is None:
            return None

        best_score = row["best_score"]
        if best_score is None:
            best_score = float("inf") if self.is_lower_better else float("-inf")
        self.best_score = best_score

        self.best_version = row["best_version"]
        self.best_code_file = row["best_code_file"]
        self.version_scores = {int(k): v for k, v in row["version_scores"].items()}
        self.successful_versions = set(row["successful_versions"])

        self.previous_runs = [(None, None)] * len(self.version_scores)
        if self.version_scores:
            self.last_successful_version = max(self.version_scores.keys())
        if self.best_version and self.best_code_file:
            train_path = self.outputs_dir / self.best_code_file
            if train_path.exists():
                self.best_code = strip_header_from_code(train_path)

        return {
            "input_list": row["input_list"],
            "last_input_tokens": row["last_input_tokens"],
            "version": row["version"],
        }

    def run(
        self, max_time_seconds: int = 6 * 3600
    ) -> tuple[float, str | None]:
        """Run a single developer version (v1): generate → guardrails → execute → evaluate.

        The iterative feedback loop (red_flags + SOTA) has been removed. The caller
        drives multi-iteration behavior externally (see PR 228 for the new
        iteration-review + re-research loop).
        """
        self.logger.info(
            "Starting developer run for slug=%s iteration=%s",
            self.slug,
            self.iteration,
        )

        system_prompt = self._compose_system()
        version = 1
        user_prompt = self._build_user_prompt(version=version)
        input_list = [append_message("user", user_prompt)]

        try:
            response_output, train_py, last_input_tokens = self._generate_code(
                instructions=system_prompt, messages=input_list, version=version
            )
        except (ValueError, RuntimeError) as e:
            self.logger.warning("Code generation failed: %s", e)
            return self.best_score, self.best_code_file

        input_list += response_output
        version_folder = self._write_code(train_py, version)

        train_py_path = version_folder / "train.py"
        with open(str(train_py_path), "r") as f:
            code_text = f.read()
        code_clean = strip_header_from_code(train_py_path)

        guard_report = evaluate_guardrails(
            code_text=code_text,
            enable_logging_guard=_ENABLE_LOGGING_GUARD,
            enable_leakage_guard=_ENABLE_LEAKAGE_GUARD,
            enable_code_safety=_ENABLE_CODE_SAFETY,
        )
        self.logger.info(
            "Guardrail decision v%s: %s", version, guard_report["decision"]
        )
        if guard_report["decision"] == "block":
            self.logger.warning(
                "Guardrails blocked v%s: %s", version, build_block_summary(guard_report)
            )
            return self.best_score, self.best_code_file

        self._execute_code(train_py_path, version)
        self._evaluate_submission(code_clean, version, version)
        self._save_checkpoint(version, input_list, last_input_tokens)

        artifact = wandb.Artifact(f"{self.iteration}-{self.slug}", type="files")
        allowed_extensions = {".py", ".txt", ".csv"}
        for path in self.outputs_dir.iterdir():
            if path.is_file() and path.suffix.lower() in allowed_extensions:
                artifact.add_file(str(path), overwrite=True)
        try:
            artifact.save()
        except Exception as e:
            self.logger.exception("Failed to save wandb artifact: %s", e)

        return self.best_score, self.best_code_file
