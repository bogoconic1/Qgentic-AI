import logging
import math
import os
import re
import shutil
import time
import threading
from pathlib import Path
import json

from dotenv import load_dotenv
from project_config import get_config, get_config_value, get_instructions
import weave
import wandb

from tools.developer import (
    execute_code,
    monitor_logs,
    search_red_flags,
    search_sota_suggestions,
    _LOG_MONITOR_INTERVAL,
    _BASELINE_CODE_TIMEOUT,
)
from utils.guardrails import evaluate_guardrails, build_block_summary
from tools.helpers import call_llm, _build_directory_listing
from utils.llm_utils import append_message
from utils.checkpoint import (
    create_db as _create_checkpoint_db,
    save_checkpoint,
    load_latest_checkpoint,
)
from utils.grade import run_grade
from schemas.developer import CodeGeneration, SOTAResponse
from utils.code_utils import strip_header_from_code, extract_python_code
from prompts.developer_agent import (
    build_system as prompt_build_system,
    build_user as prompt_build_user,
    guardrail_fix_suffix as prompt_guardrail_fix_suffix,
    execution_failure_suffix as prompt_execution_failure_suffix,
)


# Module-level logger (not used by instances to avoid cross-contamination in parallel execution)
_module_logger = logging.getLogger(__name__)


_CONFIG = get_config()
_LLM_CFG = _CONFIG["llm"]
_PATH_CFG = _CONFIG["paths"]
_RUNTIME_CFG = _CONFIG["runtime"]
_GUARDRAIL_CFG = _CONFIG["guardrails"]
_DEVELOPER_CFG = _CONFIG["developer"]

_ENABLE_LOGGING_GUARD = bool(_GUARDRAIL_CFG["logging_basicconfig_order"])
_ENABLE_LEAKAGE_GUARD = bool(_GUARDRAIL_CFG["leakage_review"])
_ENABLE_CODE_SAFETY = bool(_GUARDRAIL_CFG["enable_code_safety"])

_USE_VALIDATION_SCORE = bool(_RUNTIME_CFG["use_validation_score"])

_DEVELOPER_MODEL = _LLM_CFG["developer_model"]
_HITL_INSTRUCTIONS = get_instructions()["# Developer Instructions"]
_HITL_SOTA = bool(_DEVELOPER_CFG["hitl_sota"])

_TASK_ROOT = Path(_PATH_CFG["task_root"])
_OUTPUTS_DIRNAME = _PATH_CFG["outputs_dirname"]


class DeveloperAgent:
    """Turns the Researcher plan into a runnable training script.

    - Generates train.py in version folder: outputs/{iteration}/{version}/train.py
    - Executes it and iterates while within a time budget
    - Success condition: writes submission.csv at outputs/{iteration}/{version}/submission.csv
    """

    # Class-level shared state across all parallel DeveloperAgent instances
    # This enables cross-model learning to avoid duplicate failures
    # Format: "Model <model_name> tried <suggestion> (score improved/worsened/remained by X: A -> B)"
    _shared_suggestions: list[str] = []
    _lock = threading.Lock()

    def __init__(
        self,
        slug: str,
        iteration: int | str,
        model_name: str | None = None,
        model_recommendations: str | None = None,
        later_recommendations: dict | None = None,
        external_data_listing: str | None = None,
        plan_content: str | None = None,
        cpu_core_range: list[int] | None = None,
        gpu_identifier: str | None = None,
        gpu_isolation_mode: str = "none",
        conda_env: str | None = None,
    ):
        load_dotenv()
        self.slug = slug
        self.iteration = (
            iteration  # Can be int (legacy) or str like "1_1" (for parallel baselines)
        )

        self.task_root = _TASK_ROOT
        self.outputs_dirname = _OUTPUTS_DIRNAME
        self.base_dir = self.task_root / slug

        # Fail fast if required helper files are missing
        cv_splits_path = self.base_dir / "cv_splits.json"
        metric_path = self.base_dir / "metric.py"
        if not cv_splits_path.exists():
            raise FileNotFoundError(f"Required file missing: {cv_splits_path}")
        if not metric_path.exists():
            raise FileNotFoundError(f"Required file missing: {metric_path}")

        self.outputs_dir = self.base_dir / self.outputs_dirname / str(iteration)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.developer_log_path = self.outputs_dir / f"developer_{iteration}.txt"
        self._configure_logger()

        self.cpu_core_range = (
            cpu_core_range  # List of CPU cores to use (e.g., [0,1,2,...,41])
        )
        self.gpu_identifier = (
            gpu_identifier  # GPU identifier: MIG UUID or GPU ID (as string)
        )
        self.gpu_isolation_mode = gpu_isolation_mode  # "mig", "multi-gpu", or "none"
        self.conda_env = (
            conda_env  # Conda environment name for isolated package installation
        )

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
        self.blacklisted_ideas: list[str] = []
        self.successful_ideas: list[
            str
        ] = []  # Suggestions that led to successful, non-blacklisted executions
        self.successful_versions: set[int] = (
            set()
        )  # Versions that executed successfully (generated submission)
        self.blacklisted_versions: set[int] = (
            set()
        )  # Versions that were explicitly blacklisted by SOTA
        self.version_scores: dict[int, float] = {}  # Map version number to its score
        self.global_suggestions: list[
            str
        ] = []  # All suggestions with score impact: "suggestion (score improved/worsened/remained by DDD: XXX -> YYY)"
        self.last_suggestion: str | None = None
        self.best_code: str | None = None
        self.best_code_file: str | None = None
        self.best_version: int | None = None
        self.last_successful_version: int | None = (
            None  # For score comparison: most recent version with a score
        )

        self.latest_submission_path: Path | None = None

        self.later_recommendations: dict | None = later_recommendations
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

        self.model_name: str | None = model_name
        self.model_recommendations: str | None = model_recommendations

        assert self.model_name is not None

        self.logger.info(
            "Initialized DeveloperAgent for slug=%s iteration=%s",
            self.slug,
            self.iteration,
        )
        self.logger.debug("Outputs directory resolved to: %s", self.outputs_dir)
        if self.conda_env:
            self.logger.info("Conda environment assigned: %s", self.conda_env)

    def _configure_logger(self) -> None:
        # Create a unique logger for this instance to avoid cross-contamination in parallel execution
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
        if self.cpu_core_range is not None:
            self.logger.info(
                "CPU core range set for parallel execution: %d cores",
                len(self.cpu_core_range),
            )
        if self.gpu_identifier is not None:
            self.logger.info(
                "GPU identifier assigned: %s (mode: %s)",
                self.gpu_identifier,
                self.gpu_isolation_mode,
            )
        return prompt_build_system(
            description=self.description,
            directory_listing=directory_listing,
            model_name=self.model_name,
            slug=self.slug,
            cpu_core_range=self.cpu_core_range,
            gpu_identifier=self.gpu_identifier,
            gpu_isolation_mode=self.gpu_isolation_mode,
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
            model_recommendations=self.model_recommendations,
        )

    def _format_later_recommendations(self) -> str:
        """Format NICE_TO_HAVE recommendations as a string for SOTA search context."""
        if not self.later_recommendations:
            return "No NICE_TO_HAVE recommendations available."

        sections = []

        preprocessing = self.later_recommendations["preprocessing"]
        if preprocessing:
            sections.append("## Preprocessing NICE_TO_HAVE Recommendations")
            for category, content in preprocessing.items():
                later_items = content["NICE_TO_HAVE"]
                if later_items:
                    sections.append(f"\n### {category.replace('_', ' ').title()}")
                    for item in later_items:
                        if item["strategy"]:
                            sections.append(f"- {item['strategy']}")
                            if item["explanation"]:
                                sections.append(f"  {item['explanation']}")

        later_losses = self.later_recommendations["loss_function"]["NICE_TO_HAVE"]
        if later_losses:
            sections.append("\n## Loss Function NICE_TO_HAVE Recommendations")
            for item in later_losses:
                if item["loss_function"]:
                    sections.append(f"- {item['loss_function']}")
                    if item["explanation"]:
                        sections.append(f"  {item['explanation']}")

        later_section = self.later_recommendations["hyperparameters"]["NICE_TO_HAVE"]
        if later_section["hyperparameters"]:
            sections.append("\n## Hyperparameters NICE_TO_HAVE Recommendations")
            for item in later_section["hyperparameters"]:
                if item["hyperparameter"]:
                    sections.append(f"- {item['hyperparameter']}")
                    if item["explanation"]:
                        sections.append(f"  {item['explanation']}")

        if later_section["architectures"]:
            sections.append("\n### Architecture NICE_TO_HAVE Recommendations")
            for item in later_section["architectures"]:
                if item["architecture"]:
                    sections.append(f"- {item['architecture']}")
                    if item["explanation"]:
                        sections.append(f"  {item['explanation']}")

        inf_section = self.later_recommendations["inference_strategies"]["NICE_TO_HAVE"]
        if inf_section["inference_strategies"]:
            sections.append("\n## Inference Strategies NICE_TO_HAVE Recommendations")
            for item in later_section["inference_strategies"]:
                if item["strategy"]:
                    sections.append(f"- {item['strategy']}")
                    if item["explanation"]:
                        sections.append(f"  {item['explanation']}")

        return (
            "\n".join(sections)
            if sections
            else "No NICE_TO_HAVE recommendations available."
        )

    @weave.op()
    def _generate_code(
        self, instructions: str, messages: list[dict[str, str]]
    ) -> tuple[list[dict], dict[str, str], int | None]:
        """Generate code using structured output.

        Returns:
            Tuple of (message_history_entry, code_dict, input_tokens)
            - message_history_entry: List with assistant message containing markdown
            - code_dict: Dict with 'train_py' key
            - input_tokens: Number of input tokens used in this API call (for adaptive trimming)
        """
        self.logger.info(
            "Requesting code generation from model for iteration %s", self.iteration
        )

        schema = CodeGeneration

        result, input_tokens = call_llm(
            model=_DEVELOPER_MODEL,
            system_instruction=instructions,
            messages=messages,
            enable_google_search=True,
            text_format=schema,
            include_usage=True,
        )

        self.logger.info("Model response received for iteration %s", self.iteration)

        code_dict = result.model_dump()
        self.logger.debug("Generated code dict keys: %s", list(code_dict.keys()))

        raw_content = code_dict["train_py"]
        extracted = extract_python_code(raw_content)
        if extracted and extracted != raw_content:
            code_dict["train_py"] = extracted
            self.logger.debug(
                "Extracted code from markdown (%d chars -> %d chars)",
                len(raw_content),
                len(extracted),
            )

        content = f"```python\n{code_dict['train_py']}\n```"

        output = [append_message("assistant", content)]

        if input_tokens:
            self.logger.info("API call used %d input tokens", input_tokens)

        return output, code_dict, input_tokens

    def _postprocess_code(self, code: str) -> tuple[str, int]:
        """Insert resource allocation and BASE_DIR setup at the top of generated code.

        Returns:
            Tuple of (postprocessed_code, num_header_lines_added)
        """
        lines = []
        lines.append("import os")

        if self.cpu_core_range is not None:
            lines.append("import psutil  # For CPU affinity")
            lines.append("")
            lines.append(
                "# CPU affinity (pin to specific cores to prevent resource overlap)"
            )
            lines.append(
                f"psutil.Process(os.getpid()).cpu_affinity({self.cpu_core_range})"
            )

        if self.gpu_identifier is not None:
            lines.append(
                f'os.environ["CUDA_VISIBLE_DEVICES"] = "{self.gpu_identifier}"'
            )
        lines.append(
            f'BASE_DIR = "task/{self.slug}" if not os.getenv(\'KAGGLE_KERNEL_RUN_TYPE\') else "/kaggle/input/{self.slug}"'
        )
        lines.append("")

        header = "\n".join(lines)
        num_header_lines = len(lines)

        return header + "\n" + code, num_header_lines

    def _get_parent_iteration_folder(self) -> Path:
        """Get parent iteration folder for copying shared files.

        For example:
        - If self.iteration is "16_2", returns outputs/16
        - If self.iteration is "16", returns outputs/16

        Returns:
            Path to parent iteration folder (e.g., task/csiro-biomass/outputs/16)
        """
        iteration_str = str(self.iteration)
        base_iteration = iteration_str.split("_")[0]
        return self.base_dir / self.outputs_dirname / base_iteration

    def _write_code(self, code_dict: dict[str, str], version: int) -> Path:
        """Write code to version folder structure.

        Creates folder structure:
        outputs/{iteration}/
        ├── {version}/
        │   ├── train.py (with headers)
        │   ├── train.json (metadata)
        │   ├── cv_splits.json (copied from parent)
        │   └── metric.py (copied from parent)

        Args:
            code_dict: Dict with 'train_py' key
            version: Version number

        Returns:
            Path to the version folder
        """
        version_folder = self.outputs_dir / str(version)
        version_folder.mkdir(parents=True, exist_ok=True)
        self.logger.info("Writing code to version folder: %s", version_folder)

        train_py = code_dict["train_py"]
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

        Launches the script non-blocking, then polls with monitor_logs() every
        _LOG_MONITOR_INTERVAL seconds. The monitor LLM inspects recent output
        and can run bash commands (nvidia-smi, ps, etc.) to check system state.
        If the monitor returns "kill", the process group is terminated immediately.

        Args:
            code_path: Path to the code file to execute (e.g., outputs/16_2/1/train.py)
            version: Version number for logging

        Returns:
            Execution output string
        """
        timeout_seconds = _BASELINE_CODE_TIMEOUT

        job = execute_code(
            str(code_path),
            timeout_seconds=timeout_seconds,
            conda_env=self.conda_env,
        )
        self.logger.info(
            "Launched execution for v%s (pid=%d, timeout=%ds)",
            version,
            job.pid,
            timeout_seconds,
        )

        while not job.done():
            if job.check_timeout():
                self.logger.warning("Hard timeout reached for v%s", version)
                return job.kill("Hard timeout exceeded")

            try:
                verdict = monitor_logs(
                    log_output=job.recent_output(),
                    seconds_since_last_output=job.idle_time(),
                    total_elapsed_seconds=job.elapsed(),
                    pid=job.pid,
                )
                self.logger.info(
                    "Monitor verdict for v%s: %s (%s)",
                    version,
                    verdict.action,
                    verdict.reason,
                )
                if verdict.action == "kill":
                    return job.kill(verdict.reason)
            except Exception:
                self.logger.exception(
                    "Monitor call failed for v%s — continuing execution", version
                )

            time.sleep(_LOG_MONITOR_INTERVAL)

        output = job.result()
        self.logger.info("Execution output captured for version v%s", version)
        self.logger.debug("Execution output: %s", output)

        return output

    def _format_suggestion_entry(
        self, suggestion: str, previous_score: float | None, current_score: float | None
    ) -> str:
        """Format a suggestion entry with score impact information.

        Returns formatted string like:
        - "suggestion (score improved by DDD: XXX -> YYY)"
        - "suggestion (score worsened by DDD: XXX -> YYY)"
        - "suggestion (score remained the same: XXX -> YYY)"
        """
        prev_display = self._format_score_value(previous_score)
        curr_display = self._format_score_value(current_score)

        if prev_display == "N/A" or curr_display == "N/A":
            return f"{suggestion} (score: {prev_display} -> {curr_display})"

        delta = current_score - previous_score
        abs_delta = abs(delta)

        if abs_delta < 1e-9:  # Essentially the same
            impact = "remained the same"
            return f"{suggestion} (score {impact}: {prev_display} -> {curr_display})"
        elif (delta > 0 and not self.is_lower_better) or (
            delta < 0 and self.is_lower_better
        ):
            impact = f"improved by {abs_delta:.6f}"
        else:
            impact = f"worsened by {abs_delta:.6f}"

        return f"{suggestion} (score {impact}: {prev_display} -> {curr_display})"

    def _register_blacklist(self, suggestion: str, reason: str | None = None) -> None:
        if not suggestion:
            return
        entry = suggestion
        if reason:
            entry = f"{suggestion} -- {reason}"

        if entry not in self.blacklisted_ideas:
            self.blacklisted_ideas.append(entry)

    def _register_shared_suggestion(
        self,
        suggestion: str,
        previous_score: float | None,
        current_score: float | None,
        is_blacklisted: bool = False,
    ) -> None:
        """Register suggestion outcome to shared pool with model name and score impact.

        Format: "Model <model_name> tried <suggestion> (score improved/worsened/remained by X: A -> B)"
        """
        if not suggestion:
            return

        prev_display = self._format_score_value(previous_score)
        curr_display = self._format_score_value(current_score)

        model_prefix = f"Model {self.model_name} tried"

        if prev_display == "N/A" or curr_display == "N/A":
            entry = (
                f"{model_prefix} {suggestion} (score: {prev_display} -> {curr_display})"
            )
        else:
            delta = current_score - previous_score
            abs_delta = abs(delta)

            if abs_delta < 1e-9:
                impact = "remained the same"
                entry = f"{model_prefix} {suggestion} (score {impact}: {prev_display} -> {curr_display})"
            elif (delta > 0 and not self.is_lower_better) or (
                delta < 0 and self.is_lower_better
            ):
                impact = f"improved by {abs_delta:.6f}"
                entry = f"{model_prefix} {suggestion} (score {impact}: {prev_display} -> {curr_display})"
            else:
                impact = f"worsened by {abs_delta:.6f}"
                entry = f"{model_prefix} {suggestion} (score {impact}: {prev_display} -> {curr_display})"

        # Add to shared pool (thread-safe)
        with DeveloperAgent._lock:
            if entry not in DeveloperAgent._shared_suggestions:
                DeveloperAgent._shared_suggestions.append(entry)
                status = "blacklisted" if is_blacklisted else "successful"
                self.logger.info("Added to shared suggestions (%s): %s", status, entry)

    def _get_shared_suggestions(self) -> list[str]:
        """Get snapshot of all shared suggestions from all models."""
        with DeveloperAgent._lock:
            return DeveloperAgent._shared_suggestions.copy()

    def _find_most_recent_valid_version(
        self, current_version: int
    ) -> tuple[int, str | None]:
        """Find the most recent version that is valid for rollback (folder-based).

        A version is valid if:
        1. Version folder exists
        2. submission.csv exists in the version folder (successful execution)
        3. Version was NOT blacklisted by SOTA suggestions

        Args:
            current_version: The current version number

        Returns:
            Tuple of (version_number, code_content_markdown)
            - code_content_markdown contains train.py (headers stripped)
            Returns (1, None) if no valid version found
        """
        for v in range(current_version - 1, 0, -1):
            version_folder = self.outputs_dir / str(v)
            if not version_folder.exists():
                continue

            submission_path = version_folder / "submission.csv"
            if not submission_path.exists():
                self.logger.debug(
                    f"v{v} skipped: no submission.csv in {version_folder}"
                )
                continue

            is_blacklisted = v in self.blacklisted_versions
            if is_blacklisted:
                self.logger.debug(f"v{v} skipped: blacklisted")
                continue

            self.logger.info(f"Found most recent valid version for rollback: v{v}")

            try:
                train_path = version_folder / "train.py"
                if train_path.exists():
                    train_py = strip_header_from_code(train_path)
                else:
                    train_py = "# train.py not found"

                code_content = f"train.py:\n```python\n{train_py}\n```"

                return v, code_content
            except Exception as e:
                self.logger.error(f"Failed to read code from v{v}: {e}")
                continue

        # No valid version found
        self.logger.warning(
            "No valid rollback version found; falling back to v1 (initial version)"
        )
        return 1, None

    def _extract_final_summary(self, red_flags_response: str) -> str:
        """Extract just the Final Summary section from red flags response.

        Args:
            red_flags_response: Full markdown response from search_red_flags()

        Returns:
            The Final Summary text, or full response if section not found
        """
        match = re.search(r"### Final Summary\s*\n(.+)", red_flags_response, re.DOTALL)

        if match:
            summary = match.group(1).strip()
            self.logger.info("Extracted Final Summary (%d chars)", len(summary))
            return summary
        else:
            # Fallback: return entire response if section not found
            self.logger.warning(
                "Could not extract Final Summary section, using full response"
            )
            return red_flags_response

    def _build_extended_data_listing(self, version: int) -> str:
        """Build directory listing for SOTA suggestions including base dir and current models.

        Includes:
        - All files and train/test folders from task/<slug>/
        - external_data_* folders from outputs/<iteration>/
        - models_{version}/ folder from outputs/<iteration>/

        Args:
            version: Current version number

        Returns:
            Directory listing string
        """
        lines = []

        lines.append("=== Base Directory (task data) ===")
        base_listing = self._build_base_dir_listing()
        lines.append(base_listing)

        if (
            self.external_data_listing
            and self.external_data_listing != "No external data directories found."
        ):
            lines.append("\n=== External Data ===")
            lines.append(self.external_data_listing)

        models_dir = self.outputs_dir / f"models_{version}"
        if models_dir.exists() and models_dir.is_dir():
            lines.append(f"\n=== Models (version {version}) ===")
            lines.append(f"models_{version}/")
            dir_listing = _build_directory_listing(str(models_dir))
            lines.append(dir_listing)

        return "\n".join(lines)

    def _build_base_dir_listing(self) -> str:
        """Build directory listing for base task directory (train/, test/, root files).

        Returns:
            Directory listing showing train/, test/ folders and root-level files
        """
        lines = []
        base_dir = Path(self.base_dir)

        root_files = sorted([f.name for f in base_dir.iterdir() if f.is_file()])

        lines.append("./")
        for file_name in root_files:
            lines.append(f"    {file_name}")

        train_dir = base_dir / "train"
        if train_dir.exists() and train_dir.is_dir():
            lines.append("    train/")
            train_listing = _build_directory_listing(str(train_dir))
            for line in train_listing.split("\n"):
                if line.strip():
                    lines.append(f"    {line}")

        test_dir = base_dir / "test"
        if test_dir.exists() and test_dir.is_dir():
            lines.append("    test/")
            test_listing = _build_directory_listing(str(test_dir))
            for line in test_listing.split("\n"):
                if line.strip():
                    lines.append(f"    {line}")

        return "\n".join(lines)

    def _call_sota_suggestions(self, attempt_number: int = 1, **kwargs):
        """
        Call SOTA suggestions tool with appropriate parameters.

        Args:
            attempt_number: Which attempt this is (1, 2, 3, ...) for interleaving strategy
            **kwargs: Arguments to pass to search_sota_suggestions

        Returns:
            Parsed SOTAResponse object or None if parsing fails
        """
        return search_sota_suggestions(attempt_number=attempt_number, **kwargs)

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

            if _USE_VALIDATION_SCORE:
                self.logger.info(
                    "Using validation score from train_stats.json (use_validation_score=True)"
                )
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
            else:
                grade_feedback = ""
                try:
                    info, grade_feedback, returncode, stderr = run_grade(
                        str(submission_path), self.slug
                    )
                    self.logger.info("Grade feedback: %s", grade_feedback)
                    if returncode != 0:
                        self.logger.warning(
                            "Grading command returned non-zero exit (%s). stderr=\n%s",
                            returncode,
                            stderr,
                        )
                    else:
                        self.logger.info(
                            "Grading command completed successfully for version %s",
                            version,
                        )
                        run_score = info.get("score")
                        self._log_attempt_score(attempt, run_score)
                        self.logger.info("Your result on the test set is %s", run_score)

                        if run_score is not None:
                            self.version_scores[version] = run_score
                            self.last_successful_version = version
                except Exception as exc:
                    grade_feedback = f"Failed to run grading tool: {exc}"
                    self.logger.exception(
                        "Grading command failed for version %s", version
                    )

                code_context += (
                    f"<leaderboard_score>\n{run_score}\n</leaderboard_score>\n"
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

    def _gather_sota_feedback(
        self, code_context: str, version: int, attempt_number: int = 1
    ):
        """Gather SOTA feedback through red flags analysis and SOTA suggestions.

        Args:
            code_context: Code with execution logs and analysis
            version: Current version number to include models_{version}/ in directory listing
            attempt_number: Which attempt this is (1, 2, 3, ...) for interleaving strategy

        Returns:
            Parsed SOTAResponse object or None if gathering/parsing failed
        """
        try:
            later_context = self._format_later_recommendations()

            extended_listing = self._build_extended_data_listing(version)

            # STAGE 1: Identify red flags via direct analysis
            self.logger.info("Stage 1: Identifying red flags via direct analysis...")

            version_folder = self.outputs_dir / str(version)
            training_images = []
            for img_name in ["loss_curve.png", "metric_curve.png"]:
                img_path = version_folder / img_name
                if img_path.exists():
                    training_images.append(img_path)
                else:
                    self.logger.debug(f"Training image not found: {img_path}")

            train_stats_path = version_folder / "train_stats.json"
            train_stats = None
            if train_stats_path.exists():
                try:
                    with open(train_stats_path, "r") as f:
                        train_stats = json.load(f)
                    self.logger.info(
                        "Loaded train_stats.json with %d keys", len(train_stats)
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to read train_stats.json: {e}")

            red_flags_response = search_red_flags(
                description=self.description,
                context=code_context,
                images=training_images if training_images else None,
                train_stats=train_stats,
            )
            self.logger.info(
                "Red flags response length: %d chars", len(red_flags_response)
            )

            final_summary = self._extract_final_summary(red_flags_response)

            # STAGE 2: Generate SOTA suggestions based on red flags
            self.logger.info(
                "Stage 2: Generating SOTA suggestions based on red flags..."
            )

            shared_suggestions = self._get_shared_suggestions()

            self.logger.info(
                "Using %d shared suggestions from all models (including this one)",
                len(shared_suggestions),
            )

            sota_response = self._call_sota_suggestions(
                description=self.description,
                context=code_context,
                red_flags=final_summary,
                executed_suggestion=self.last_suggestion,
                failed_ideas=self.blacklisted_ideas,  # Keep instance-level for context
                shared_suggestions=shared_suggestions,  # New: shared across all models
                later_recommendations=later_context,
                external_data_listing=extended_listing,
                plan_content=self.plan_content,
                attempt_number=attempt_number,
                slug=self.slug,
                data_path=str(self.base_dir),
                cpu_core_range=self.cpu_core_range,
                gpu_identifier=self.gpu_identifier,
                file_suffix=str(
                    self.iteration
                ),  # Pass iteration as file suffix to prevent race conditions
                version=version,
                images=training_images if training_images else None,
                train_stats=train_stats,
                hitl_sota=_HITL_SOTA,
            )

            # HITL: Show suggestion and let user accept or override
            if _HITL_SOTA and sota_response:
                print(f"\n{'=' * 60}")
                print(f"[HITL] Model: {self.model_name} | Version: {version}")
                print(f"[HITL] Suggestion: {sota_response.suggestion}")
                print(f"[HITL] Reason: {sota_response.suggestion_reason}")
                print(
                    f"[HITL] Blacklist previous: {sota_response.blacklist} ({sota_response.blacklist_reason})"
                )
                print(f"{'=' * 60}")
                user_input = input(
                    "[HITL] Press Enter to accept, or type replacement: "
                ).strip()
                if user_input:
                    user_code = input(
                        "[HITL] Enter code snippet (or press Enter to skip): "
                    ).strip()
                    sota_response = SOTAResponse(
                        blacklist=sota_response.blacklist,
                        blacklist_reason=sota_response.blacklist_reason,
                        suggestion=user_input,
                        suggestion_reason="Human override",
                        suggestion_code=user_code,
                    )
                    self.logger.info(
                        "HITL: User overrode suggestion with: %s", user_input
                    )
                else:
                    self.logger.info("HITL: User accepted automated suggestion")

            return sota_response
        except Exception:
            self.logger.exception("Failed to fetch red flags or SOTA suggestions")
            return None

    def _get_checkpoint_db(self):
        """Open (or return cached) checkpoint database connection."""
        if self._checkpoint_conn is None:
            self._checkpoint_conn = _create_checkpoint_db()
        return self._checkpoint_conn

    def _save_checkpoint(
        self, version, input_list, last_input_tokens, sota_suggestions_call_id
    ):
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
            "blacklisted_versions": list(self.blacklisted_versions),
            "blacklisted_ideas": self.blacklisted_ideas,
            "successful_ideas": self.successful_ideas,
            "global_suggestions": self.global_suggestions,
            "input_list": input_list,
            "last_input_tokens": last_input_tokens,
            "last_suggestion": self.last_suggestion,
            "sota_suggestions_call_id": sota_suggestions_call_id,
        }
        conn = self._get_checkpoint_db()
        save_checkpoint(
            conn, self.slug, str(self.iteration), self.model_name, version, state
        )
        self.logger.info("Checkpoint saved for v%s", version)

    def _load_latest_checkpoint(self):
        """Load the most recent checkpoint and restore instance state.

        Returns dict with run() locals (input_list, last_input_tokens,
        sota_suggestions_call_id, version) or None if no checkpoint exists.
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
        self.blacklisted_versions = set(row["blacklisted_versions"])
        self.blacklisted_ideas = row["blacklisted_ideas"]
        self.successful_ideas = row["successful_ideas"]
        self.global_suggestions = row["global_suggestions"]
        self.last_suggestion = row["last_suggestion"]

        # Reconstruct derived state
        self.previous_runs = [(None, None)] * len(self.version_scores)
        if self.version_scores:
            self.last_successful_version = max(self.version_scores.keys())
        if self.best_version and self.best_code_file:
            train_path = self.outputs_dir / self.best_code_file
            if train_path.exists():
                self.best_code = strip_header_from_code(train_path)

        # Re-populate class-level shared suggestions from this model's history
        with DeveloperAgent._lock:
            for entry in self.global_suggestions:
                if entry not in DeveloperAgent._shared_suggestions:
                    DeveloperAgent._shared_suggestions.append(entry)

        return {
            "input_list": row["input_list"],
            "last_input_tokens": row["last_input_tokens"],
            "sota_suggestions_call_id": row["sota_suggestions_call_id"],
            "version": row["version"],
        }

    def _run_feedback_and_build_next_instruction(
        self,
        version,
        code_context,
        run_score,
        previous_successful_version,
        base_score,
        submission_exists,
        output,
        input_list,
        sota_suggestions_call_id,
    ):
        """Run step 6: gather feedback and append next instruction to input_list.

        Returns (updated_sota_suggestions_call_id, should_break).
        """
        if submission_exists:
            sota_suggestions_call_id += 1
            sota_response = self._gather_sota_feedback(
                code_context, version=version, attempt_number=sota_suggestions_call_id
            )

            if sota_response:
                suggestion_text = sota_response.suggestion.strip()
                blacklist_flag = bool(sota_response.blacklist)
                blacklist_reason = sota_response.blacklist_reason.strip()
                suggestion_code = sota_response.suggestion_code.strip()
            else:
                suggestion_text = ""
                blacklist_flag = False
                blacklist_reason = ""
                suggestion_code = ""

            self.logger.info(
                "SOTA suggestion: %s (blacklist=%s, code_len=%d)",
                suggestion_text,
                blacklist_flag,
                len(suggestion_code),
            )

            if previous_successful_version is None:
                initial_entry = f"Initial implementation (score: {self._format_score_value(run_score)})"
                self.global_suggestions.append(initial_entry)
                self.logger.info("Recorded initial implementation: %s", initial_entry)
            elif self.last_suggestion:
                current_score = run_score
                suggestion_entry = self._format_suggestion_entry(
                    self.last_suggestion, base_score, current_score
                )
                self.global_suggestions.append(suggestion_entry)
                self.logger.info(
                    "Recorded suggestion with impact: %s", suggestion_entry
                )
                self._register_shared_suggestion(
                    self.last_suggestion,
                    base_score,
                    current_score,
                    is_blacklisted=blacklist_flag,
                )

            if blacklist_flag and self.last_suggestion:
                self._register_blacklist(self.last_suggestion, blacklist_reason)
                self.blacklisted_versions.add(version)
                self.logger.info(
                    "Previous suggestion marked as blacklisted: %s (reason: %s) - marking v%s as blacklisted",
                    self.last_suggestion,
                    blacklist_reason or "N/A",
                    version,
                )
            elif (
                not blacklist_flag
                and self.last_suggestion
                and version in self.successful_versions
            ):
                if self.last_suggestion not in self.successful_ideas:
                    self.successful_ideas.append(self.last_suggestion)
                    self.logger.info(
                        "Previous suggestion marked as successful: %s (v%s executed successfully and not blacklisted)",
                        self.last_suggestion,
                        version,
                    )

            if suggestion_text:
                self.logger.info("Summary of SOTA suggestion: %s", suggestion_text)
            else:
                self.logger.info(
                    "SOTA response did not include a new suggestion summary."
                )

            if suggestion_text and suggestion_text.strip() == "No suggestions.":
                self.logger.info(
                    "Model indicated 'No suggestions.' - breaking out of loop and returning best result"
                )
                self.logger.info(
                    "Final best score: %s (version %s)",
                    self.best_score,
                    self.best_version,
                )
                return sota_suggestions_call_id, True

            if suggestion_text:
                self.last_suggestion = suggestion_text
            elif blacklist_flag:
                self.last_suggestion = None

            next_log_path = self.outputs_dir / f"{version + 1}/train.txt"
            next_submission_path = self.outputs_dir / f"{version + 1}/submission.csv"
            next_version_folder = self.outputs_dir / str(version + 1)

            suggestion_block = ""
            if suggestion_text:
                suggestion_block += f"<suggestion>\n{suggestion_text}\n</suggestion>\n"
            else:
                suggestion_block += (
                    "<suggestion>\nNo suggestion provided.\n</suggestion>\n"
                )

            if suggestion_code:
                suggestion_block += (
                    "Suggested code snippet:\n```python\n" + suggestion_code + "\n```\n"
                )
            else:
                suggestion_block += "Suggested code snippet: No code provided.\n"

            rollback_code = None
            if blacklist_flag:
                _, rollback_code = self._find_most_recent_valid_version(version)

            next_instr = (
                f"{suggestion_block}\n"
                f"Remember:\n"
                f"- write logs to {next_log_path}\n"
                f"- save artifacts to {next_version_folder}/: submission.csv, valid_preds.csv, train_stats.json, models, loss_curve.png, metric_curve.png"
            )

            if blacklist_flag and rollback_code:
                input_list.append(
                    append_message(
                        "user",
                        "The previous code has been blacklisted. Here is the most recent valid (successful and non-blacklisted) version for your reference (please start work from this version): \n"
                        + rollback_code,
                    )
                )
            elif blacklist_flag:
                self.logger.warning(
                    "Blacklist triggered but no valid rollback version available"
                )

            input_list.append(append_message("user", next_instr))

        else:
            next_log_path = self.outputs_dir / f"{version + 1}/train.txt"
            next_submission_path = self.outputs_dir / f"{version + 1}/submission.csv"
            next_version_folder = self.outputs_dir / str(version + 1)
            version_folder = self.outputs_dir / str(version)

            is_timeout = "Code execution timed out after" in output
            is_oom = "CUDA out of memory" in output or "OutOfMemoryError" in output

            if is_timeout or is_oom:
                error_type = "Timeout" if is_timeout else "OOM"
                self.logger.info(
                    f"{error_type} detected - running red flags analysis on logs and code"
                )

                if is_timeout:
                    error_context = "\n<timeout_error>\nThe script was not able to execute within 1 hour. Please investigate.\n</timeout_error>\n"
                else:
                    error_context = "\n<oom_error>\nCUDA out of memory error detected. The model or batch size is too large for available GPU memory. Please investigate.\n</oom_error>\n"

                code_context_error = code_context + error_context

                train_stats_path = version_folder / "train_stats.json"
                train_stats = None
                if train_stats_path.exists():
                    try:
                        with open(train_stats_path, "r") as f:
                            train_stats = json.load(f)
                        self.logger.info(
                            "Loaded train_stats.json for error analysis (%d keys)",
                            len(train_stats),
                        )
                    except Exception as e:
                        self.logger.warning(f"Failed to read train_stats.json: {e}")

                try:
                    red_flags_response = search_red_flags(
                        description=self.description,
                        context=code_context_error,
                        train_stats=train_stats,
                    )
                    self.logger.info(
                        f"Red flags analysis complete for {error_type} (length: %d chars)",
                        len(red_flags_response),
                    )

                    final_summary = self._extract_final_summary(red_flags_response)

                    error_message = "TIMEOUT" if is_timeout else "OUT OF MEMORY"
                    next_instr = f"""
Your code FAILED during execution due to {error_message}!
{output}

Performance analysis:
{final_summary}

{prompt_execution_failure_suffix(next_log_path, next_submission_path, next_version_folder)}
"""
                except Exception:
                    self.logger.exception(
                        f"Failed to run red flags analysis for {error_type}"
                    )
                    error_message = "TIMEOUT" if is_timeout else "OUT OF MEMORY"
                    next_instr = f"""
Your code FAILED during execution due to {error_message}!
This is the stack trace and advice on how to fix the error:
{output}

{prompt_execution_failure_suffix(next_log_path, next_submission_path, next_version_folder)}
"""
            else:
                next_instr = f"""
Your code FAILED during execution!
This is the stack trace and advice on how to fix the error:
{output}

{prompt_execution_failure_suffix(next_log_path, next_submission_path, next_version_folder)}
"""

            input_list.append(append_message("user", next_instr))

        return sota_suggestions_call_id, False

    @weave.op()
    def run(
        self, max_time_seconds: int = 6 * 3600
    ) -> tuple[float, str | None, list[str], list[str]]:
        self.logger.info(
            "Starting developer run for slug=%s iteration=%s",
            self.slug,
            self.iteration,
        )
        self.logger.info("cpu core range: %s", self.cpu_core_range)
        self.logger.info(
            "gpu identifier: %s (mode: %s)",
            self.gpu_identifier,
            self.gpu_isolation_mode,
        )

        start_time = time.time()
        deadline = start_time + max_time_seconds

        run_score = 0
        last_input_tokens = None

        system_prompt = self._compose_system()

        self.logger.info(
            "External data listing (%d chars): %s",
            len(self.external_data_listing),
            self.external_data_listing[:200] + "..."
            if len(self.external_data_listing) > 200
            else self.external_data_listing,
        )
        self.logger.info(
            "Plan content (%d chars): %s",
            len(self.plan_content),
            self.plan_content[:200] + "..."
            if len(self.plan_content) > 200
            else self.plan_content,
        )

        checkpoint = self._load_latest_checkpoint()
        if checkpoint:
            input_list = checkpoint["input_list"]
            last_input_tokens = checkpoint["last_input_tokens"]
            sota_suggestions_call_id = checkpoint["sota_suggestions_call_id"]
            attempt = checkpoint["version"]
            self.logger.info(
                "Resumed from checkpoint v%s (best_score=%s)", attempt, self.best_score
            )

            # Replay step 6 for the checkpointed version (checkpoint is saved after step 5)
            resumed_version = attempt
            version_folder = self.outputs_dir / str(resumed_version)
            train_py_path = version_folder / "train.py"
            if train_py_path.exists():
                code_clean = strip_header_from_code(train_py_path)
                code_context, run_score, prev_ver, base_score, sub_exists = (
                    self._evaluate_submission(
                        code_clean, resumed_version, resumed_version
                    )
                )
                sota_suggestions_call_id, _ = (
                    self._run_feedback_and_build_next_instruction(
                        resumed_version,
                        code_context,
                        run_score,
                        prev_ver,
                        base_score,
                        sub_exists,
                        "",
                        input_list,
                        sota_suggestions_call_id,
                    )
                )
            self.logger.info(
                "Step 6 replayed for v%s, entering main loop", resumed_version
            )
        else:
            user_prompt = self._build_user_prompt(version=1)
            input_list = [append_message("user", user_prompt)]
            attempt = 0
            sota_suggestions_call_id = 0
        while True:
            now = time.time()
            if max_time_seconds is not None and now >= deadline:
                self.logger.info(
                    "Time budget exhausted (%.2f minutes)",
                    (deadline - start_time) / 60.0,
                )
                break

            # Adaptive trimming: if previous call used >80% of token limit, remove 4 messages
            max_input_tokens = get_config_value(
                "runtime", "max_developer_input_tokens", default=250000
            )
            threshold = max_input_tokens * 0.8

            if last_input_tokens and last_input_tokens > threshold:
                self.logger.info(
                    "Token threshold exceeded: %d > %d (80%% of %d), removing messages",
                    last_input_tokens,
                    int(threshold),
                    max_input_tokens,
                )
                for _ in range(min(4, len(input_list) - 1)):
                    input_list.pop(0)

                # Ensure first message is from user (required by API)
                while input_list and input_list[0].get("role") != "user":
                    input_list.pop(0)

                self.logger.info(
                    "Trimmed to %d messages for attempt %s",
                    len(input_list),
                    attempt + 1,
                )

            attempt += 1

            artifact = wandb.Artifact(f"{self.iteration}-{self.slug}", type="files")

            minutes_left = (
                ((deadline - now) / 60.0)
                if max_time_seconds is not None
                else float("inf")
            )
            self.logger.info("Attempt %s (time left ~%.1f min)", attempt, minutes_left)
            version = attempt

            response_output, code_dict, last_input_tokens = self._generate_code(
                instructions=system_prompt, messages=input_list
            )
            input_list += response_output

            version_folder = self._write_code(code_dict, version)

            # ---------------------------
            # Pre-exec guardrail checks
            # ---------------------------
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
                summary_text = build_block_summary(guard_report)
                next_log_path = self.outputs_dir / f"{version + 1}/train.txt"
                next_submission_path = (
                    self.outputs_dir / f"{version + 1}/submission.csv"
                )
                next_version_folder = self.outputs_dir / str(version + 1)
                fix_instr = prompt_guardrail_fix_suffix(
                    next_log_path, next_submission_path, next_version_folder
                )
                guardrail_prompt = summary_text + fix_instr
                input_list.append(append_message("user", guardrail_prompt))
                self.logger.info(
                    "User prompt with guardrail feedback: %s", guardrail_prompt
                )
                continue

            # Execute the code
            output = self._execute_code(train_py_path, version)

            # Evaluate submission and build enriched context
            (
                code_context,
                run_score,
                previous_successful_version,
                base_score,
                submission_exists,
            ) = self._evaluate_submission(code_clean, version, attempt)

            # Checkpoint after step 5 (before feedback)
            try:
                self._save_checkpoint(
                    version, input_list, last_input_tokens, sota_suggestions_call_id
                )
            except Exception as exc:
                self.logger.exception(
                    "Failed to save checkpoint for v%s: %s", version, exc
                )

            # Step 6: Gather feedback and build next instruction
            sota_suggestions_call_id, should_break = (
                self._run_feedback_and_build_next_instruction(
                    version,
                    code_context,
                    run_score,
                    previous_successful_version,
                    base_score,
                    submission_exists,
                    output,
                    input_list,
                    sota_suggestions_call_id,
                )
            )
            if should_break:
                break

            self.logger.info("previous runs count: %s", len(self.previous_runs))

            allowed_extensions = {".py", ".txt", ".csv"}
            for path in self.outputs_dir.iterdir():
                if not path.is_file():
                    self.logger.debug(
                        "Skipping non-file path when logging artifact: %s", path
                    )
                    continue

                if path.suffix.lower() in allowed_extensions:
                    artifact.add_file(str(path), overwrite=True)
                else:
                    self.logger.debug(
                        "Skipping file due to extension filtering: %s (extension: %s)",
                        path.name,
                        path.suffix,
                    )

            try:
                artifact.save()
            except Exception as e:
                self.logger.exception("Failed to save wandb artifact: %s", e)

        return (
            self.best_score,
            self.best_code_file,
            self.blacklisted_ideas,
            self.successful_ideas,
        )
