import logging
import math
import os
import re
import shutil
import time
import threading
from pathlib import Path
from typing import Optional
import json

from dotenv import load_dotenv
from project_config import get_config
import weave
import wandb

from tools.developer import (
    execute_code,
    search_red_flags,
    search_sota_suggestions,
)
from utils.guardrails import evaluate_guardrails, build_block_summary
from tools.helpers import call_llm_with_retry, call_llm_with_retry_anthropic, call_llm_with_retry_google, _build_directory_listing
from utils.llm_utils import detect_provider, extract_text_from_response, append_message
from utils.diffs import (
    extract_diff_block,
    normalize_diff_payload,
    apply_patch as util_apply_patch,
)
from utils.grade import run_grade
from utils.code_utils import strip_header_from_code, extract_structured_code
from prompts.developer_agent import (
    build_system as prompt_build_system,
    build_user as prompt_build_user,
    patch_mode_directive as prompt_patch_mode_directive,
    guardrail_fix_suffix as prompt_guardrail_fix_suffix,
    execution_failure_suffix as prompt_execution_failure_suffix,
)


# Module-level logger (not used by instances to avoid cross-contamination in parallel execution)
_module_logger = logging.getLogger(__name__)


_CONFIG = get_config()
_LLM_CFG = _CONFIG.get("llm")
_PATH_CFG = _CONFIG.get("paths")
_RUNTIME_CFG = _CONFIG.get("runtime")
_GUARDRAIL_CFG = _CONFIG.get("guardrails")
_DEVELOPER_CFG = _CONFIG.get("developer")

_ENABLE_LOGGING_GUARD = bool(_GUARDRAIL_CFG.get("logging_basicconfig_order"))

_PATCH_MODE_ENABLED = bool(_RUNTIME_CFG.get("patch_mode_enabled"))
_USE_VALIDATION_SCORE = bool(_RUNTIME_CFG.get("use_validation_score"))

_DEVELOPER_MODEL = _LLM_CFG.get("developer_model")
_HITL_INSTRUCTIONS = _DEVELOPER_CFG.get("hitl_instructions", [])

_TASK_ROOT = Path(_PATH_CFG.get("task_root"))
_OUTPUTS_DIRNAME = _PATH_CFG.get("outputs_dirname")

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

    def __init__(self, slug: str, iteration: int | str, model_name: Optional[str] = None, model_recommendations: Optional[str] = None, later_recommendations: Optional[dict] = None, external_data_listing: Optional[str] = None, plan_content: Optional[str] = None, cpu_core_range: Optional[list[int]] = None, gpu_identifier: Optional[str] = None, gpu_isolation_mode: str = "none", conda_env: Optional[str] = None):
        load_dotenv()
        self.slug = slug
        self.iteration = iteration  # Can be int (legacy) or str like "1_1" (for parallel baselines)

        self.task_root = _TASK_ROOT
        self.outputs_dirname = _OUTPUTS_DIRNAME
        self.base_dir = self.task_root / slug
        self.outputs_dir = self.base_dir / self.outputs_dirname / str(iteration)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.developer_log_path = self.outputs_dir / f"developer_{iteration}.txt"
        self._configure_logger()

        # Resource allocation for parallel execution
        self.cpu_core_range = cpu_core_range  # List of CPU cores to use (e.g., [0,1,2,...,41])
        self.gpu_identifier = gpu_identifier  # GPU identifier: MIG UUID or GPU ID (as string)
        self.gpu_isolation_mode = gpu_isolation_mode  # "mig", "multi-gpu", or "none"
        self.conda_env = conda_env  # Conda environment name for isolated package installation

        # Metric-related defaults; overwritten once benchmark info is available
        self.gold_threshold: Optional[float] = None
        self.is_lower_better: bool = False
        self.best_score: float = float("-inf")

        # Iteration state
        self.previous_runs: list[tuple[str, float]] = []
        self.blacklisted_ideas: list[str] = []
        self.successful_ideas: list[str] = []  # Suggestions that led to successful, non-blacklisted executions
        self.successful_versions: set[int] = set()  # Versions that executed successfully (generated submission)
        self.blacklisted_versions: set[int] = set()  # Versions that were explicitly blacklisted by SOTA
        self.version_scores: dict[int, float] = {}  # Map version number to its score
        self.global_suggestions: list[str] = []  # All suggestions with score impact: "suggestion (score improved/worsened/remained by DDD: XXX -> YYY)"
        self.last_suggestion: Optional[str] = None
        self.best_code: Optional[str] = None
        self.best_code_file: Optional[str] = None
        self.best_version: Optional[int] = None
        self.next_patch_base_version: Optional[int] = None  # For patch mode: which file to diff against
        self.last_successful_version: Optional[int] = None  # For score comparison: most recent version with a score

        self._load_benchmark_info()
        self.best_score = float("inf") if self.is_lower_better else float("-inf")

        # File targets
        self.latest_submission_path: Optional[Path] = None
        self.benchmark_info: Optional[dict] = None

        # LATER recommendations for progressive enhancement
        self.later_recommendations: Optional[dict] = later_recommendations
        self.threshold_directive: str = ""

        # External data and plan content (passed from orchestrator)
        self.external_data_listing: str = external_data_listing or "No external data directories found."
        self.plan_content: str = plan_content or "No plan.md found."

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        # Patch mode configuration
        self.patch_mode_enabled = _PATCH_MODE_ENABLED

        # Model-specific strategy recommendations for the developer system prompt
        self.model_name: Optional[str] = model_name
        self.model_recommendations: Optional[str] = model_recommendations

        assert self.model_name is not None

        self.logger.info(
            "Initialized DeveloperAgent for slug=%s iteration=%s", self.slug, self.iteration
        )
        self.logger.debug("Outputs directory resolved to: %s", self.outputs_dir)
        if self.conda_env:
            self.logger.info("Conda environment assigned: %s", self.conda_env)

    def _configure_logger(self) -> None:
        # Create a unique logger for this instance to avoid cross-contamination in parallel execution
        self.logger = logging.getLogger(f"{__name__}.{self.slug}.{self.iteration}")

        # Clear any existing handlers to ensure clean slate
        self.logger.handlers = []

        # Add file handler for this specific instance
        file_handler = logging.FileHandler(self.developer_log_path)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        )
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.DEBUG)

        # Prevent propagation to parent loggers to avoid duplicate logs
        self.logger.propagate = False

    def _code_filename(self, version: int) -> str:
        """Return code path relative to outputs_dir (folder-based).

        For folder-based structure: {version}/train.py
        e.g., "1/train.py", "2/train.py"
        """
        return f"{version}/train.py"

    def _log_filename(self, version: int) -> str:
        """Return log path relative to outputs_dir (folder-based).

        For folder-based structure: {version}/train.txt
        e.g., "1/train.txt", "2/train.txt"

        This captures stdout/stderr from train.py execution.
        Structured training metadata is written by train.py to train_stats.json.
        """
        return f"{version}/train.txt"

    def _submission_filename(self, version: int) -> str:
        """Return submission path relative to outputs_dir (folder-based).

        For folder-based structure: {version}/submission.csv
        e.g., "1/submission.csv", "2/submission.csv"
        """
        return f"{version}/submission.csv"

    def _get_code_timeout(self) -> int:
        """
        Get the timeout for code execution in seconds.
        Can be overridden by subclasses (e.g., EnsemblerAgent).

        Returns:
            Timeout in seconds for code execution (5400 = 1.5 hours for baseline)
        """
        from tools.developer import _BASELINE_CODE_TIMEOUT
        return _BASELINE_CODE_TIMEOUT



    def _load_benchmark_info(self) -> None:
        self.benchmark_info = None
        sample_submission = self.base_dir / "sample_submission.csv"
        if not sample_submission.exists():
            self.logger.warning(
                "Sample submission not found at %s; skipping baseline grading",
                sample_submission,
            )
            return

        info, stdout, returncode, stderr = run_grade(str(sample_submission), self.slug)
        try:
            self.logger.info("Baseline grade feedback: %s", stdout)
        except Exception:
            self.logger.debug("Failed to log baseline grade feedback")
        if returncode != 0:
            self.logger.warning(
                "Baseline grading command returned non-zero exit (%s). stderr=\n%s",
                returncode,
                stderr,
            )
            return
        self.benchmark_info = info
        self.gold_threshold = info.get("gold_threshold")
        self.is_lower_better = info.get("is_lower_better")
        self.logger.info("is_lower_better=%s", self.is_lower_better)

    def _compose_system(self, allow_multi_fold: bool = False) -> str:
        self.logger.debug("Composing system prompt for slug=%s (allow_multi_fold=%s)", self.slug, allow_multi_fold)
        with open(self.base_dir / "description.md", "r") as f:
            self.description = f.read()
        self.logger.debug("Description length: %s characters", len(self.description))
        directory_listing = _build_directory_listing(self.base_dir)
        self.logger.debug(
            "Directory listing prepared for %s (length=%s)", self.base_dir, len(directory_listing)
        )
        if self.cpu_core_range is not None:
            self.logger.info("CPU core range set for parallel execution: %d cores", len(self.cpu_core_range))
        if self.gpu_identifier is not None:
            self.logger.info("GPU identifier assigned: %s (mode: %s)", self.gpu_identifier, self.gpu_isolation_mode)
        return prompt_build_system(
            description=self.description,
            directory_listing=directory_listing,
            model_name=self.model_name,
            slug=self.slug,
            cpu_core_range=self.cpu_core_range,
            gpu_identifier=self.gpu_identifier,
            gpu_isolation_mode=self.gpu_isolation_mode,
            allow_multi_fold=allow_multi_fold,
            hitl_instructions=_HITL_INSTRUCTIONS,
        )

    def _build_user_prompt(self, version: int) -> str:
        self.logger.debug("Building user prompt")
        base_dir_display = self.base_dir
        outputs_dir_display = self.outputs_dir
        log_path_display = self.outputs_dir / self._log_filename(version)
        submission_path_display = self.outputs_dir / self._submission_filename(version)
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

        # Preprocessing
        preprocessing = self.later_recommendations.get("preprocessing", {})
        if preprocessing:
            sections.append("## Preprocessing NICE_TO_HAVE Recommendations")
            for category, content in preprocessing.items():
                if isinstance(content, dict) and "NICE_TO_HAVE" in content:
                    later_items = content["NICE_TO_HAVE"]
                    if later_items:
                        sections.append(f"\n### {category.replace('_', ' ').title()}")
                        for item in later_items:
                            if isinstance(item, dict):
                                strategy = item.get("strategy", "")
                                explanation = item.get("explanation", "")
                                if strategy:
                                    sections.append(f"- {strategy}")
                                    if explanation:
                                        sections.append(f"  {explanation}")

        # Loss function
        loss_fn = self.later_recommendations.get("loss_function", {})
        if loss_fn and "NICE_TO_HAVE" in loss_fn:
            sections.append("\n## Loss Function NICE_TO_HAVE Recommendations")
            later_losses = loss_fn["NICE_TO_HAVE"]
            if isinstance(later_losses, list):
                for item in later_losses:
                    if isinstance(item, dict):
                        loss_name = item.get("loss_function", "")
                        explanation = item.get("explanation", "")
                        if loss_name:
                            sections.append(f"- {loss_name}")
                            if explanation:
                                sections.append(f"  {explanation}")

        # Hyperparameters
        hyperparams = self.later_recommendations.get("hyperparameters", {})
        if hyperparams and "NICE_TO_HAVE" in hyperparams:
            later_section = hyperparams["NICE_TO_HAVE"]

            hp_list = later_section.get("hyperparameters", [])
            if hp_list:
                sections.append("\n## Hyperparameters NICE_TO_HAVE Recommendations")
                for item in hp_list:
                    if isinstance(item, dict):
                        hp = item.get("hyperparameter", "")
                        explanation = item.get("explanation", "")
                        if hp:
                            sections.append(f"- {hp}")
                            if explanation:
                                sections.append(f"  {explanation}")

            arch_list = later_section.get("architectures", [])
            if arch_list:
                sections.append("\n### Architecture NICE_TO_HAVE Recommendations")
                for item in arch_list:
                    if isinstance(item, dict):
                        arch = item.get("architecture", "")
                        explanation = item.get("explanation", "")
                        if arch:
                            sections.append(f"- {arch}")
                            if explanation:
                                sections.append(f"  {explanation}")

        # Inference strategies
        inference = self.later_recommendations.get("inference_strategies", {})
        if inference and "NICE_TO_HAVE" in inference:
            later_section = inference["NICE_TO_HAVE"]
            if "inference_strategies" in later_section:
                sections.append("\n## Inference Strategies NICE_TO_HAVE Recommendations")
                strategies = later_section["inference_strategies"]
                if isinstance(strategies, list):
                    for item in strategies:
                        if isinstance(item, dict):
                            strategy = item.get("strategy", "")
                            explanation = item.get("explanation", "")
                            if strategy:
                                sections.append(f"- {strategy}")
                                if explanation:
                                    sections.append(f"  {explanation}")

        return "\n".join(sections) if sections else "No NICE_TO_HAVE recommendations available."

    @staticmethod
    def _format_with_line_numbers(code: str) -> str:
        lines = code.splitlines()
        return "\n".join(f"{idx:04d}: {line}" for idx, line in enumerate(lines, start=1))

    def _replace_last_message_with_code(self, messages: list[dict], version_folder: Path) -> None:
        """Replace the last assistant message (diff) with full code (headers stripped).

        Reads train.py from version folder, strips headers, and replaces
        the last message in history.

        Args:
            messages: Message history list to modify in-place
            version_folder: Path to version folder (e.g., outputs/16_2/1/)
        """
        if not messages or messages[-1].get("role") != "assistant":
            self.logger.warning("Cannot replace last message - not an assistant message")
            return

        from utils.code_utils import strip_header_from_code

        # Read and strip headers from train.py
        train_py_path = version_folder / "train.py"
        if train_py_path.exists():
            train_py_stripped = strip_header_from_code(train_py_path)
        else:
            self.logger.warning("train.py not found in %s", version_folder)
            train_py_stripped = ""

        # Reconstruct markdown with full code (headers stripped)
        content = f"```python\n{train_py_stripped}\n```"

        # Replace the content
        messages[-1]["content"] = content
        self.logger.debug("Replaced last message (diff) with full code from %s", version_folder)

    @weave.op()
    def _generate_code(self, instructions: str, messages: list[dict[str, str]], expect_patch: bool = False) -> tuple[list[dict], dict[str, str]]:
        """Generate code using structured output.

        Returns:
            Tuple of (message_history_entry, code_dict)
            - message_history_entry: List with assistant message containing markdown
            - code_dict: Dict with 'train_py' key
        """
        self.logger.info("Requesting code generation from model for iteration %s", self.iteration)

        # Choose schema based on patch mode
        from schemas.developer import CodeGeneration, CodePatch
        schema = CodePatch if expect_patch else CodeGeneration
        self.logger.debug("Using schema: %s", schema.__name__)

        # Detect provider
        provider = detect_provider(_DEVELOPER_MODEL)

        if provider == "openai":
            result = call_llm_with_retry(
                model=_DEVELOPER_MODEL,
                instructions=instructions,
                tools=[],
                messages=messages,
                web_search_enabled=True,
                text_format=schema
            )
        elif provider == "anthropic":
            result = call_llm_with_retry_anthropic(
                model=_DEVELOPER_MODEL,
                instructions=instructions,
                tools=[],
                messages=messages,
                web_search_enabled=True,
                text_format=schema
            )
        elif provider == "google":
            result = call_llm_with_retry_google(
                model=_DEVELOPER_MODEL,
                system_instruction=instructions,
                messages=messages,
                enable_google_search=True,
                text_format=schema
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        self.logger.info("Model response received for iteration %s", self.iteration)

        # Extract code dict from Pydantic model
        code_dict = result.model_dump()
        self.logger.debug("Generated code dict keys: %s", list(code_dict.keys()))

        # Extract code from markdown backticks (handles both full code and patch modes)
        raw_content = code_dict.get('train_py', '')
        if raw_content:
            extracted = extract_structured_code(raw_content)
            if extracted and extracted != raw_content:
                code_dict['train_py'] = extracted
                self.logger.debug("Extracted code from markdown (%d chars -> %d chars)", len(raw_content), len(extracted))

        # Reconstruct markdown for message history
        # For patches: show diff, will be replaced with full code after successful application
        # For full code: show python
        if expect_patch:
            train_py_diff = code_dict.get("train_py", "").strip()
            content = f"```diff\n{train_py_diff}\n```" if train_py_diff else "No changes"
        else:
            content = f"```python\n{code_dict['train_py']}\n```"

        # Create message history entry
        output = [append_message(provider, "assistant", content)]

        return output, code_dict

    @staticmethod
    def _normalize_diff_payload(base_path: Path, diff_text: str) -> Optional[str]:
        return normalize_diff_payload(base_path, diff_text)

    def _apply_structured_patch(self, base_version: int, patch_dict: dict[str, str]) -> tuple[Optional[dict[str, str]], Optional[str]]:
        """Apply diff to train.py from base version folder.

        Reads from base_version folder (e.g., 16_2/1/) and returns patched content
        to be written to target version folder (e.g., 16_2/2/) by _write_code().

        Args:
            base_version: Version number to patch from
            patch_dict: Dict with 'train_py' key containing diff string
                       (empty string means file unchanged)

        Returns:
            Tuple of (code_dict, error_msg):
            - code_dict: Dict with 'train_py' key containing patched content.
                        Empty string means file unchanged. None if patch application fails.
            - error_msg: Error message describing patch failures, or None if all succeeded.
        """
        if base_version <= 0:
            self.logger.warning("Patch requested but base version is invalid: %s", base_version)
            return None, f"Invalid base version: {base_version}"

        # Base version folder
        base_folder = self.outputs_dir / str(base_version)
        if not base_folder.exists():
            self.logger.warning("Patch requested but base folder does not exist: %s", base_folder)
            return None, f"Base folder not found: {base_folder}"

        # Read previous version's train.py
        train_path = base_folder / "train.py"
        base_train = train_path.read_text() if train_path.exists() else ""

        # Extract diff from patch_dict
        train_diff = patch_dict.get("train_py", "").strip()

        result = {"train_py": ""}
        errors = []

        # Apply patch to train.py if diff provided
        if train_diff:
            import tempfile
            import subprocess
            import os

            # Write base train.py to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(base_train)
                temp_train_path = Path(f.name)

            # Normalize diff to fix incorrect line numbers
            from utils.diffs import normalize_diff_payload
            normalized_diff = normalize_diff_payload(temp_train_path, train_diff)

            if normalized_diff is None:
                error_msg = "Failed to normalize diff for train.py"
                self.logger.error(error_msg)
                errors.append(f"train.py: {error_msg}")
                os.unlink(temp_train_path)
            else:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
                    f.write(normalized_diff)
                    temp_patch = f.name

                try:
                    proc = subprocess.run(['patch', str(temp_train_path), temp_patch], check=True, capture_output=True, text=True)
                    with open(temp_train_path, 'r') as f:
                        result["train_py"] = f.read()
                    self.logger.info("Successfully patched train.py from version %d", base_version)
                except subprocess.CalledProcessError as e:
                    error_msg = f"Patch failed for train.py:\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}"
                    self.logger.error(error_msg)
                    errors.append(error_msg)
                finally:
                    os.unlink(temp_train_path)
                    os.unlink(temp_patch)

        # Return result and error message
        if errors:
            error_summary = "\n\n".join(errors)
            return result if any(result.values()) else None, error_summary
        return result, None

    def _append_patch_directive(self, instruction: str, version: int) -> str:
        if not self.patch_mode_enabled:
            return instruction
        instruction = instruction.replace("Please modify your code to fix the error!", "Please write a git diff within ```diff to fix the error!")
        instruction = instruction.replace("Please regenerate the script addressing the above guardrail issues.", "Please write a git diff within ```diff to fix the above issues.")
        base_filename = self._code_filename(version)
        directive = prompt_patch_mode_directive(base_filename)
        return f"{instruction}\n\n{directive}"

    def _postprocess_code(self, code: str) -> tuple[str, int]:
        """Insert resource allocation and BASE_DIR setup at the top of generated code.

        Returns:
            Tuple of (postprocessed_code, num_header_lines_added)
        """
        # Check if the code already has the resource allocation header (e.g., from patch mode)
        # Must check for actual assignment of CUDA_VISIBLE_DEVICES, not just reading it
        has_cpu_affinity = 'psutil.Process(os.getpid()).cpu_affinity' in code
        has_cuda_assignment = 'os.environ["CUDA_VISIBLE_DEVICES"]' in code
        has_base_dir = 'BASE_DIR = "task/' in code

        if has_cpu_affinity and has_base_dir and has_cuda_assignment:
            self.logger.debug("Code already contains resource allocation header, skipping postprocessing")
            return code, 0

        # Build the resource allocation header
        lines = []
        lines.append("import os")

        # CPU affinity
        if self.cpu_core_range is not None:
            lines.append("import psutil  # For CPU affinity")
            lines.append("")
            lines.append("# CPU affinity (pin to specific cores to prevent resource overlap)")
            lines.append(f"psutil.Process(os.getpid()).cpu_affinity({self.cpu_core_range})")

        # GPU assignment (works for both MIG and multi-GPU)
        if self.gpu_identifier is not None:
            gpu_device = self.gpu_identifier
        else:
            gpu_device = '0'  # Default to GPU 0

        lines.append(f'os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu_device}"')
        lines.append(f'BASE_DIR = "task/{self.slug}" if not os.getenv(\'KAGGLE_KERNEL_RUN_TYPE\') else "/kaggle/input/{self.slug}"')
        lines.append("")

        header = "\n".join(lines)
        num_header_lines = len(lines)

        # Insert header at the top of the code
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
        # Extract base iteration number (before any underscore)
        base_iteration = iteration_str.split('_')[0]
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
        # Create version folder
        version_folder = self.outputs_dir / str(version)
        version_folder.mkdir(parents=True, exist_ok=True)
        self.logger.info("Writing code to version folder: %s", version_folder)

        # Postprocess and write train.py
        train_py = code_dict.get('train_py', '')
        if not train_py:
            raise ValueError("train_py is required in code_dict")

        postprocessed_code, num_header_lines = self._postprocess_code(train_py)
        train_path = version_folder / 'train.py'
        train_path.write_text(postprocessed_code)
        self.logger.debug("Written train.py (%d characters)", len(postprocessed_code))

        # Write metadata with header line count
        metadata_path = version_folder / 'train.json'
        metadata_path.write_text(json.dumps({"num_header_lines": num_header_lines}))
        self.logger.debug("Written train.json with %d header lines", num_header_lines)

        # Copy shared files from parent iteration folder
        # Determine parent iteration folder (e.g., outputs/16 if current is outputs/16_2)
        parent_iteration = self._get_parent_iteration_folder()

        # Copy cv_splits.json if it exists
        cv_splits_src = parent_iteration / 'cv_splits.json'
        if cv_splits_src.exists():
            cv_splits_dst = version_folder / 'cv_splits.json'
            shutil.copy2(cv_splits_src, cv_splits_dst)
            self.logger.debug("Copied cv_splits.json from %s", cv_splits_src)
        else:
            self.logger.warning("cv_splits.json not found in parent iteration folder: %s", cv_splits_src)

        # Copy metric.py if it exists
        metric_src = parent_iteration / 'metric.py'
        if metric_src.exists():
            metric_dst = version_folder / 'metric.py'
            shutil.copy2(metric_src, metric_dst)
            self.logger.debug("Copied metric.py from %s", metric_src)
        else:
            self.logger.warning("metric.py not found in parent iteration folder: %s", metric_src)

        return version_folder

    @staticmethod
    def _read_code_metadata(code_path: Path) -> dict:
        """Read metadata JSON for a code file.

        Args:
            code_path: Path to the .py code file

        Returns:
            Dict with metadata (e.g., {"num_header_lines": 5})
            Returns {"num_header_lines": 0} if metadata file doesn't exist
        """
        metadata_path = code_path.with_suffix('.json')
        if not metadata_path.exists():
            return {"num_header_lines": 0}

        with open(metadata_path, 'r') as f:
            return json.load(f)

    def _log_attempt_score(self, attempt: int, score: Optional[float]) -> None:
        """Send attempt/score metrics to wandb while guarding against logging errors."""
        try:
            wandb.log({"attempt": attempt, "score": score})
            self.logger.debug("Logged attempt %s with score %s to wandb", attempt, score)
        except Exception:
            self.logger.exception("Failed to log attempt %s metrics to wandb", attempt)

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

    def _execute_code(self, code_path: Path, version: int) -> str:
        """Execute code and return execution output.

        Args:
            code_path: Path to the code file to execute (e.g., outputs/16_2/1/train.py)
            version: Version number for logging

        Returns:
            Execution output string
        """
        timeout_seconds = self._get_code_timeout()

        output = execute_code(
            str(code_path),
            timeout_seconds=timeout_seconds,
            conda_env=self.conda_env
        )
        self.logger.info("Execution output captured for version v%s", version)
        self.logger.debug("Execution output: %s", output)

        return output

    def _format_suggestion_entry(self, suggestion: str, previous_score: Optional[float], current_score: Optional[float]) -> str:
        """Format a suggestion entry with score impact information.

        Returns formatted string like:
        - "suggestion (score improved by DDD: XXX -> YYY)"
        - "suggestion (score worsened by DDD: XXX -> YYY)"
        - "suggestion (score remained the same: XXX -> YYY)"
        """
        prev_display = self._format_score_value(previous_score)
        curr_display = self._format_score_value(current_score)

        # If either score is N/A, just append scores without calculating delta
        if prev_display == "N/A" or curr_display == "N/A":
            return f"{suggestion} (score: {prev_display} -> {curr_display})"

        # Calculate delta
        delta = current_score - previous_score
        abs_delta = abs(delta)

        # Determine if it's an improvement based on metric direction
        if abs_delta < 1e-9:  # Essentially the same
            impact = "remained the same"
            return f"{suggestion} (score {impact}: {prev_display} -> {curr_display})"
        elif (delta > 0 and not self.is_lower_better) or (delta < 0 and self.is_lower_better):
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

        # Add to instance-level blacklist
        if entry not in self.blacklisted_ideas:
            self.blacklisted_ideas.append(entry)

    def _register_shared_suggestion(self, suggestion: str, previous_score: Optional[float], current_score: Optional[float], is_blacklisted: bool = False) -> None:
        """Register suggestion outcome to shared pool with model name and score impact.

        Format: "Model <model_name> tried <suggestion> (score improved/worsened/remained by X: A -> B)"
        """
        if not suggestion:
            return

        # Format the suggestion entry with model name and score impact
        prev_display = self._format_score_value(previous_score)
        curr_display = self._format_score_value(current_score)

        # Build the entry with model name
        model_prefix = f"Model {self.model_name} tried"

        if prev_display == "N/A" or curr_display == "N/A":
            entry = f"{model_prefix} {suggestion} (score: {prev_display} -> {curr_display})"
        else:
            delta = current_score - previous_score
            abs_delta = abs(delta)

            if abs_delta < 1e-9:
                impact = "remained the same"
                entry = f"{model_prefix} {suggestion} (score {impact}: {prev_display} -> {curr_display})"
            elif (delta > 0 and not self.is_lower_better) or (delta < 0 and self.is_lower_better):
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

    def _find_most_recent_valid_version(self, current_version: int) -> tuple[int, Optional[str]]:
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
        # Iterate backwards from current_version-1 to find most recent valid version
        for v in range(current_version - 1, 0, -1):
            # Check if version folder exists
            version_folder = self.outputs_dir / str(v)
            if not version_folder.exists():
                continue

            # Check if submission.csv exists (indicates successful execution)
            submission_path = version_folder / 'submission.csv'
            if not submission_path.exists():
                self.logger.debug(f"v{v} skipped: no submission.csv in {version_folder}")
                continue

            # Check if blacklisted
            is_blacklisted = v in self.blacklisted_versions
            if is_blacklisted:
                self.logger.debug(f"v{v} skipped: blacklisted")
                continue

            # Valid version found - read train.py
            self.logger.info(f"Found most recent valid version for rollback: v{v}")

            try:
                # Read train.py with headers stripped
                train_path = version_folder / 'train.py'
                if train_path.exists():
                    train_py = strip_header_from_code(train_path)
                else:
                    train_py = "# train.py not found"

                # Construct markdown
                code_content = f"train.py:\n```python\n{train_py}\n```"

                return v, code_content
            except Exception as e:
                self.logger.error(f"Failed to read code from v{v}: {e}")
                continue

        # No valid version found
        self.logger.warning("No valid rollback version found; falling back to v1 (initial version)")
        return 1, None

    def _extract_final_summary(self, red_flags_response: str) -> str:
        """Extract just the Final Summary section from red flags response.

        Args:
            red_flags_response: Full markdown response from search_red_flags()

        Returns:
            The Final Summary text, or full response if section not found
        """
        # Find "### Final Summary" section
        match = re.search(r'### Final Summary\s*\n(.+)', red_flags_response, re.DOTALL)

        if match:
            summary = match.group(1).strip()
            self.logger.info("Extracted Final Summary (%d chars)", len(summary))
            return summary
        else:
            # Fallback: return entire response if section not found
            self.logger.warning("Could not extract Final Summary section, using full response")
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
        from pathlib import Path

        lines = []

        # 1. Base directory listing (train/, test/, root files)
        lines.append("=== Base Directory (task data) ===")
        base_listing = self._build_base_dir_listing()
        lines.append(base_listing)

        # 2. External data directories
        if self.external_data_listing and self.external_data_listing != "No external data directories found.":
            lines.append("\n=== External Data ===")
            lines.append(self.external_data_listing)

        # 3. Models directory for current version
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
        from pathlib import Path

        lines = []
        base_dir = Path(self.base_dir)

        # Get root-level files
        root_files = sorted([f.name for f in base_dir.iterdir() if f.is_file()])

        # Show root directory
        lines.append("./")
        for file_name in root_files:
            lines.append(f"    {file_name}")

        # Show train/ directory if it exists
        train_dir = base_dir / "train"
        if train_dir.exists() and train_dir.is_dir():
            lines.append("    train/")
            train_listing = _build_directory_listing(str(train_dir))
            # Indent the train listing
            for line in train_listing.split('\n'):
                if line.strip():
                    lines.append(f"    {line}")

        # Show test/ directory if it exists
        test_dir = base_dir / "test"
        if test_dir.exists() and test_dir.is_dir():
            lines.append("    test/")
            test_listing = _build_directory_listing(str(test_dir))
            # Indent the test listing
            for line in test_listing.split('\n'):
                if line.strip():
                    lines.append(f"    {line}")

        return "\n".join(lines)

    def _call_sota_suggestions(self, attempt_number: int = 1, **kwargs):
        """
        Call SOTA suggestions tool with appropriate parameters.

        Can be overridden by subclasses to modify behavior (e.g., is_ensemble).

        Args:
            attempt_number: Which attempt this is (1, 2, 3, ...) for interleaving strategy
            **kwargs: Arguments to pass to search_sota_suggestions

        Returns:
            Parsed SOTAResponse object or None if parsing fails
        """
        return search_sota_suggestions(attempt_number=attempt_number, **kwargs)

    def _evaluate_submission(
        self,
        code_clean: str,
        version: int,
        attempt: int
    ) -> tuple[str, float, Optional[int], Optional[float], bool]:
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
        submission_path = self.outputs_dir / self._submission_filename(version)
        code_context = "<code>\n" + code_clean + "\n</code>\n"

        run_score = float("inf") if self.is_lower_better else float("-inf")

        # Initialize variables that will be returned (set properly if submission exists)
        previous_successful_version: Optional[int] = None
        base_score: Optional[float] = None
        submission_exists = submission_path.exists()

        if submission_exists:
            self.latest_submission_path = submission_path
            # Mark this version as successful (generated submission)
            self.successful_versions.add(version)
            self.logger.info(
                "Submission detected at %s after attempt %s (marking v%s as successful)",
                submission_path, attempt, version
            )
            previous_successful_version = self.last_successful_version  # Initialize before try

            if _USE_VALIDATION_SCORE:
                # Use validation score from train_stats.json instead of parsing logs
                self.logger.info("Using validation score from train_stats.json (use_validation_score=True)")
                try:
                    # Read train_stats.json from version folder
                    version_folder = self.outputs_dir / str(version)
                    train_stats_path = version_folder / 'train_stats.json'

                    if train_stats_path.exists():
                        with open(train_stats_path, 'r') as f:
                            train_stats = json.load(f)

                        # Extract cv_mean as the validation score
                        run_score = train_stats.get('cv_mean')

                        if run_score is not None:
                            self._log_attempt_score(attempt, run_score)
                            self.logger.info("Your validation score is %s", run_score)
                            # Store score for this version and track for comparison
                            self.version_scores[version] = run_score
                            self.last_successful_version = version
                        else:
                            self.logger.warning("cv_mean not found in train_stats.json for version %s", version)
                    else:
                        self.logger.warning("train_stats.json not found for version %s", version)
                except Exception as exc:
                    self.logger.exception("Failed to read train_stats.json: %s", exc)

                code_context += f"<validation_score>\n{run_score}\n</validation_score>\n"
            else:
                # Use MLE-bench grading (original behavior)
                grade_feedback = ""
                try:
                    info, grade_feedback, returncode, stderr = run_grade(str(submission_path), self.slug)
                    try:
                        self.logger.info("Grade feedback: %s", grade_feedback)
                    except Exception:
                        self.logger.debug("Failed to log grade feedback for version %s", version)
                    if returncode != 0:
                        self.logger.warning(
                            "Grading command returned non-zero exit (%s). stderr=\n%s",
                            returncode,
                            stderr,
                        )
                    else:
                        self.logger.info("Grading command completed successfully for version %s", version)
                        run_score = info.get('score') if info else None
                        self._log_attempt_score(attempt, run_score)
                        self.logger.info("Your result on the test set is %s", run_score)
                        # Store score for this version and track for comparison
                        if run_score is not None:
                            self.version_scores[version] = run_score
                            self.last_successful_version = version
                except Exception as exc:
                    grade_feedback = f"Failed to run grading tool: {exc}"
                    self.logger.exception("Grading command failed for version %s", version)

                code_context += f"<leaderboard_score>\n{run_score}\n</leaderboard_score>\n"

            # Compare against most recent successful version (with a score)
            if previous_successful_version is not None:
                base_version = previous_successful_version
                base_score = self.version_scores[base_version]
            else:
                # This is the first version with a score, no comparison base
                base_version = None
                base_score = None

            run_score_display = self._format_score_value(run_score)
            improvement = self._is_improvement(run_score, base_score) if base_score is not None else False

            # Always check and update global best, regardless of base comparison
            if self._is_improvement(run_score, self.best_score):
                self.best_score = run_score
                self.best_version = version
                self.best_code = code_clean
                self.best_code_file = self._code_filename(version)
                self.logger.info("New global best achieved: %s (version %s)", run_score, version)

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

            # Build analysis message with context
            analysis_msg = f"The current score is {run_score_display}."

            # Add previous score context if available
            if base_score is not None:
                base_score_display = self._format_score_value(base_score)
                analysis_msg += f" Your score before implementing this suggestion was {base_score_display}."

            # Add target context
            if self.gold_threshold is not None and run_score is not None:
                # Compare current score to target (accounting for metric direction)
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

        # Return previous_successful_version and base_score for use in run() method
        # These are None if no submission was generated or if this is the first successful version
        return code_context, run_score, previous_successful_version, base_score, submission_exists

    def _gather_sota_feedback(self, code_context: str, version: int, attempt_number: int = 1):
        """Gather SOTA feedback through red flags analysis and SOTA suggestions.

        Args:
            code_context: Code with execution logs and analysis
            version: Current version number to include models_{version}/ in directory listing
            attempt_number: Which attempt this is (1, 2, 3, ...) for interleaving strategy

        Returns:
            Parsed SOTAResponse object or None if gathering/parsing failed
        """
        try:
            # Format LATER recommendations for context
            later_context = self._format_later_recommendations()

            # Build extended directory listing including models_{version}/
            extended_listing = self._build_extended_data_listing(version)

            # STAGE 1: Identify red flags via direct analysis
            self.logger.info("Stage 1: Identifying red flags via direct analysis...")

            # Collect training visualizations and stats from version folder
            version_folder = self.outputs_dir / str(version)
            training_images = []
            for img_name in ['loss_curve.png', 'metric_curve.png']:
                img_path = version_folder / img_name
                if img_path.exists():
                    training_images.append(img_path)
                else:
                    self.logger.debug(f"Training image not found: {img_path}")

            # Read train_stats.json if available
            train_stats_path = version_folder / 'train_stats.json'
            train_stats = None
            if train_stats_path.exists():
                try:
                    with open(train_stats_path, 'r') as f:
                        train_stats = json.load(f)
                    self.logger.info("Loaded train_stats.json with %d keys", len(train_stats))
                except Exception as e:
                    self.logger.warning(f"Failed to read train_stats.json: {e}")

            red_flags_response = search_red_flags(
                description=self.description,
                context=code_context,
                images=training_images if training_images else None,
                train_stats=train_stats,
            )
            self.logger.info("Red flags response length: %d chars", len(red_flags_response))

            # Extract Final Summary from red flags response
            final_summary = self._extract_final_summary(red_flags_response)

            # STAGE 2: Generate SOTA suggestions based on red flags
            self.logger.info("Stage 2: Generating SOTA suggestions based on red flags...")

            # Get shared suggestions from all parallel models
            shared_suggestions = self._get_shared_suggestions()

            self.logger.info("Using %d shared suggestions from all models (including this one)",
                           len(shared_suggestions))

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
                file_suffix=str(self.iteration),  # Pass iteration as file suffix to prevent race conditions
                images=training_images if training_images else None,
                train_stats=train_stats,
            )
            return sota_response
        except Exception as exc:
            self.logger.exception("Failed to fetch red flags or SOTA suggestions")
            return None

    @weave.op()
    def run(self, max_time_seconds: int = 6 * 3600) -> tuple[float, Optional[str], list[str], list[str]]:
        self.logger.info(
            "Starting developer run for slug=%s iteration=%s",
            self.slug,
            self.iteration,
        )
        self.logger.info("cpu core range: %s", self.cpu_core_range)
        self.logger.info("gpu identifier: %s (mode: %s)", self.gpu_identifier, self.gpu_isolation_mode)
        
        start_time = time.time()
        deadline = start_time + max_time_seconds

        run_score = 0

        # Enable multi-fold from start if using validation scores (need proper CV for reliable scores)
        initial_allow_multi_fold = _USE_VALIDATION_SCORE
        if initial_allow_multi_fold:
            self.logger.info("Multi-fold training enabled from start (use_validation_score=True)")
        system_prompt = self._compose_system(allow_multi_fold=initial_allow_multi_fold)
        user_prompt = self._build_user_prompt(version=1)

        # Detect provider early for consistent message formatting
        provider = detect_provider(_DEVELOPER_MODEL)
        input_list = [append_message(provider, "user", user_prompt)]

        # Log external data listing and plan content (passed from orchestrator)
        self.logger.info("External data listing (%d chars): %s", len(self.external_data_listing), self.external_data_listing[:200] + "..." if len(self.external_data_listing) > 200 else self.external_data_listing)
        self.logger.info("Plan content (%d chars): %s", len(self.plan_content), self.plan_content[:200] + "..." if len(self.plan_content) > 200 else self.plan_content)

        attempt = 0
        sota_suggestions_call_id = 0
        while True:
            now = time.time()
            if max_time_seconds is not None and now >= deadline:
                self.logger.info("Time budget exhausted (%.2f minutes)", (deadline - start_time) / 60.0)
                break

            # Trim to keep at most 60 messages, ensuring first message is from "user"
            try:
                if len(input_list) > 60:
                    # Trim from the front
                    input_list = input_list[-60:]
                    # Ensure first message is from user (required by API)
                    while input_list:
                        try:
                            if input_list[0].get("role") == "user":
                                break  # Found user message, stop trimming
                            input_list.pop(0)
                        except Exception as e:
                            self.logger.exception("Message has no attribute role: %s", e)
                            input_list.pop(0)

                    self.logger.info("Trimmed input messages to last 60 for attempt %s", attempt + 1)

            except Exception as e:
                self.logger.exception("Failed to trim input messages: %s", e)
                break

            attempt += 1

            # Rebuild system prompt for attempt 2+ to enable multi-fold training (unless already enabled)
            if attempt == 2 and not initial_allow_multi_fold:
                self.logger.info("Rebuilding system prompt for attempt 2+ with multi-fold enabled")
                system_prompt = self._compose_system(allow_multi_fold=True)

            artifact = wandb.Artifact(f'{self.iteration}-{self.slug}', type='files')

            minutes_left = ((deadline - now) / 60.0) if max_time_seconds is not None else float('inf')
            try:
                self.logger.info("Attempt %s (time left ~%.1f min)", attempt, minutes_left)
            except Exception:
                self.logger.info("Attempt %s", attempt)
            version = attempt
            expect_patch = self.patch_mode_enabled and attempt > 1

            patch_retry_count = 0
            max_patch_retries = 2
            code_dict = None

            while True:
                response_output, generated = self._generate_code(instructions=system_prompt, messages=input_list, expect_patch=expect_patch)
                input_list += response_output

                if expect_patch:
                    # Determine base version for patch application
                    preferred_base = self.next_patch_base_version if self.next_patch_base_version else (version - 1)
                    base_folder = self.outputs_dir / str(preferred_base)
                    if not base_folder.exists():
                        self.logger.warning(
                            "Configured patch base v%s folder not found; falling back to previous version v%s",
                            preferred_base,
                            version - 1,
                        )
                        preferred_base = version - 1
                    base_version = preferred_base
                    self.logger.info("Applying patch relative to base v%s -> target v%s", base_version, version)

                    # Apply structured patch (diff for train.py)
                    code_dict, error_msg = self._apply_structured_patch(base_version, generated)

                    if error_msg is None:
                        # Patch succeeded
                        self.logger.info("Patch applied successfully")
                        break

                    # Patch failed - error_msg contains detailed traceback
                    patch_retry_count += 1
                    self.logger.warning("Patch application failed (attempt %d/%d)", patch_retry_count, max_patch_retries)

                    if patch_retry_count < max_patch_retries:
                        # Retry once: pass actual error back to LLM to fix the patch
                        retry_msg = f"Patch application failed with the following errors:\n\n{error_msg}\n\nPlease fix the diff syntax and regenerate the patch."
                        input_list.append(append_message(provider, "user", retry_msg))
                        # Still expect patch for retry
                        continue
                    else:
                        # Second failure: fallback to full code generation
                        self.logger.warning("Patch failed twice; requesting full code instead")
                        fallback_msg = f"Patch application failed multiple times:\n\n{error_msg}\n\nPlease provide the complete train.py file instead of a diff."
                        input_list.append(append_message(provider, "user", fallback_msg))
                        expect_patch = False
                        continue
                else:
                    # Full code generation (iteration 1 or after patch failures)
                    code_dict = generated
                    break

            # Write code to folder structure (train.py)
            version_folder = self._write_code(code_dict, version)

            # If patch succeeded, replace the diff message with full code (headers stripped)
            if expect_patch and patch_retry_count > 0 and error_msg is None:
                self._replace_last_message_with_code(input_list, version_folder)

            # ---------------------------
            # Pre-exec guardrail checks
            # ---------------------------
            train_py_path = version_folder / "train.py"
            with open(str(train_py_path), "r") as f:
                code_text = f.read()

            # Strip the header lines to get the clean LLM-generated code
            code_clean = strip_header_from_code(train_py_path)

            guard_report = evaluate_guardrails(
                code_text=code_text,
                enable_logging_guard=_ENABLE_LOGGING_GUARD,
            )

            try:
                self.logger.info("Guardrail decision v%s: %s", version, guard_report.get("decision"))
            except Exception:
                self.logger.debug("Failed to log final guardrail decision for v%s", version)

            if guard_report.get("decision") == "block":
                # Build feedback and ask for a corrected script without executing
                summary_text = build_block_summary(guard_report)
                next_log_path = self.outputs_dir / self._log_filename(version + 1)
                next_submission_path = self.outputs_dir / self._submission_filename(version + 1)
                next_version_folder = self.outputs_dir / str(version + 1)
                fix_instr = prompt_guardrail_fix_suffix(next_log_path, next_submission_path, next_version_folder)
                guardrail_prompt = summary_text + fix_instr
                base_version_for_next_patch = version
                guardrail_prompt = self._append_patch_directive(guardrail_prompt, base_version_for_next_patch)
                input_list.append(append_message(provider, "user", guardrail_prompt))
                self.next_patch_base_version = base_version_for_next_patch
                self.logger.info("Next patch will be based on v%s due to guardrail block", base_version_for_next_patch)
                self.logger.info("User prompt with guardrail feedback: %s", guardrail_prompt)
                # continue to next attempt without execution
                continue

            # Execute the code
            output = self._execute_code(train_py_path, version)

            # Evaluate submission and build enriched context
            code_context, run_score, previous_successful_version, base_score, submission_exists = self._evaluate_submission(code_clean, version, attempt)

            # Only continue gathering feedback if THIS attempt produced a submission
            if submission_exists:
                # Increment SOTA suggestions call counter (separate from attempt counter)
                sota_suggestions_call_id += 1
                # Gather SOTA feedback (red flags + suggestions)
                # sota_response is now a parsed SOTAResponse object (or None)
                sota_response = self._gather_sota_feedback(code_context, version=version, attempt_number=sota_suggestions_call_id)

                # Extract fields directly from parsed object
                if sota_response:
                    suggestion_text = sota_response.suggestion.strip()
                    blacklist_flag = bool(sota_response.blacklist)
                    blacklist_reason = sota_response.blacklist_reason.strip()
                    suggestion_code = sota_response.suggestion_code.strip() if hasattr(sota_response, 'suggestion_code') else ""
                else:
                    suggestion_text = ""
                    blacklist_flag = False
                    blacklist_reason = ""
                    suggestion_code = ""

                self.logger.info("SOTA suggestion: %s (blacklist=%s, code_len=%d)", suggestion_text, blacklist_flag, len(suggestion_code))

                # Record the previous suggestion with its score impact (before updating self.last_suggestion)
                if previous_successful_version is None:
                    # This is the first successful execution (no previous version to compare against)
                    initial_entry = f"Initial implementation (score: {self._format_score_value(run_score)})"
                    self.global_suggestions.append(initial_entry)
                    self.logger.info("Recorded initial implementation: %s", initial_entry)
                elif self.last_suggestion:
                    # We have a previous suggestion and a previous successful version to compare against
                    # base_score was already calculated above (from previous_successful_version)
                    current_score = run_score

                    suggestion_entry = self._format_suggestion_entry(self.last_suggestion, base_score, current_score)
                    self.global_suggestions.append(suggestion_entry)
                    self.logger.info("Recorded suggestion with impact: %s", suggestion_entry)

                    # Register to shared pool with model name and score impact
                    self._register_shared_suggestion(
                        self.last_suggestion,
                        base_score,
                        current_score,
                        is_blacklisted=blacklist_flag
                    )

                if blacklist_flag and self.last_suggestion:
                    self._register_blacklist(self.last_suggestion, blacklist_reason)
                    # Mark current version as blacklisted
                    self.blacklisted_versions.add(version)
                    self.logger.info(
                        "Previous suggestion marked as blacklisted: %s (reason: %s) - marking v%s as blacklisted",
                        self.last_suggestion,
                        blacklist_reason or "N/A",
                        version
                    )
                elif not blacklist_flag and self.last_suggestion and version in self.successful_versions:
                    # Register as successful idea if: not blacklisted + executed successfully
                    if self.last_suggestion not in self.successful_ideas:
                        self.successful_ideas.append(self.last_suggestion)
                        self.logger.info(
                            "Previous suggestion marked as successful: %s (v%s executed successfully and not blacklisted)",
                            self.last_suggestion,
                            version
                        )

                if suggestion_text:
                    self.logger.info("Summary of SOTA suggestion: %s", suggestion_text)
                else:
                    self.logger.info("SOTA response did not include a new suggestion summary.")

                # Check if model indicates no further suggestions are possible
                if suggestion_text and suggestion_text.strip() == "No suggestions.":
                    self.logger.info("Model indicated 'No suggestions.' - breaking out of loop and returning best result")
                    self.logger.info("Final best score: %s (version %s)", self.best_score, self.best_version)
                    break

                if suggestion_text:
                    self.last_suggestion = suggestion_text
                elif blacklist_flag:
                    # if suggestion missing but blacklist decision made, reset last suggestion
                    self.last_suggestion = None

                next_log_path = self.outputs_dir / self._log_filename(version + 1)
                next_submission_path = self.outputs_dir / self._submission_filename(version + 1)
                next_version_folder = self.outputs_dir / str(version + 1)

                suggestion_block = ""
                if suggestion_text:
                    suggestion_block += f"<suggestion>\n{suggestion_text}\n</suggestion>\n"
                else:
                    suggestion_block += "<suggestion>\nNo suggestion provided.\n</suggestion>\n"

                if suggestion_code:
                    suggestion_block += "Suggested code snippet:\n```python\n" + suggestion_code + "\n```\n"
                else:
                    suggestion_block += "Suggested code snippet: No code provided.\n"

                # Choose consistent base for the next patch: use most recent valid when blacklisted, else current version.
                rollback_code = None
                rollback_version = None
                if blacklist_flag:
                    rollback_version, rollback_code = self._find_most_recent_valid_version(version)
                    base_version_for_next_patch = rollback_version
                else:
                    base_version_for_next_patch = version

                next_instr = (
                    f"{suggestion_block}\n"
                    f"Remember:\n"
                    f"- write logs to {next_log_path}\n"
                    f"- save artifacts to {next_version_folder}/: submission.csv, valid_preds.csv, train_stats.json, models, loss_curve.png, metric_curve.png"
                )
                next_instr = self._append_patch_directive(next_instr, base_version_for_next_patch)

                if blacklist_flag and rollback_code:
                    input_list.append(append_message(provider, "user", 'The previous code has been blacklisted. Here is the most recent valid (successful and non-blacklisted) version for your reference (please start work from this version): \n' + rollback_code))
                elif blacklist_flag:
                    self.logger.warning("Blacklist triggered but no valid rollback version available")

                input_list.append(append_message(provider, "user", next_instr))
                self.next_patch_base_version = base_version_for_next_patch
                self.logger.info("Next patch will be based on v%s (blacklist=%s)", base_version_for_next_patch, blacklist_flag)

            else:
                next_log_path = self.outputs_dir / self._log_filename(version + 1)
                next_submission_path = self.outputs_dir / self._submission_filename(version + 1)
                next_version_folder = self.outputs_dir / str(version + 1)

                # Check if this is a timeout or OOM error
                is_timeout = "Code execution timed out after" in output
                is_oom = "CUDA out of memory" in output or "OutOfMemoryError" in output

                if is_timeout or is_oom:
                    # For timeout/OOM errors, run red flags analysis to diagnose issues
                    error_type = "Timeout" if is_timeout else "OOM"
                    self.logger.info(f"{error_type} detected - running red flags analysis on logs and code")

                    # Add error context to code_context for red flags analysis
                    if is_timeout:
                        error_context = "\n<timeout_error>\nThe script was not able to execute within 1 hour. Please investigate.\n</timeout_error>\n"
                    else:
                        error_context = "\n<oom_error>\nCUDA out of memory error detected. The model or batch size is too large for available GPU memory. Please investigate.\n</oom_error>\n"

                    code_context_error = code_context + error_context

                    # Try to read train_stats.json if available (may not exist for timeout/OOM)
                    train_stats_path = version_folder / 'train_stats.json'
                    train_stats = None
                    if train_stats_path.exists():
                        try:
                            with open(train_stats_path, 'r') as f:
                                train_stats = json.load(f)
                            self.logger.info("Loaded train_stats.json for error analysis (%d keys)", len(train_stats))
                        except Exception as e:
                            self.logger.warning(f"Failed to read train_stats.json: {e}")

                    try:
                        red_flags_response = search_red_flags(
                            description=self.description,
                            context=code_context_error,
                            train_stats=train_stats,
                        )
                        self.logger.info(f"Red flags analysis complete for {error_type} (length: %d chars)", len(red_flags_response))

                        # Extract Final Summary from red flags response
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
                        self.logger.exception(f"Failed to run red flags analysis for {error_type}")
                        # Fallback to basic error message
                        error_message = "TIMEOUT" if is_timeout else "OUT OF MEMORY"
                        next_instr = f"""
                        Your code FAILED during execution due to {error_message}!
                        This is the stack trace and advice on how to fix the error:
                        {output}

                        {prompt_execution_failure_suffix(next_log_path, next_submission_path, next_version_folder)}
                        """
                else:
                    # For regular bugs/errors, just show the error (web search already done in execute_code)
                    next_instr = f"""
                    Your code FAILED during execution!
                    This is the stack trace and advice on how to fix the error:
                    {output}

                    {prompt_execution_failure_suffix(next_log_path, next_submission_path, next_version_folder)}
                    """

                base_version_for_next_patch = version
                next_instr = self._append_patch_directive(next_instr, base_version_for_next_patch)
                input_list.append(append_message(provider, "user", next_instr))
                self.next_patch_base_version = base_version_for_next_patch
                self.logger.info("Next patch will be based on v%s due to execution failure (timeout=%s)", base_version_for_next_patch, is_timeout)

            self.logger.info("previous runs count: %s", len(self.previous_runs))

            # Only log specific file types to wandb artifacts
            allowed_extensions = {'.py', '.txt', '.csv'}
            for path in self.outputs_dir.iterdir():
                if not path.is_file():
                    self.logger.debug("Skipping non-file path when logging artifact: %s", path)
                    continue

                # Check if file extension is in allowed list (case-insensitive)
                if path.suffix.lower() in allowed_extensions:
                    artifact.add_file(str(path), overwrite=True)
                else:
                    self.logger.debug("Skipping file due to extension filtering: %s (extension: %s)", path.name, path.suffix)

            try:
                artifact.save()
            except Exception as e:
                self.logger.exception("Failed to save wandb artifact: %s", e)

        return self.best_score, self.best_code_file, self.blacklisted_ideas, self.successful_ideas
