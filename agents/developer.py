import difflib
import logging
import math
import os
import re
import subprocess
import textwrap
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

_PATCH_MODE_ENABLED = bool(_RUNTIME_CFG.get("patch_mode_enabled", False))

_BASE_URL = _LLM_CFG.get("base_url", "https://openrouter.ai/api/v1")
_API_KEY_ENV = _LLM_CFG.get("api_key_env", "OPENROUTER_API_KEY")
_DEVELOPER_MODEL = _LLM_CFG.get("developer_model", "google/gemini-2.5-pro")

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
        return f"""Role: Lead Developer for Machine-Learning Competition Team. Your task is to produce a single, self-contained Python script, specifically targeted at developing a solution for a Kaggle Competition.

Begin with a concise checklist (3-7 bullets) of what you will do; keep items conceptual, not implementation-level.

**Hard Constraints:**
- Deliver a single-file script.
- Utilize CUDA wherever possible.
- Insert detailed `logging.info` statements only for validation results (every fold, every model used, overall OOF). Only log other code sections (such as data loading or config setup) if they're directly relevant to validation.
- Place `logging.basicConfig()` at the very start of your code, before any other logging statements.
- Always train with `bfloat16` when using PyTorch, transformers, or other deep learning libraries. Gradient checkpointing must be disabled.
- Do **not** code any fallback methods.
- If you use LightGBM, it has to be on CPU.
- **Do not** use `transformers.Trainer` or `transformers.TrainingArguments`.
- **Do not** use `try/except` blocks to bypass exceptions.
- Log the **final validation results** after training.
- Design the pipeline so it is highly customizable (i.e., it's easy to add or swap techniques, models, etc).
- You should use pretrained models over training from scratch, whenever possible.
- If you use external datasets, make sure you are only appending them to the training set, not the validation set.
- **IMPORTANT:** At the very top, add a `DEBUG` flag. The pipeline must run sequentially twice: once with `DEBUG=True` (using a small subset of data, e.g., 256 samples and 1 epoch, but others unchanged) and then once with `DEBUG=False` (using the full training config). Clearly log when the script is in DEBUG or FULL mode.
- **IMPORTANT:** For deep learning pipelines, if at the end of the 1st epoch of fold 0, the loss or metric is NaN or exactly 0, raise an Exception to stop the run immediately.

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
BASE_DIR = "task/{self.slug}" if not os.getenv('KAGGLE_KERNEL_RUN_TYPE') else "/kaggle/input/{self.slug}"
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
        """Extract unified diff payload from the completion content."""
        if not content:
            return ""
        fenced = re.search(r"```diff\s*(.*?)\s*```", content, re.DOTALL | re.IGNORECASE)
        if fenced:
            return fenced.group(1).strip()
        generic = re.search(r"```\s*(.*?)\s*```", content, re.DOTALL)
        if generic:
            return generic.group(1).strip()
        return content.strip()

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
    def _apply_unified_diff(base_lines: list[str], diff_lines: list[str]) -> Optional[list[str]]:
        if not diff_lines:
            return None

        result: list[str] = []
        base_idx = 0

        for line in diff_lines:
            if not line:
                continue
            if line.startswith(("diff ", "index ", "--- ", "+++ ", "@@ ")):
                continue
            prefix = line[0]
            content = line[1:]
            if prefix == ' ':
                while base_idx < len(base_lines) and base_lines[base_idx] != content:
                    result.append(base_lines[base_idx])
                    base_idx += 1
                if base_idx >= len(base_lines):
                    return None
                result.append(base_lines[base_idx])
                base_idx += 1
            elif prefix == '-':
                while base_idx < len(base_lines) and base_lines[base_idx] != content:
                    result.append(base_lines[base_idx])
                    base_idx += 1
                if base_idx >= len(base_lines):
                    return None
                base_idx += 1
            elif prefix == '+':
                result.append(content)
            elif prefix == '\\':
                continue
            else:
                return None

        if base_idx < len(base_lines):
            result.extend(base_lines[base_idx:])

        return result

    @staticmethod
    def _normalize_diff_payload(base_path: Path, diff_text: str) -> Optional[str]:
        try:
            base_lines = base_path.read_text().splitlines()
        except Exception:
            logger.exception("Failed to read base file while normalizing diff: %s", base_path)
            return None
        diff_lines = diff_text.splitlines()
        fixed_lines = DeveloperAgent._fix_hunk_headers(base_lines, diff_lines)
        new_lines = DeveloperAgent._apply_unified_diff(base_lines, fixed_lines)
        if new_lines is None:
            # Try applying the corrected diff via patch tool
            corrected_text = "\n".join(fixed_lines)
            new_lines = DeveloperAgent._apply_with_patch_tool(base_lines, corrected_text, base_path.name)
        if new_lines is None:
            # Fall back to attempting original payload
            new_lines = DeveloperAgent._apply_with_patch_tool(base_lines, diff_text, base_path.name)
        if new_lines is None:
            return None
        if base_lines == new_lines:
            return None

        normalized = list(
            difflib.unified_diff(
                base_lines,
                new_lines,
                fromfile=base_path.name,
                tofile=base_path.name,
                lineterm="",
            )
        )
        if not normalized:
            return None
        return "\n".join(normalized) + "\n"

    @staticmethod
    def _fix_hunk_headers(base_lines: list[str], diff_lines: list[str]) -> list[str]:
        """Fix incorrect @@ hunk headers to be consistent with the diff body and the base file.

        Rules implemented:
        - Split the patch into hunks on lines starting with "@@".
        - For each hunk body, count minus/plus/context lines to set b and d.
        - Recompute a (old start) by locating the first non-added line (" " or "-") in base_lines,
          choosing the occurrence closest to the previous hunk's a (if any), otherwise earliest.
        - Maintain running delta (initially 0) so c = a + delta; then delta += (plus_count - minus_count).
        - Pure-additions first hunk: assume a=1 and c=1. If a later hunk has no non-added lines,
          fall back to earliest valid match (a=1).
        - Rebuild header as: @@ -{a},{b} +{c},{d} @@ and keep body lines unchanged.
        """

        header_re = re.compile(r"^@@\s*-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@")

        output_lines = []
        i = 0
        delta = 0
        hunks = []

        for i in range(len(diff_lines)):
            if re.match(header_re, diff_lines[i]):
                A, B, C, D = map(int, header_re.match(diff_lines[i]).groups())
                hunks.append([i, A, B, C, D])

        hunks.append([len(diff_lines), 0, 0, 0, 0])

        for i in range(len(hunks)-1):
            # within this hunk
            diff_hunk_l, old_A, old_B, old_C, old_D = hunks[i]
            diff_hunk_r, _, _, _, _ = hunks[i+1]
            hunk_lines = diff_lines[diff_hunk_l+1:diff_hunk_r]
            minus_count = 0
            plus_count = 0
            first_valid_line = None
            for line in hunk_lines:
                assert not re.match(header_re, line)
                if line.startswith("-"):
                    minus_count += 1
                    if first_valid_line is None:
                        first_valid_line = line[1:]
                elif line.startswith("+"):
                    plus_count += 1
                else:
                    # in old and new file
                    minus_count += 1
                    plus_count += 1
                    if first_valid_line is None:
                        first_valid_line = line

            print("First valid line:")
            if first_valid_line is None:
                # Edge case: no anchorable content; return original diff lines
                return diff_lines
            print(first_valid_line.strip())
            new_A = None
            
            for line_no in range(old_A - 15, old_A + 15):
                if line_no < 0:
                    continue
                if line_no >= len(base_lines):
                    continue
                if base_lines[line_no].strip() == first_valid_line.strip():
                    new_A = line_no + 1
                    break

            if new_A is None:
                # edge case - the content is completely wrong (patch will not work)
                return diff_lines

            new_C = new_A + delta
            new_B = minus_count
            new_D = plus_count

            delta += (plus_count - minus_count)
            diff_lines[diff_hunk_l] = f"@@ -{new_A},{new_B} +{new_C},{new_D} @@"

        print("Fixed diff lines:")
        print(diff_lines)
        return diff_lines

    @staticmethod
    def _apply_with_patch_tool(base_lines: list[str], diff_text: str, base_filename: str) -> Optional[list[str]]:
        import tempfile

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_dir_path = Path(tmp_dir)
                tmp_base = tmp_dir_path / base_filename
                tmp_base.write_text("\n".join(base_lines) + "\n")
                patched_path = tmp_dir_path / (base_filename + ".patched")
                cmd = ["patch", "-o", patched_path.name, tmp_base.name]
                try:
                    result = subprocess.run(
                        cmd,
                        input=diff_text if diff_text.endswith("\n") else diff_text + "\n",
                        text=True,
                        capture_output=True,
                        cwd=tmp_dir,
                        check=False,
                    )
                except FileNotFoundError:
                    return None
                except Exception:
                    return None
                if not patched_path.exists():
                    return None
                try:
                    patched_lines = patched_path.read_text().splitlines()
                except Exception:
                    return None
                # Validate by forward diff so we don't return identical content
                if patched_lines == base_lines:
                    return None
                return patched_lines
        except Exception:
            return None


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
        # original_payload = diff_text if diff_text.endswith("\n") else diff_text + "\n"
        # attempts.append(("original", original_payload))

        normalized_payload = self._normalize_diff_payload(base_path, diff_text)
        attempts.append(("normalized", normalized_payload))

        cmd = ["patch", "-o", output_filename, base_filename]

        for label, payload in attempts:
            logger.debug("Attempting to apply %s diff for version %s", label, target_version)
            logger.debug("Payload: %s", payload)
            # print(payload)
            if output_path.exists():
                try:
                    output_path.unlink()
                except Exception:
                    logger.exception(
                        "Failed to remove existing output file before applying %s diff: %s",
                        label,
                        output_path,
                    )
                    return None
            try:
                result = subprocess.run(
                    cmd,
                    input=payload,
                    text=True,
                    capture_output=True,
                    cwd=self.outputs_dir,
                    check=False,
                )
            except FileNotFoundError:
                logger.exception("`patch` utility not available on system path.")
                return None
            except Exception:
                logger.exception("Unexpected failure while applying patch command.")
                return None

            if result.returncode != 0:
                logger.warning(
                    "Patch command (%s diff) returned non-zero exit code %s. stderr=\n%s",
                    label,
                    result.returncode,
                    (result.stderr or "").strip(),
                )
                if output_path.exists():
                    try:
                        output_path.unlink()
                    except Exception:
                        logger.debug(
                            "Failed to clean up partial patched file after %s diff attempt: %s",
                            label,
                            output_path,
                        )
                continue

            try:
                updated_code = output_path.read_text()
            except Exception:
                logger.exception("Failed to read patched file at %s", output_path)
                return None

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
        directive = textwrap.dedent(
            f"""
**IMPORTANT**: Please write a git diff (patch) within ```diff to fix the above issues!
- Produce a unified diff (apply patch format) that updates {base_filename}. Do not include any prefixes in the diff other than {base_filename}.
- Return only the diff enclosed in ```diff fences; do not resend the full script.
- Ensure the diff applies cleanly with the `patch` utility using the file names above.
- Use standard hunk headers with explicit line numbers, e.g. @@ -12,7 +12,9 @@.
- Refer to the <previous_code_with_line_numbers> section above when calculating line numbers.

Like this
```diff
--- {base_filename}
+++ {base_filename}
@@ -1,3 +1,3 @@
 start
-first change
+new first change
 middle
@@ -7,4 +7,4 @@
 some content
-second change
+new second change
 more content
 end
 ```
""").strip()
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
        if len(self.blacklisted_ideas) > 10:
            self.blacklisted_ideas = self.blacklisted_ideas[-10:]

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
            expect_patch = self.patch_mode_enabled and attempt > 1
            while True:
                generated = self._generate_code(self.messages, expect_patch=expect_patch)
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
                    if self.messages and self.messages[-1].get("role") == "user":
                        self.messages[-1]["content"] += (
                            "\n\nPatch application failed. Ignore the diff request and return the complete updated script enclosed in triple backticks with the `python` annotation."
                        )
                    expect_patch = False
                    continue
                else:
                    code = generated
                    break

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
                        debug_json_text = '{"severity":"warn","findings":[{"rule_id":"llm_error","snippet":"N/A","rationale":"LLM call failed","suggestion":"Manually ensure DEBUG runs before FULL and loss/metric NaN or zero values raise exceptions."}]}'
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
                            "- Ensure the script runs DEBUG mode before FULL mode and raises an Exception when loss/metric is NaN or 0."
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
                guardrail_prompt = "\n".join(summary) + fix_instr
                base_version_for_next_patch = version
                guardrail_prompt = self._append_patch_directive(guardrail_prompt, base_version_for_next_patch)
                assistant_payload = self._build_assistant_payload(
                    code,
                    include_line_numbers=self.patch_mode_enabled,
                )
                self.messages.append({"role": "assistant", "content": assistant_payload})
                self.messages.append({"role": "user", "content": guardrail_prompt})
                self.next_patch_base_version = base_version_for_next_patch
                logger.info("Next patch will be based on v%s due to guardrail block", base_version_for_next_patch)
                logger.info("User prompt with guardrail feedback: %s", guardrail_prompt)
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
                    analysis_msg = (
                        f"Nice work! The score has improved to {run_score_display}. "
                        f"{target_text}"
                    )
                else:
                    logger.info(
                        "No improvement over previous best score of %s; current score is %s",
                        previous_best,
                        run_score,
                    )
                    if previous_best_display != "N/A" and run_score_display != "N/A":
                        analysis_msg = (
                            f"The latest run scored {run_score_display}, but the best remains {previous_best_display}. "
                            "Please investigate the regression before proceeding."
                        )
                    else:
                        analysis_msg = (
                            "The latest run did not improve the benchmark. Please investigate the reasons before continuing."
                        )

                code_with_logs += f"<analysis>\n{analysis_msg}\n</analysis>\n"
                if improvement:
                    self.best_code = code
                self.previous_runs.append((code, run_score))

                try:
                    plan_texts: list[str] = []
                    try:
                        if self.plan_path.exists():
                            plan_texts.append(_safe_read(str(self.plan_path)))
                    except Exception:
                        pass
                    try:
                        for extra_plan in sorted(self.outputs_dir.glob("plan_*.md")):
                            plan_texts.append(_safe_read(str(extra_plan)))
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

                self.messages = self.messages[:2]
                if blacklist_flag and self.best_code:
                    base_code = self.best_code
                else:
                    base_code = code
                assistant_payload = self._build_assistant_payload(
                    base_code,
                    include_line_numbers=self.patch_mode_enabled,
                )
                self.messages.append({'role': 'assistant', 'content': assistant_payload})
                self.messages.append({'role': 'user', 'content': next_instr})
                self.next_patch_base_version = base_version_for_next_patch
                logger.info("Next patch will be based on v%s (blacklist=%s)", base_version_for_next_patch, blacklist_flag)

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
                base_version_for_next_patch = version
                next_instr = self._append_patch_directive(next_instr, base_version_for_next_patch)
                assistant_payload = self._build_assistant_payload(
                    code,
                    include_line_numbers=self.patch_mode_enabled,
                )
                self.messages.append({'role': 'assistant', 'content': assistant_payload})
                self.messages.append({'role': 'user', 'content': next_instr})
                self.next_patch_base_version = base_version_for_next_patch
                logger.info("Next patch will be based on v%s due to execution failure", base_version_for_next_patch)

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
