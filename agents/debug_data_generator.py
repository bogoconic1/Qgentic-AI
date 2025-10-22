"""Debug data generator for rapid development iteration."""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

from project_config import get_config
from tools.helpers import call_llm_with_retry
from tools.developer import execute_code
from prompts.debug_data_generator import build_system_prompt, build_user_prompt


logger = logging.getLogger(__name__)

_CONFIG = get_config()
_LLM_CFG = _CONFIG.get("llm")
_PATH_CFG = _CONFIG.get("paths")

_DEBUG_GENERATOR_MODEL = _LLM_CFG.get("developer_tool_model")  # Use same as developer
_TASK_ROOT = Path(_PATH_CFG.get("task_root"))
_OUTPUTS_DIRNAME = _PATH_CFG.get("outputs_dirname")

# Constants
TARGET_DEBUG_TRAIN_SIZE = 1000
TARGET_DEBUG_TEST_SIZE = 200
MIN_DATASET_SIZE_RATIO = 0.5  # Skip debug if debug would be >50% of full data


def generate_debug_data(slug: str, iteration: int) -> Dict[str, Any]:
    """
    Generate debug datasets for rapid development iteration.

    Args:
        slug: Competition slug
        iteration: Current iteration number

    Returns:
        Dictionary with debug data information:
        {
            "skip_debug": bool,
            "reason": str (if skipped),
            "sampling_code_path": str,
            "metadata": {...},
            "paths": {...}
        }
    """
    load_dotenv()

    base_dir = _TASK_ROOT / slug
    outputs_dir = base_dir / _OUTPUTS_DIRNAME / str(iteration)

    logger.info(f"Generating debug data for {slug} iteration {iteration}")

    # Step 1: Check cache (competition-level, not iteration-level)
    cache_path = base_dir / "debug_data_info.json"
    if cache_path.exists():
        logger.info("Using cached debug data")
        with open(cache_path, "r") as f:
            return json.load(f)

    # Step 2: Load inputs
    inputs = _load_inputs(base_dir, outputs_dir)

    # Step 3: Analyze data sizes
    train_size = _get_train_size(base_dir)
    test_size = _get_test_size(base_dir)
    num_classes = _get_num_classes(base_dir, inputs)

    logger.info(f"Data sizes: train={train_size}, test={test_size}, classes={num_classes}")

    # Step 4: Check if debug should be skipped
    skip, reason = _should_skip_debug(train_size, test_size, num_classes)
    if skip:
        logger.info(f"Skipping debug mode: {reason}")
        result = {"skip_debug": True, "reason": reason, "paths": {}, "metadata": {}}
        _save_cache(cache_path, result)
        return result

    # Step 5: Generate sampling code via LLM
    directory_listing = _get_directory_listing(base_dir)
    research_summary = inputs.get("research_plan_summary")

    sampling_code = _generate_sampling_code(
        slug, inputs, train_size, test_size, directory_listing, research_summary
    )

    # Step 6: Save sampling code for review
    code_path = outputs_dir / "debug_sampling_code.py"
    with open(code_path, "w") as f:
        f.write(sampling_code)
    logger.info(f"Saved sampling code to {code_path}")

    # Step 7: Execute sampling code
    try:
        _execute_sampling_code(sampling_code, base_dir)
    except Exception as e:
        logger.warning(f"Sampling code failed on first attempt: {e}")
        # Try to fix with web search (error message already includes web search from execute_code)
        try:
            fixed_code = _handle_sampling_error(str(e), sampling_code, slug, inputs, directory_listing)
            # Save fixed code
            with open(code_path, "w") as f:
                f.write(f"# ORIGINAL CODE FAILED - FIXED VERSION\n\n{fixed_code}")
            _execute_sampling_code(fixed_code, base_dir)
        except Exception as e2:
            logger.error(f"Sampling code failed after fix attempt: {e2}")
            result = {
                "skip_debug": True,
                "reason": f"Failed to generate debug data: {str(e2)}",
                "paths": {},
                "metadata": {}
            }
            _save_cache(cache_path, result)
            return result

    # Step 8: Validate debug samples
    validation_result = _validate_debug_samples(base_dir, train_size, num_classes)

    if not validation_result["valid"]:
        logger.error(f"Debug validation failed: {validation_result['errors']}")
        result = {
            "skip_debug": True,
            "reason": f"Validation failed: {'; '.join(validation_result['errors'])}",
            "paths": {},
            "metadata": {}
        }
        _save_cache(cache_path, result)
        return result

    # Step 9: Create result
    result = {
        "skip_debug": False,
        "sampling_code_path": str(code_path),
        "metadata": validation_result["metadata"],
        "paths": validation_result["paths"]
    }

    logger.info(f"Debug data created successfully: {result['metadata']}")

    # Step 10: Cache result
    _save_cache(cache_path, result)

    return result


def _load_inputs(base_dir: Path, outputs_dir: Path) -> Dict[str, Any]:
    """Load starter and researcher inputs."""
    inputs = {}

    # Load starter suggestions
    starter_path = outputs_dir / "starter_suggestions.json"
    if starter_path.exists():
        with open(starter_path, "r") as f:
            starter = json.load(f)
            inputs["task_type"] = starter.get("task_type", "tabular")
            inputs["task_summary"] = starter.get("task_summary", "")

    # Load research plan (optional)
    plan_path = outputs_dir / "plan.md"
    if plan_path.exists():
        with open(plan_path, "r") as f:
            plan = f.read()
            # Extract summary (first few hundred chars)
            inputs["research_plan_summary"] = plan[:500] + "..." if len(plan) > 500 else plan

    return inputs


def _get_train_size(base_dir: Path) -> int:
    """Detect training set size from various formats."""

    # Check for train.csv
    if (base_dir / "train.csv").exists():
        try:
            return len(pd.read_csv(base_dir / "train.csv"))
        except Exception as e:
            logger.warning(f"Failed to read train.csv: {e}")

    # Check for train/ directory (images, audio)
    if (base_dir / "train").is_dir():
        try:
            files = list((base_dir / "train").rglob("*.*"))
            return len([f for f in files if f.is_file()])
        except Exception as e:
            logger.warning(f"Failed to count train/ files: {e}")

    # Check for train_features.csv (multi-file)
    if (base_dir / "train_features.csv").exists():
        try:
            return len(pd.read_csv(base_dir / "train_features.csv"))
        except Exception as e:
            logger.warning(f"Failed to read train_features.csv: {e}")

    raise RuntimeError(f"Cannot determine train size for {base_dir}")


def _get_test_size(base_dir: Path) -> int:
    """Detect test set size."""

    # Check for test.csv
    if (base_dir / "test.csv").exists():
        try:
            return len(pd.read_csv(base_dir / "test.csv"))
        except Exception:
            pass

    # Check for test/ directory
    if (base_dir / "test").is_dir():
        try:
            files = list((base_dir / "test").rglob("*.*"))
            return len([f for f in files if f.is_file()])
        except Exception:
            pass

    # Check sample_submission.csv as proxy
    if (base_dir / "sample_submission.csv").exists():
        try:
            return len(pd.read_csv(base_dir / "sample_submission.csv")) - 1  # -1 for header
        except Exception:
            pass

    logger.warning(f"Cannot determine test size for {base_dir}, defaulting to 0")
    return 0


def _get_num_classes(base_dir: Path, inputs: Dict[str, Any]) -> Optional[int]:
    """Detect number of classes for classification tasks."""
    task_type = inputs.get("task_type", "")

    if "classification" not in task_type.lower():
        return None

    # Try to read from train.csv
    if (base_dir / "train.csv").exists():
        try:
            df = pd.read_csv(base_dir / "train.csv")
            # Try common target column names
            for col in ["target", "label", "class", "Id", df.columns[-1]]:
                if col in df.columns:
                    return df[col].nunique()
        except Exception:
            pass

    return None


def _should_skip_debug(train_size: int, test_size: int, num_classes: Optional[int]) -> Tuple[bool, Optional[str]]:
    """Determine if debug mode should be skipped."""

    # Rule 1: Train too small (>50% would be debug)
    if train_size < TARGET_DEBUG_TRAIN_SIZE * 2:
        ratio = TARGET_DEBUG_TRAIN_SIZE / train_size
        return True, f"Train set too small ({train_size} samples, debug would be {ratio:.1%})"

    # Rule 2: Too many classes relative to debug size
    if num_classes and num_classes / TARGET_DEBUG_TRAIN_SIZE > 0.8:
        return True, f"Too many classes ({num_classes}) for debug sample ({TARGET_DEBUG_TRAIN_SIZE}) to be representative"

    return False, None


def _get_directory_listing(base_dir: Path) -> str:
    """Get formatted directory listing of competition files."""
    lines = []
    lines.append(f"Competition directory: {base_dir.name}/")

    # List immediate files and directories
    for item in sorted(base_dir.iterdir()):
        if item.name == "outputs":
            continue  # Skip outputs directory

        if item.is_file():
            size = item.stat().st_size
            if size < 1024:
                size_str = f"{size}B"
            elif size < 1024 * 1024:
                size_str = f"{size/1024:.1f}KB"
            else:
                size_str = f"{size/(1024*1024):.1f}MB"

            lines.append(f"  - {item.name} ({size_str})")
        elif item.is_dir():
            num_files = len(list(item.rglob("*.*")))
            lines.append(f"  - {item.name}/ ({num_files} files)")

    return "\n".join(lines)


def _generate_sampling_code(
    slug: str,
    inputs: Dict[str, Any],
    train_size: int,
    test_size: int,
    directory_listing: str,
    research_summary: Optional[str]
) -> str:
    """Generate sampling code via LLM."""

    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(
        slug=slug,
        task_type=inputs.get("task_type", "tabular"),
        task_summary=inputs.get("task_summary", ""),
        train_size=train_size,
        test_size=test_size,
        directory_listing=directory_listing,
        research_summary=research_summary
    )

    logger.info("Calling LLM to generate sampling code...")

    response = call_llm_with_retry(
        model=_DEBUG_GENERATOR_MODEL,
        instructions=system_prompt,
        tools=[],
        messages=[{"role": "user", "content": user_prompt}],
        web_search_enabled=False
    )

    content = response.output_text or ""

    # Extract code block
    code = _extract_code_block(content)

    if not code:
        raise RuntimeError("LLM did not return valid Python code")

    return code


def _extract_code_block(text: str) -> Optional[str]:
    """Extract Python code from markdown fence."""
    # Try fenced code block
    pattern = r"```python\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Try generic code fence
    pattern = r"```\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return None


def _execute_sampling_code(code: str, base_dir: Path) -> None:
    """Execute the sampling code."""
    logger.info("Executing sampling code...")

    # Write code to temp file
    temp_script = base_dir / "temp_sampling.py"

    with open(temp_script, "w") as f:
        f.write(code)

    try:
        # Execute using tools.developer.execute_code
        result = execute_code(str(temp_script))

        # Check if execution failed (execute_code returns stderr+web_search on failure)
        if "Traceback (most recent call last)" in result or "ERROR:" in result:
            logger.error(f"Sampling code failed:\n{result}")
            raise RuntimeError(f"Sampling code failed: {result}")

        # Success
        logger.info(f"Sampling code output:\n{result}")

    finally:
        # Clean up temp file
        if temp_script.exists():
            temp_script.unlink()


def _handle_sampling_error(
    error_message: str,
    sampling_code: str,
    slug: str,
    inputs: Dict[str, Any],
    directory_listing: str
) -> str:
    """Handle sampling code execution error by using web search results to fix code.

    Note: error_message already includes web search results from execute_code().
    """

    logger.error(f"Sampling code failed:\n{error_message}")

    # Create fix prompt with error context (error_message already includes web search)
    fix_prompt = f"""The debug sampling code failed with this error (includes web search guidance):

```
{error_message}
```

Original code:
```python
{sampling_code}
```

Competition info:
- Slug: {slug}
- Task type: {inputs.get('task_type')}
- Directory structure:
{directory_listing}

Please fix the code based on the error and web search guidance above. Common issues:
- File paths incorrect (check if using task/{slug}/ correctly)
- Missing columns in CSV
- File not found errors
- Type errors in sampling logic

Return the complete corrected code in a ```python fence.
"""

    logger.info("Calling LLM to fix sampling code based on error+web search results...")

    response = call_llm_with_retry(
        model=_DEBUG_GENERATOR_MODEL,
        instructions="You are debugging Python code for data sampling. Fix the error and return working code.",
        tools=[],
        messages=[{"role": "user", "content": fix_prompt}],
        web_search_enabled=False  # No need - error_message already has web search results
    )

    fixed_code = _extract_code_block(response.output_text or "")

    if not fixed_code:
        raise RuntimeError("LLM failed to provide fixed code")

    return fixed_code


def _validate_debug_samples(base_dir: Path, original_train_size: int, num_classes: Optional[int]) -> Dict[str, Any]:
    """Validate that debug samples were created correctly."""

    errors = []
    debug_dir = base_dir / "debug"
    paths = {}
    metadata = {}

    # Check debug directory exists
    if not debug_dir.exists():
        errors.append("debug/ directory not created")
        return {"valid": False, "errors": errors, "metadata": {}, "paths": {}}

    # Check train_debug.csv
    train_debug_path = debug_dir / "train_debug.csv"
    if not train_debug_path.exists():
        errors.append("train_debug.csv not found")
        return {"valid": False, "errors": errors, "metadata": {}, "paths": {}}

    paths["train_debug_csv"] = str(train_debug_path)

    # Load and validate train_debug
    try:
        train_debug = pd.read_csv(train_debug_path)
    except Exception as e:
        errors.append(f"Failed to read train_debug.csv: {e}")
        return {"valid": False, "errors": errors, "metadata": {}, "paths": {}}

    # Check size
    if len(train_debug) == 0:
        errors.append("train_debug.csv is empty")
    elif len(train_debug) > original_train_size:
        errors.append(f"Debug train size ({len(train_debug)}) > original ({original_train_size})")

    # Check for NaN
    if train_debug.isnull().any().any():
        errors.append("train_debug.csv contains NaN values")

    # Check columns match original (if original train.csv exists)
    if (base_dir / "train.csv").exists():
        try:
            original_train = pd.read_csv(base_dir / "train.csv")
            if list(train_debug.columns) != list(original_train.columns):
                errors.append("train_debug.csv columns don't match original")
        except Exception:
            pass  # Skip if can't read original

    # For classification: check classes
    debug_classes = None
    if num_classes:
        # Try to find target column
        target_col = None
        for col in ["target", "label", "class", "Id", train_debug.columns[-1]]:
            if col in train_debug.columns:
                target_col = col
                break

        if target_col:
            debug_classes = train_debug[target_col].nunique()

            # Check if we have reasonable class coverage
            if debug_classes < num_classes * 0.8:  # Allow 20% missing for very rare classes
                missing = num_classes - debug_classes
                errors.append(f"Missing {missing} classes in debug data ({debug_classes}/{num_classes} present)")

    # Check for image/audio directories
    if (base_dir / "train").is_dir():
        train_debug_dir = debug_dir / "train_debug"
        if train_debug_dir.exists():
            paths["train_debug_dir"] = str(train_debug_dir)
        else:
            # Check if there's an image column that should have corresponding files
            if "Image" in train_debug.columns or "image" in train_debug.columns:
                errors.append("train_debug/ directory not found but images referenced in CSV")

    # Check test_debug
    test_debug_path = debug_dir / "test_debug.csv"
    test_debug_size = 0
    if test_debug_path.exists():
        paths["test_debug_csv"] = str(test_debug_path)
        try:
            test_debug = pd.read_csv(test_debug_path)
            test_debug_size = len(test_debug)

            if len(test_debug) == 0:
                errors.append("test_debug.csv is empty")
        except Exception as e:
            errors.append(f"Failed to read test_debug.csv: {e}")

    if (debug_dir / "test_debug").is_dir():
        paths["test_debug_dir"] = str(debug_dir / "test_debug")

    # Create metadata
    metadata = {
        "train_size": original_train_size,
        "debug_train_size": len(train_debug),
        "test_size": _get_test_size(base_dir),
        "debug_test_size": test_debug_size,
        "num_classes": num_classes,
        "debug_num_classes": debug_classes,
        "sampling_ratio": len(train_debug) / original_train_size if original_train_size > 0 else 0,
    }

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "metadata": metadata,
        "paths": paths
    }


def _save_cache(cache_path: Path, data: Dict[str, Any]) -> None:
    """Save debug data info to cache."""
    with open(cache_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Cached debug data info to {cache_path}")
