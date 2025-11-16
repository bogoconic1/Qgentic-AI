import json
import logging
import re
import subprocess
from typing import Optional, Tuple


logger = logging.getLogger(__name__)


def parse_grade_output(stdout: str) -> Optional[dict]:
    """Extract JSON object from mlebench stdout and parse it.

    mlebench sometimes prints extra lines around the JSON. We locate the first
    opening brace and the last closing brace to slice a minimal JSON payload.
    """
    try:
        start = stdout.index("{")
        end = stdout.rindex("}") + 1
        json_text = stdout[start:end]
        return json.loads(json_text)
    except Exception:
        return None


def parse_validation_score(log_content: str) -> Optional[float]:
    """Extract validation score from log content.

    Looks for <final_validation_score>SCORE_VALUE</final_validation_score> pattern.
    If multiple occurrences exist, returns the LAST one.

    Args:
        log_content: The execution log content as string

    Returns:
        The validation score as float, or None if not found or invalid
    """
    try:
        # Match <final_validation_score>SCORE_VALUE</final_validation_score>
        pattern = r'<final_validation_score>\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*</final_validation_score>'
        matches = re.findall(pattern, log_content)
        if matches:
            # Get the last occurrence
            score = float(matches[-1])
            logger.info("Extracted validation score from logs: %s (found %d occurrence(s), using last)", score, len(matches))
            return score
        else:
            logger.warning("No <final_validation_score> tag found in logs")
            return None
    except (ValueError, IndexError) as exc:
        logger.warning("Failed to parse validation score: %s", exc)
        return None


def run_grade(file_path: str, slug: str) -> Tuple[Optional[dict], str, int, str]:
    """Run mlebench grade-sample on the given file and return (info, stdout, returncode, stderr)."""
    cmd = [
        "mlebench",
        "grade-sample",
        str(file_path),
        slug,
    ]
    logger.info("Running grading command: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        logger.exception("Failed to execute grading command")
        return None, "", 1, "execution failure"

    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    info = parse_grade_output(stdout)
    return info, stdout, result.returncode, stderr


