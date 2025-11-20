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


