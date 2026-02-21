import json
import logging
import subprocess


logger = logging.getLogger(__name__)


def parse_grade_output(stdout: str) -> dict:
    """Extract JSON object from mlebench stdout and parse it.

    mlebench sometimes prints extra lines around the JSON. We locate the first
    opening brace and the last closing brace to slice a minimal JSON payload.
    """
    start = stdout.index("{")
    end = stdout.rindex("}") + 1
    return json.loads(stdout[start:end])


def run_grade(file_path: str, slug: str) -> tuple[dict, str, int, str]:
    """Run mlebench grade-sample on the given file and return (info, stdout, returncode, stderr)."""
    cmd = [
        "mlebench",
        "grade-sample",
        str(file_path),
        slug,
    ]
    logger.info("Running grading command: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    info = parse_grade_output(stdout)
    return info, stdout, result.returncode, stderr
