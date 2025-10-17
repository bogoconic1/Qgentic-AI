import logging
import re
import subprocess
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


def extract_diff_block(content: str) -> str:
    if not content:
        return ""
    fenced = re.search(r"```diff\s*(.*?)\s*```", content, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    generic = re.search(r"```\s*(.*?)\s*```", content, re.DOTALL)
    if generic:
        return generic.group(1).strip()
    return content.strip()


def _fix_hunk_headers(base_lines: list[str], diff_lines: list[str]) -> list[str]:
    header_re = re.compile(r"^@@\s*-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@")

    delta = 0
    hunks = []
    for i in range(len(diff_lines)):
        if re.match(header_re, diff_lines[i]):
            A, B, C, D = map(int, header_re.match(diff_lines[i]).groups())
            hunks.append([i, A, B or 0, C, D or 0])
    hunks.append([len(diff_lines), 0, 0, 0, 0])

    for i in range(len(hunks) - 1):
        diff_hunk_l, old_A, old_B, old_C, old_D = hunks[i]
        diff_hunk_r, _, _, _, _ = hunks[i + 1]
        hunk_lines = diff_lines[diff_hunk_l + 1 : diff_hunk_r]
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
                minus_count += 1
                plus_count += 1
                if first_valid_line is None:
                    first_valid_line = line

        if first_valid_line is None:
            return diff_lines

        new_A = None
        for line_no in range(old_A - 15, old_A + 15):
            if line_no < 0 or line_no >= len(base_lines):
                continue
            if base_lines[line_no].strip() == first_valid_line.strip():
                new_A = line_no + 1
                break
        if new_A is None:
            return diff_lines

        new_C = new_A + delta
        new_B = minus_count
        new_D = plus_count

        delta += plus_count - minus_count
        diff_lines[diff_hunk_l] = f"@@ -{new_A},{new_B} +{new_C},{new_D} @@"

    return diff_lines


def normalize_diff_payload(base_path: Path, diff_text: str) -> Optional[str]:
    import os
    print(os.getcwd())
    print(os.listdir(os.getcwd()))
    try:
        base_lines = base_path.read_text().splitlines()
    except Exception:
        logger.exception("Failed to read base file while normalizing diff: %s", base_path)
        return None
    diff_lines = diff_text.splitlines()
    fixed_lines = _fix_hunk_headers(base_lines, diff_lines)
    return "\n".join(fixed_lines) + "\n"


def apply_patch(outputs_dir: Path, base_filename: str, output_filename: str, payload: str) -> Optional[str]:
    output_path = outputs_dir / output_filename
    if output_path.exists():
        try:
            output_path.unlink()
        except Exception:
            logger.exception(
                "Failed to remove existing output file before applying diff: %s",
                output_path,
            )
            return None

    cmd = ["patch", "-o", output_filename, base_filename]
    try:
        result = subprocess.run(
            cmd,
            input=payload,
            text=True,
            capture_output=True,
            cwd=outputs_dir,
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
            "Patch command returned non-zero exit code %s. stderr=\n%s",
            result.returncode,
            (result.stderr or "").strip(),
        )
        if output_path.exists():
            try:
                output_path.unlink()
            except Exception:
                logger.debug("Failed to clean up partial patched file: %s", output_path)
        return None

    try:
        patched_text =output_path.read_text()
        if "```diff" in patched_text[:300]:
            logger.warning("Patched file contains ```diff block. Returning None.")
            return None
        return patched_text
    except Exception:
        logger.exception("Failed to read patched file at %s", output_path)
        return None


