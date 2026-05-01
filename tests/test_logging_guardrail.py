"""Unit tests for the third-party-import check added to the logging guardrail."""

from guardrails.developer import (
    check_logging_basicconfig_order,
    check_solution_txt_filehandler,
)


def test_basicconfig_after_import_fails():
    code = """\
import logging
import torch
logging.basicConfig(level=logging.INFO)
"""
    result = check_logging_basicconfig_order(code)
    assert result["status"] == "fail"
    assert "torch" in result["violations"][0]["reason"]


def test_basicconfig_after_from_import_fails():
    code = """\
import logging
from transformers import AutoModel
logging.basicConfig(level=logging.INFO)
"""
    result = check_logging_basicconfig_order(code)
    assert result["status"] == "fail"
    assert "transformers" in result["violations"][0]["reason"]


def test_basicconfig_before_thirdparty_passes():
    code = """\
import logging
logging.basicConfig(level=logging.INFO)
import torch
"""
    assert check_logging_basicconfig_order(code)["status"] == "pass"


def test_stdlib_imports_between_logging_and_basicconfig_pass():
    code = """\
import logging
import os
import json
logging.basicConfig(level=logging.INFO)
import torch
"""
    assert check_logging_basicconfig_order(code)["status"] == "pass"


def test_no_logging_import_passes():
    code = """\
import torch
x = 1
"""
    result = check_logging_basicconfig_order(code)
    assert result["status"] == "pass"
    assert result["basicConfig_line"] is None


# ---------------------------------------------------------------------------
# check_solution_txt_filehandler — SOLUTION.txt FileHandler in basicConfig
# ---------------------------------------------------------------------------


def test_filehandler_solution_txt_passes():
    code = """\
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parent / "SOLUTION.txt", mode="w"),
    ],
)
"""
    assert check_solution_txt_filehandler(code)["status"] == "pass"


def test_filehandler_missing_handlers_kwarg_fails():
    code = """\
import logging
logging.basicConfig(level=logging.INFO)
"""
    result = check_solution_txt_filehandler(code)
    assert result["status"] == "fail"
    assert "handlers=" in result["violations"][0]["reason"]


def test_filehandler_wrong_filename_fails():
    code = """\
import logging
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.FileHandler("WRONG.txt")],
)
"""
    result = check_solution_txt_filehandler(code)
    assert result["status"] == "fail"
    assert "SOLUTION.txt" in result["violations"][0]["reason"]


def test_filehandler_no_basicconfig_fails():
    code = """\
import logging
"""
    result = check_solution_txt_filehandler(code)
    assert result["status"] == "fail"
    assert "basicConfig" in result["violations"][0]["reason"]
