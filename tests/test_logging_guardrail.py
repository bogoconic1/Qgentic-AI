"""Unit tests for the third-party-import check added to the logging guardrail."""

from guardrails.developer import check_logging_basicconfig_order


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
