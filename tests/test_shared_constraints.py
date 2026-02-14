"""
Tests for shared_constraints module.
"""

from __future__ import annotations

import pytest

from prompts.developer_agent import _get_hard_constraints as developer_get_hard_constraints
from prompts.shared_constraints import get_hard_constraints


class TestSharedConstraints:
    """Test suite for shared constraints functionality."""

    def test_developer_constraints_basic(self):
        """Test that developer constraints are generated correctly with basic params."""
        result = developer_get_hard_constraints(model_name="ResNet50")

        # Check for model-specific constraints
        assert "Use ONLY `ResNet50`" in result
        assert "do not swap out `ResNet50`" in result

        # Check for common constraints
        assert "Deliver a fully-contained, single-file script" in result
        assert "Use CUDA if available" in result
        assert "DO NOT** explicitly set `os.environ['CUDA_VISIBLE_DEVICES']`" in result
        assert "logging.basicConfig()" in result
        assert "kagglehub" in result

        # Check for developer-specific constraint ending (backward compatibility)
        assert "while` loops" in result

    def test_shared_function_with_all_parameters(self):
        """Test the shared function directly with all parameter combinations."""
        dev_result = get_hard_constraints(model_name="XGBoost")
        assert "Use ONLY `XGBoost`" in dev_result

    def test_common_constraints_present(self):
        """Test that common constraints appear in developer output."""
        dev_result = developer_get_hard_constraints(model_name="RandomForest")

        common_items = [
            "Deliver a fully-contained, single-file script",
            "Use CUDA if available",
            "CUDA_VISIBLE_DEVICES",
            "logging.basicConfig()",
            "gradient checkpointing",
            "XGBoost, LightGBM, or CatBoost",
            "Optuna for up to 300 seconds",
            "eval_strategy instead of evaluation_strategy",
            "try/except",
            "Prefer pretrained models",
            "External datasets",
            "NaN, STOP training",
            "while` loops",
            "kagglehub",
            "dataset_download",
        ]

        for item in common_items:
            assert item in dev_result, f"Missing in developer: {item}"

    def test_backward_compatibility_developer(self):
        """Test that the developer wrapper produces expected output."""
        result = developer_get_hard_constraints("TestModel")

        assert result.startswith("**Hard Constraints:**")
        assert "Use ONLY `TestModel`" in result
        assert "while` loops" in result
        assert "Modular pipeline" in result

    def test_different_model_names(self):
        """Test that different model names are correctly substituted."""
        models = ["ResNet18", "BERT-base", "XGBoost", "LightGBM"]

        for model in models:
            result = developer_get_hard_constraints(model_name=model)
            assert f"Use ONLY `{model}`" in result
            assert f"do not swap out `{model}`" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
