"""
Tests for shared_constraints module to ensure refactoring maintains backward compatibility.
"""

from __future__ import annotations

import pytest

from prompts.developer_agent import _get_hard_constraints as developer_get_hard_constraints
from prompts.ensembler_agent import _get_hard_constraints as ensembler_get_hard_constraints
from prompts.shared_constraints import get_hard_constraints


class TestSharedConstraints:
    """Test suite for shared constraints functionality."""

    def test_developer_constraints_basic(self):
        """Test that developer constraints are generated correctly with basic params."""
        result = developer_get_hard_constraints(model_name="ResNet50", allow_multi_fold=False)

        # Check for model-specific constraints
        assert "Use ONLY `ResNet50`" in result
        assert "do not swap out `ResNet50`" in result

        # Check for fold constraint (should be present when allow_multi_fold=False)
        assert "Just train and validate on fold 0" in result

        # Check for common constraints
        assert "Deliver a fully-contained, single-file script" in result
        assert "Use CUDA if available" in result
        assert "DO NOT** explicitly set `os.environ['CUDA_VISIBLE_DEVICES']`" in result
        assert "logging.basicConfig()" in result
        assert "kagglehub" in result

        # Check for developer-specific typo (backward compatibility)
        assert "DEBUG MODE.s" in result

        # Check that ensemble directive is NOT present
        assert "CRITICAL: YOU MUST COPY EVERYTHING" not in result

    def test_developer_constraints_multi_fold(self):
        """Test that developer constraints respect allow_multi_fold parameter."""
        result = developer_get_hard_constraints(model_name="BERT", allow_multi_fold=True)

        # Fold constraint should NOT be present when allow_multi_fold=True
        assert "Just train and validate on fold 0" not in result

        # Model-specific constraints should still be present
        assert "Use ONLY `BERT`" in result

    def test_ensembler_constraints_basic(self):
        """Test that ensembler constraints are generated correctly."""
        result = ensembler_get_hard_constraints()

        # Check that model-specific constraints are NOT present (no model_name)
        assert "Use ONLY `" not in result
        assert "do not swap out `" not in result

        # Check that fold constraint is NOT present (ensembler doesn't use multi-fold)
        assert "Just train and validate on fold 0" not in result

        # Check for common constraints
        assert "Deliver a fully-contained, single-file script" in result
        assert "Use CUDA if available" in result
        assert "DO NOT** explicitly set `os.environ['CUDA_VISIBLE_DEVICES']`" in result

        # Check for ensemble-specific additions
        assert "ensemble weights" in result  # Should be in logging statement
        assert "best epoch/iteration number" in result  # Should use "/iteration" suffix

        # Check for ensemble-specific directive
        assert "CRITICAL: YOU MUST COPY EVERYTHING" in result
        assert "For each baseline model in your ensemble" in result
        assert "Preprocessing" in result
        assert "Feature Engineering and Transformations" in result

        # Check that developer-specific typo is NOT present (no "DEBUG MODE.s")
        assert "DEBUG MODE.s" not in result
        assert "DEBUG MODE." in result  # But should have "DEBUG MODE." without the 's'

    def test_shared_function_with_all_parameters(self):
        """Test the shared function directly with all parameter combinations."""
        # Test developer-like configuration
        dev_result = get_hard_constraints(
            model_name="XGBoost",
            allow_multi_fold=False,
            include_ensemble_copy_directive=False,
        )
        assert "Use ONLY `XGBoost`" in dev_result
        assert "Just train and validate on fold 0" in dev_result
        assert "CRITICAL: YOU MUST COPY EVERYTHING" not in dev_result

        # Test ensembler-like configuration
        ens_result = get_hard_constraints(
            model_name=None,
            allow_multi_fold=False,
            include_ensemble_copy_directive=True,
        )
        assert "Use ONLY `" not in ens_result
        assert "CRITICAL: YOU MUST COPY EVERYTHING" in ens_result
        assert "ensemble weights" in ens_result

    def test_common_constraints_in_both(self):
        """Test that common constraints appear in both developer and ensembler outputs."""
        dev_result = developer_get_hard_constraints(model_name="RandomForest")
        ens_result = ensembler_get_hard_constraints()

        # List of common constraints that should appear in both
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
            "DEBUG flag",
            "NaN, STOP training",
            "while` loops",
            "DEBUG MODE",
            "kagglehub",
            "dataset_download",
            "DEBUG mode guidelines",
            "sample train to 1000 rows",
            "reduce epochs to 1",
        ]

        for item in common_items:
            assert item in dev_result, f"Missing in developer: {item}"
            assert item in ens_result, f"Missing in ensembler: {item}"

    def test_backward_compatibility_developer(self):
        """Test that the developer wrapper maintains exact backward compatibility."""
        # The wrapper should produce identical output to what the old function would have
        result = developer_get_hard_constraints("TestModel", allow_multi_fold=False)

        # Check all the key elements that the original function had
        assert result.startswith("**Hard Constraints:**")
        assert "Use ONLY `TestModel`" in result
        assert "Just train and validate on fold 0" in result
        assert "DEBUG MODE.s" in result  # The typo should be preserved
        assert "Modular pipeline" in result

    def test_backward_compatibility_ensembler(self):
        """Test that the ensembler wrapper maintains exact backward compatibility."""
        result = ensembler_get_hard_constraints()

        # Check all the key elements that the original function had
        assert result.startswith("**Hard Constraints:**")
        assert "CRITICAL: YOU MUST COPY EVERYTHING" in result
        assert "ensemble weights" in result
        assert "best epoch/iteration number" in result
        assert "DEBUG MODE.s" not in result  # Ensembler didn't have the typo

    def test_different_model_names(self):
        """Test that different model names are correctly substituted."""
        models = ["ResNet18", "BERT-base", "XGBoost", "LightGBM"]

        for model in models:
            result = developer_get_hard_constraints(model_name=model)
            assert f"Use ONLY `{model}`" in result
            assert f"do not swap out `{model}`" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
