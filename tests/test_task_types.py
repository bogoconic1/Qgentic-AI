"""Test task type validation and preprocessing category selection."""

import pytest
from constants import (
    VALID_TASK_TYPES,
    normalize_task_type,
    get_preprocessing_categories,
    PREPROCESSING_CATEGORIES,
)


def test_valid_task_types():
    """Test that VALID_TASK_TYPES contains expected values."""
    assert "computer_vision" in VALID_TASK_TYPES
    assert "nlp" in VALID_TASK_TYPES
    assert "tabular" in VALID_TASK_TYPES
    assert "time_series" in VALID_TASK_TYPES
    assert "audio" in VALID_TASK_TYPES
    assert len(VALID_TASK_TYPES) == 5


def test_normalize_task_type_exact_match():
    """Test normalization with exact matches."""
    assert normalize_task_type("computer_vision") == "computer_vision"
    assert normalize_task_type("nlp") == "nlp"
    assert normalize_task_type("tabular") == "tabular"
    assert normalize_task_type("time_series") == "time_series"
    assert normalize_task_type("audio") == "audio"


def test_normalize_task_type_case_insensitive():
    """Test normalization is case-insensitive."""
    assert normalize_task_type("Computer Vision") == "computer_vision"
    assert normalize_task_type("NLP") == "nlp"
    assert normalize_task_type("TABULAR") == "tabular"
    assert normalize_task_type("Time Series") == "time_series"
    assert normalize_task_type("AUDIO") == "audio"


def test_normalize_task_type_variations():
    """Test normalization with common variations."""
    # NLP variations
    assert normalize_task_type("Natural Language Processing") == "nlp"
    assert normalize_task_type("text classification") == "nlp"
    assert normalize_task_type("sentiment analysis") == "nlp"

    # Vision variations
    assert normalize_task_type("vision") == "computer_vision"
    assert normalize_task_type("image classification") == "computer_vision"
    assert normalize_task_type("object detection") == "computer_vision"

    # Tabular variations
    assert normalize_task_type("structured data") == "tabular"
    assert normalize_task_type("xgboost") == "tabular"

    # Time series variations
    assert normalize_task_type("forecasting") == "time_series"
    assert normalize_task_type("temporal") == "time_series"

    # Audio variations
    assert normalize_task_type("speech recognition") == "audio"
    assert normalize_task_type("sound") == "audio"


def test_normalize_task_type_empty():
    """Test normalization with empty string raises ValueError."""
    with pytest.raises(ValueError, match="task_type cannot be empty"):
        normalize_task_type("")


def test_normalize_task_type_invalid():
    """Test normalization with invalid task type returns normalized string."""
    # Should return the normalized (lowercase, underscored) string
    result = normalize_task_type("completely_invalid_task")
    assert result == "completely_invalid_task"
    # Caller should validate against VALID_TASK_TYPES


def test_get_preprocessing_categories_computer_vision():
    """Test preprocessing categories for computer vision tasks."""
    categories = get_preprocessing_categories("computer_vision")
    assert "preprocessing" in categories
    assert "data_augmentation" in categories
    # Should NOT have feature engineering categories
    assert "feature_creation" not in categories
    assert "feature_selection" not in categories
    assert "tokenization" not in categories
    assert len(categories) == 2


def test_get_preprocessing_categories_nlp():
    """Test preprocessing categories for NLP tasks."""
    categories = get_preprocessing_categories("nlp")
    assert "preprocessing" in categories
    assert "tokenization" in categories
    assert "data_augmentation" in categories
    # Should NOT have feature engineering categories
    assert "feature_creation" not in categories
    assert "feature_selection" not in categories
    assert len(categories) == 3


def test_get_preprocessing_categories_tabular():
    """Test preprocessing categories for tabular tasks."""
    categories = get_preprocessing_categories("tabular")
    assert "preprocessing" in categories
    assert "feature_creation" in categories
    assert "feature_selection" in categories
    assert "feature_transformation" in categories
    # Should NOT have tokenization or data augmentation
    assert "tokenization" not in categories
    assert "data_augmentation" not in categories
    assert len(categories) == 4


def test_get_preprocessing_categories_time_series():
    """Test preprocessing categories for time series tasks."""
    categories = get_preprocessing_categories("time_series")
    assert "preprocessing" in categories
    assert "feature_creation" in categories
    assert "feature_transformation" in categories
    assert "data_augmentation" in categories
    # Should NOT have feature selection or tokenization
    assert "feature_selection" not in categories
    assert "tokenization" not in categories
    assert len(categories) == 4


def test_get_preprocessing_categories_audio():
    """Test preprocessing categories for audio tasks."""
    categories = get_preprocessing_categories("audio")
    assert "preprocessing" in categories
    assert "data_augmentation" in categories
    # Should NOT have feature engineering or tokenization
    assert "feature_creation" not in categories
    assert "feature_selection" not in categories
    assert "feature_transformation" not in categories
    assert "tokenization" not in categories
    assert len(categories) == 2


def test_get_preprocessing_categories_fallback():
    """Test preprocessing categories for unknown task type returns all categories."""
    categories = get_preprocessing_categories("unknown_task")
    # Should return all possible categories as fallback
    assert "preprocessing" in categories
    assert "feature_creation" in categories
    assert "feature_selection" in categories
    assert "feature_transformation" in categories
    assert "tokenization" in categories
    assert "data_augmentation" in categories


def test_preprocessing_categories_mapping():
    """Test that PREPROCESSING_CATEGORIES mapping is complete."""
    for task_type in VALID_TASK_TYPES:
        assert task_type in PREPROCESSING_CATEGORIES
        categories = PREPROCESSING_CATEGORIES[task_type]
        assert isinstance(categories, list)
        assert len(categories) > 0
        # All should have preprocessing
        assert "preprocessing" in categories


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
