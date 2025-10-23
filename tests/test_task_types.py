"""Test task type validation and preprocessing category selection."""

import pytest
from constants import (
    VALID_TASK_TYPES,
    ALL_PREPROCESSING_CATEGORIES,
    normalize_task_type,
    select_preprocessing_categories_dynamically,
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


def test_all_preprocessing_categories_complete():
    """Test that ALL_PREPROCESSING_CATEGORIES contains all expected categories."""
    expected = [
        "preprocessing",
        "feature_creation",
        "feature_selection",
        "feature_transformation",
        "tokenization",
        "data_augmentation",
    ]
    assert len(ALL_PREPROCESSING_CATEGORIES) == len(expected)
    for category in expected:
        assert category in ALL_PREPROCESSING_CATEGORIES


def test_dynamic_category_selection_returns_valid_categories():
    """Test that dynamic selection returns only valid categories."""
    # Simple NLP competition description
    description = """
    This is a text classification competition where you need to classify
    customer reviews into positive or negative sentiment.
    """
    categories = select_preprocessing_categories_dynamically(
        task_type="nlp",
        competition_description=description,
    )

    assert isinstance(categories, list)
    assert len(categories) > 0
    # All returned categories should be valid
    for category in categories:
        assert category in ALL_PREPROCESSING_CATEGORIES


def test_dynamic_category_selection_with_research_plan():
    """Test that dynamic selection considers research plan."""
    description = """
    Patent phrase matching competition. Match semantic similarity of phrases.
    """
    research_plan = """
    The dataset contains:
    - anchor and target text phrases
    - CPC codes (categorical metadata)
    - Multiple anchors appear many times with different targets
    """

    categories = select_preprocessing_categories_dynamically(
        task_type="nlp",
        competition_description=description,
        research_plan=research_plan,
    )

    assert isinstance(categories, list)
    assert len(categories) > 0
    # Should recognize potential for feature engineering from metadata
    for category in categories:
        assert category in ALL_PREPROCESSING_CATEGORIES


def test_dynamic_category_selection_fallback_on_error():
    """Test that dynamic selection falls back to all categories on error."""
    # Pass invalid/empty description to trigger fallback
    categories = select_preprocessing_categories_dynamically(
        task_type="nlp",
        competition_description="",  # Empty description
    )

    # Should fallback to all categories
    assert isinstance(categories, list)
    assert len(categories) > 0
    for category in categories:
        assert category in ALL_PREPROCESSING_CATEGORIES


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
