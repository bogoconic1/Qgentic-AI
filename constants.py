"""Constants and configuration mappings for Qgentic-AI."""

# Valid task types for competitions
VALID_TASK_TYPES = [
    "computer_vision",  # Image classification, object detection, segmentation, etc.
    "nlp",             # Text classification, NER, QA, summarization, etc.
    "tabular",         # Structured/CSV data with traditional ML or tabular NNs
    "time_series",     # Temporal forecasting, anomaly detection
    "audio"            # Speech, sound classification, audio generation
]

# Preprocessing categories by task type
PREPROCESSING_CATEGORIES = {
    "computer_vision": ["preprocessing", "data_augmentation"],
    "nlp": ["preprocessing", "tokenization", "data_augmentation"],
    "tabular": ["preprocessing", "feature_creation", "feature_selection", "feature_transformation"],
    "time_series": ["preprocessing", "feature_creation", "feature_transformation", "data_augmentation"],
    "audio": ["preprocessing", "data_augmentation"]
}

# Task type normalization mappings
TASK_TYPE_MAPPINGS = {
    # NLP variations
    "natural_language_processing": "nlp",
    "text": "nlp",
    "language": "nlp",
    "text_classification": "nlp",
    "text_regression": "nlp",
    "sentiment": "nlp",
    "ner": "nlp",
    "named_entity_recognition": "nlp",
    "question_answering": "nlp",
    "summarization": "nlp",
    "translation": "nlp",
    "bert": "nlp",
    "gpt": "nlp",
    "transformer": "nlp",
    "llm": "nlp",

    # Computer Vision variations
    "computer_vision": "computer_vision",
    "vision": "computer_vision",
    "image": "computer_vision",
    "cv": "computer_vision",
    "image_classification": "computer_vision",
    "object_detection": "computer_vision",
    "segmentation": "computer_vision",
    "cnn": "computer_vision",
    "resnet": "computer_vision",
    "vit": "computer_vision",
    "efficientnet": "computer_vision",

    # Tabular variations
    "tabular": "tabular",
    "structured": "tabular",
    "csv": "tabular",
    "xgboost": "tabular",
    "catboost": "tabular",
    "lightgbm": "tabular",
    "random_forest": "tabular",
    "gradient_boosting": "tabular",

    # Time Series variations
    "time_series": "time_series",
    "timeseries": "time_series",
    "temporal": "time_series",
    "forecasting": "time_series",
    "sequential": "time_series",

    # Audio variations
    "audio": "audio",
    "speech": "audio",
    "sound": "audio",
    "acoustic": "audio",
    "speech_recognition": "audio",
}


def normalize_task_type(task_type: str) -> str:
    """Normalize task_type to canonical form.

    Args:
        task_type: Raw task type string (e.g., "Computer Vision", "text classification")

    Returns:
        Canonical task type (one of VALID_TASK_TYPES)

    Raises:
        ValueError: If task_type cannot be normalized to a valid type
    """
    if not task_type:
        raise ValueError("task_type cannot be empty")

    # Normalize to lowercase with underscores
    task_lower = task_type.lower().replace(" ", "_").replace("-", "_")

    # Direct match
    if task_lower in VALID_TASK_TYPES:
        return task_lower

    # Check mappings for partial matches
    for key, canonical in TASK_TYPE_MAPPINGS.items():
        if key in task_lower or task_lower in key:
            return canonical

    # No match found - return the normalized string for error reporting
    # The caller should validate against VALID_TASK_TYPES
    return task_lower


def get_preprocessing_categories(task_type: str) -> list[str]:
    """Get relevant preprocessing categories for a task type.

    Args:
        task_type: Canonical task type (one of VALID_TASK_TYPES)

    Returns:
        List of preprocessing category names
    """
    # Normalize first
    normalized = normalize_task_type(task_type)

    # Return categories or fallback to all categories
    return PREPROCESSING_CATEGORIES.get(
        normalized,
        ["preprocessing", "feature_creation", "feature_selection",
         "feature_transformation", "tokenization", "data_augmentation"]
    )
