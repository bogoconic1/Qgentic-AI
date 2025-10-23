"""Constants and configuration mappings for Qgentic-AI."""

# Valid task types for competitions
VALID_TASK_TYPES = [
    "computer_vision",  # Image classification, object detection, segmentation, etc.
    "nlp",             # Text classification, NER, QA, summarization, etc.
    "tabular",         # Structured/CSV data with traditional ML or tabular NNs
    "time_series",     # Temporal forecasting, anomaly detection
    "audio"            # Speech, sound classification, audio generation
]

# All available preprocessing categories across all task types
ALL_PREPROCESSING_CATEGORIES = [
    "preprocessing",          # General data cleaning, handling missing values, normalization
    "feature_creation",       # Creating new features from existing ones
    "feature_selection",      # Selecting/pruning features
    "feature_transformation", # Scaling, encoding, dimensionality reduction
    "tokenization",          # Text tokenization and vocabulary handling
    "data_augmentation"      # Data augmentation techniques (works for all modalities)
]

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


def select_preprocessing_categories_dynamically(
    task_type: str,
    competition_description: str,
    research_plan: str | None = None,
    model_name: str | None = None,
) -> list[str]:
    """Dynamically select relevant preprocessing categories based on competition characteristics.

    Uses LLM to analyze the competition and determine which preprocessing categories
    are actually relevant, rather than relying on hardcoded task type mappings.

    Args:
        task_type: Canonical task type (one of VALID_TASK_TYPES)
        competition_description: Full competition description text
        research_plan: Optional research plan with EDA insights
        model_name: Optional model name for context

    Returns:
        List of relevant preprocessing category names
    """
    from tools.helpers import call_llm_with_retry

    # Build context for LLM
    context = f"""<competition_description>
{competition_description}
</competition_description>

<task_type>{task_type}</task_type>"""

    if research_plan:
        context += f"""

<research_plan>
{research_plan}
</research_plan>"""

    if model_name:
        context += f"""

<model_name>{model_name}</model_name>"""

    # Category selection prompt
    prompt = f"""{context}

# Task: Select Relevant Preprocessing Categories

Given the competition above, select which preprocessing categories are RELEVANT for this specific task.

## Available Categories:
1. **preprocessing**: General data cleaning, handling missing values, normalization, input formatting
2. **feature_creation**: Creating new features from existing data (e.g., text length, domain statistics, interaction features)
3. **feature_selection**: Selecting/pruning features to improve performance or reduce overfitting
4. **feature_transformation**: Scaling, encoding, dimensionality reduction
5. **tokenization**: Text tokenization and vocabulary handling
6. **data_augmentation**: Data augmentation techniques (works for all modalities: images, text, tabular, audio, time series)

## Instructions:
- Consider what data types are present (text, images, tabular metadata, time series, audio)
- Consider what auxiliary features or metadata exist that could be engineered
- Consider what preprocessing techniques could improve the competition metric
- DO NOT limit yourself by traditional task type boundaries - if a category is useful, include it
- For example:
  - NLP tasks often benefit from feature_creation (text statistics, metadata features)
  - Tabular tasks can benefit from data_augmentation (SMOTE, noise injection)
  - Image tasks may have metadata that requires feature_creation
  - Multi-modal tasks need multiple categories

## Output Format:
Return ONLY a JSON array of category names, nothing else.

Example: ["preprocessing", "tokenization", "feature_creation", "data_augmentation"]
"""

    try:
        response = call_llm_with_retry(
            model="gpt-5-mini",  # Use fast, cheap model for category selection
            instructions="You are an expert at analyzing machine learning competitions. Select the most relevant preprocessing categories based on the competition characteristics.",
            tools=[],
            messages=[{"role": "user", "content": prompt}],
            web_search_enabled=False,
        )

        content = response.output_text or ""

        # Extract JSON array from response
        import json
        import re

        # Try to find JSON array
        match = re.search(r'\[[\s\S]*?\]', content)
        if match:
            categories = json.loads(match.group(0))

            # Validate categories
            valid_categories = [c for c in categories if c in ALL_PREPROCESSING_CATEGORIES]

            if valid_categories:
                return valid_categories

    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("Dynamic category selection failed: %s. Returning all categories.", e)

    # Fallback: return all categories if dynamic selection fails
    return ALL_PREPROCESSING_CATEGORIES
