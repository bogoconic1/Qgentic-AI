"""Pydantic schemas for structured outputs."""

from schemas.starter import StarterSuggestions
from schemas.researcher import DatasetDiscovery
from schemas.model_recommender import (
    ModelSelection,
    PreprocessingRecommendations,
    LossFunctionRecommendations,
    HyperparameterRecommendations,
    InferenceStrategyRecommendations,
)
from schemas.developer import (
    StackTraceSolution,
    SOTAResponse,
)

__all__ = [
    "StarterSuggestions",
    "DatasetDiscovery",
    "ModelSelection",
    "PreprocessingRecommendations",
    "LossFunctionRecommendations",
    "HyperparameterRecommendations",
    "InferenceStrategyRecommendations",
    "StackTraceSolution",
    "SOTAResponse",
]
