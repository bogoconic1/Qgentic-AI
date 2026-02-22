"""Schemas for ModelRecommenderAgent structured outputs."""

from pydantic import BaseModel


# Model Selection Schemas


class RecommendedModel(BaseModel):
    """Individual model recommendation."""

    reasoning: str
    name: str
    


class ModelSelection(BaseModel):
    """Schema for model selection response."""

    recommended_models: list[RecommendedModel]


class RefinedModel(BaseModel):
    """Individual refined model selection with detailed reasoning."""

    reasoning: str
    name: str
    selected_from_family: str


class RefinedModelSelection(BaseModel):
    """Schema for refined model selection (Stage 3) - 8 models selected from 16 candidates."""

    final_models: list[RefinedModel]


# Preprocessing Schemas


class StrategyItem(BaseModel):
    """Individual strategy with explanation."""

    reasoning: str
    strategy: str


class CategoryRecommendations(BaseModel):
    """MUST_HAVE and NICE_TO_HAVE lists for a category."""

    MUST_HAVE: list[StrategyItem]
    NICE_TO_HAVE: list[StrategyItem]


class PreprocessingRecommendations(BaseModel):
    """Schema for preprocessing recommendations. Keys are category names."""

    categories: dict[str, CategoryRecommendations]


# Loss Function Schemas


class LossFunctionMustHave(BaseModel):
    """MUST_HAVE loss function (single)."""

    reasoning: str
    loss_function: str


class LossFunctionNiceToHave(BaseModel):
    """NICE_TO_HAVE loss function item."""

    reasoning: str
    loss_function: str


class LossFunctionRecommendations(BaseModel):
    """Schema for loss function recommendations."""

    MUST_HAVE: LossFunctionMustHave
    NICE_TO_HAVE: list[LossFunctionNiceToHave]


# Hyperparameter Schemas


class HyperparameterItem(BaseModel):
    """Individual hyperparameter recommendation."""

    reasoning: str
    hyperparameter: str


class ArchitectureItem(BaseModel):
    """Individual architecture recommendation."""

    reasoning: str
    architecture: str


class HyperparameterSection(BaseModel):
    """Hyperparameters and architectures for MUST_HAVE or NICE_TO_HAVE."""

    hyperparameters: list[HyperparameterItem]
    architectures: list[ArchitectureItem]


class HyperparameterRecommendations(BaseModel):
    """Schema for hyperparameter tuning recommendations."""

    MUST_HAVE: HyperparameterSection
    NICE_TO_HAVE: HyperparameterSection


# Inference Strategy Schemas


class InferenceStrategyItem(BaseModel):
    """Individual inference strategy."""

    reasoning: str
    strategy: str


class InferenceStrategySection(BaseModel):
    """List of inference strategies."""

    inference_strategies: list[InferenceStrategyItem]


class InferenceStrategyRecommendations(BaseModel):
    """Schema for inference strategy recommendations."""

    MUST_HAVE: InferenceStrategySection
    NICE_TO_HAVE: InferenceStrategySection
