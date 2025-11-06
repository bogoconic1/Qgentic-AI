"""Schemas for ModelRecommenderAgent structured outputs."""

from pydantic import BaseModel


# Model Selection Schemas

class FoldSplitStrategy(BaseModel):
    """Fold split strategy recommendation."""
    strategy: str


class RecommendedModel(BaseModel):
    """Individual model recommendation."""
    name: str
    reason: str


class ModelSelection(BaseModel):
    """Schema for model selection response."""
    fold_split_strategy: FoldSplitStrategy
    recommended_models: list[RecommendedModel]


# Preprocessing Schemas

class StrategyItem(BaseModel):
    """Individual strategy with explanation."""
    strategy: str
    explanation: str


class CategoryRecommendations(BaseModel):
    """MUST_HAVE and NICE_TO_HAVE lists for a category."""
    MUST_HAVE: list[StrategyItem]
    NICE_TO_HAVE: list[StrategyItem]


# THIS IS BUGGY AND NOT USED
class PreprocessingRecommendations(BaseModel):
    """Schema for preprocessing recommendations - flexible categories."""
    # Allow any category name with MUST_HAVE/NICE_TO_HAVE structure
    class Config:
        extra = "allow"  # Allow additional fields


# Loss Function Schemas

class LossFunctionMustHave(BaseModel):
    """MUST_HAVE loss function (single)."""
    loss_function: str
    explanation: str


class LossFunctionNiceToHave(BaseModel):
    """NICE_TO_HAVE loss function item."""
    loss_function: str
    explanation: str


class LossFunctionRecommendations(BaseModel):
    """Schema for loss function recommendations."""
    MUST_HAVE: LossFunctionMustHave
    NICE_TO_HAVE: list[LossFunctionNiceToHave]


# Hyperparameter Schemas

class HyperparameterItem(BaseModel):
    """Individual hyperparameter recommendation."""
    hyperparameter: str
    explanation: str


class ArchitectureItem(BaseModel):
    """Individual architecture recommendation."""
    architecture: str
    explanation: str


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
    strategy: str
    explanation: str


class InferenceStrategySection(BaseModel):
    """List of inference strategies."""
    inference_strategies: list[InferenceStrategyItem]


class InferenceStrategyRecommendations(BaseModel):
    """Schema for inference strategy recommendations."""
    MUST_HAVE: InferenceStrategySection
    NICE_TO_HAVE: InferenceStrategySection
