from pydantic import BaseModel


class FeatureSpec(BaseModel):
    """Schema for a single feature specification."""
    feature_name: str
    code: str
    rationale: str
    category: str


class EvolutionaryFeatures(BaseModel):
    """Schema for evolutionary feature engineering output."""
    features: list[FeatureSpec]
