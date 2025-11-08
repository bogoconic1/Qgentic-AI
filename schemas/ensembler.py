from pydantic import BaseModel


class EnsembleStrategy(BaseModel):
    """Schema for a single ensemble strategy."""
    strategy: str
    models_needed: list[str] = []
    ensemble_method: str
    implementation_guidance: str


class EnsembleStrategies(BaseModel):
    """Schema for ensemble strategy generation."""
    checklist: list[str]
    strategies: list[EnsembleStrategy]
    validation_summary: str
