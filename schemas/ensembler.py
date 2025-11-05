from pydantic import BaseModel


class EnsembleStrategy(BaseModel):
    """Schema for a single ensemble strategy."""
    strategy: str
    models_needed: list[str] = []


class EnsembleStrategies(BaseModel):
    """Schema for ensemble strategy generation."""
    strategies: list[EnsembleStrategy]
