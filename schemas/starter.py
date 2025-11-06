"""Schemas for StarterAgent structured outputs."""

from pydantic import BaseModel


class StarterSuggestions(BaseModel):
    """Schema for starter agent task type classification and summary."""
    task_types: list[str]
    task_summary: str
