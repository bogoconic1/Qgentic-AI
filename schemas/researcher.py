"""Schemas for ResearcherAgent structured outputs."""

from pydantic import BaseModel


class DatasetDiscovery(BaseModel):
    """Schema for dataset discovery responses."""
    datasets: list[str]
