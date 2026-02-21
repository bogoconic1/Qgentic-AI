"""Schemas for StarterAgent structured outputs."""

from typing import Literal

from pydantic import BaseModel

TaskType = Literal["computer_vision", "nlp", "tabular", "time_series", "audio"]


class StarterSuggestions(BaseModel):
    """Schema for starter agent task type classification and summary."""

    task_types: list[TaskType]
    task_summary: str
