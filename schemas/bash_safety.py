"""Schema for the bash-safety LLM judge."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class BashSafetyVerdict(BaseModel):
    """Verdict from the LLM judge on whether a shell command is safe to run."""

    verdict: Literal["allow", "block"]
    reason: str = Field(
        description="One-line justification — must be present even when allow."
    )
