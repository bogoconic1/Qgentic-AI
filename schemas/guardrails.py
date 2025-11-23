from typing import Literal, List
from pydantic import BaseModel


class LeakageFinding(BaseModel):
    """A single data leakage finding."""
    rule_id: str
    snippet: str
    rationale: str
    suggestion: str


class LeakageReviewResponse(BaseModel):
    """Schema for LLM-based data leakage review response."""
    findings: List[LeakageFinding]
    severity: Literal["block", "warn", "none"]
