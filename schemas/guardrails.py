from typing import Literal
from pydantic import BaseModel


class LeakageFinding(BaseModel):
    """A single data leakage finding."""

    rule_id: str
    snippet: str
    rationale: str
    suggestion: str


class LeakageReviewResponse(BaseModel):
    """Schema for LLM-based data leakage review response."""

    findings: list[LeakageFinding]
    severity: Literal["block", "warn", "none"]


class CodeSafetyCheck(BaseModel):
    """Schema for LLM-based code safety analysis."""

    decision: str  # "allow" or "block"
    confidence: float  # 0.0-1.0
    reasoning: str
    violations: list[str]  # List of specific security issues found
    suggested_fix: str  # How to fix the issues
