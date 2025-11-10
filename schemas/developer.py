from pydantic import BaseModel


class StackTraceSolution(BaseModel):
    """Schema for stack trace debugging solutions."""
    checklist: list[str]
    web_search_findings: str
    reasoning_and_solution: str
    validation: str
    further_steps: str


class SOTAResponse(BaseModel):
    """Schema for SOTA suggestions combining blacklist decision and new suggestion."""
    blacklist: bool
    blacklist_reason: str
    suggestion: str
    suggestion_reason: str


class CodeSafetyCheck(BaseModel):
    """Schema for LLM-based code safety analysis."""
    decision: str  # "allow" or "block"
    confidence: float  # 0.0-1.0
    reasoning: str
    violations: list[str]  # List of specific security issues found
    suggested_fix: str  # How to fix the issues
