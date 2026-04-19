from pydantic import BaseModel


class StackTraceSolution(BaseModel):
    """Schema for stack trace debugging solutions."""

    reasoning: str
    web_search_findings: str
    solution: str


class LogMonitorVerdict(BaseModel):
    """LLM verdict on whether a running training script is healthy."""

    reasoning: str
    action: str  # "continue" or "kill"


class Review(BaseModel):
    """TODO: re-enable when Main Agent takes over review (#230)."""
    pass
