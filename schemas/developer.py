from pydantic import BaseModel


class StackTraceSolution(BaseModel):
    """Schema for stack trace debugging solutions."""

    reasoning: str
    web_search_findings: str
    solution: str


class SOTAResponse(BaseModel):
    """Schema for SOTA suggestions combining blacklist decision and new suggestion."""

    plan_summary: str
    red_flags_summary: str
    shared_experiences_summary: str
    research_summary: str
    blacklist_reasoning: str
    blacklist: bool
    suggestion_reasoning: str
    suggestion: str
    suggestion_code: str


class LogMonitorVerdict(BaseModel):
    """LLM verdict on whether a running training script is healthy."""

    reasoning: str
    action: str  # "continue" or "kill"


class RedFlagsResponse(BaseModel):
    """Red flags analysis result."""

    reasoning: str
    code_issues: str
    log_issues: str
    web_search_findings: str
    final_summary: str


class CodeGeneration(BaseModel):
    """Schema for generating training code in folder structure.

    The train_py field should contain the complete training script.
    """
    reasoning: str
    train_py: str
