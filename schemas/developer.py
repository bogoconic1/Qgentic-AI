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
    suggestion_code: str


class LogMonitorVerdict(BaseModel):
    """LLM verdict on whether a running training script is healthy."""
    action: str   # "continue" or "kill"
    reason: str   # explanation — fed back to the developer agent if killed


class CodeGeneration(BaseModel):
    """Schema for generating training code in folder structure.

    The train_py field should contain the complete training script.
    """
    train_py: str
