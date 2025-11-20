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


class CodeGeneration(BaseModel):
    """Schema for generating training code and configuration in folder structure.

    Both fields should contain full file contents.
    """
    config_yaml: str
    train_py: str


class CodePatch(BaseModel):
    """Schema for patching existing training code and configuration.

    Each field contains a diff string for that specific file, or empty string if unchanged.
    - config_yaml: diff for config.yaml (e.g., '--- a/config.yaml\\n+++ b/config.yaml\\n...') or ""
    - train_py: diff for train.py (e.g., '--- a/train.py\\n+++ b/train.py\\n...') or ""
    """
    config_yaml: str
    train_py: str
