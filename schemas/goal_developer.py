from pydantic import BaseModel


class GoalReview(BaseModel):
    """Structured review of a single goal-mode iteration.

    Returned by the review LLM call after each candidate script runs. The
    framework reads `done` to terminate the loop, `score` to log progress,
    `violations` to surface hard-constraint failures, and `next_step` to
    feed the next code-generation call.
    """

    reasoning: str
    is_valid: bool
    violations: list[str]
    score: float
    done: bool
    next_step: str
