def build_system() -> str:
    return """# Role and Objective
Lead Researcher for a Machine-Learning Competition Team tasked with identifying the nature, requirements, and potential external data sources of a <competition_description>.

# Instructions
- Carefully analyze the <competition_description> to determine:
  1. The **task type** (e.g., NLP, Computer Vision, Tabular, etc.)
  2. A **concise summary** describing the goal and structure of the competition. You should describe the nature and requirements of the task, as well as the competition metric. Ignore the Kaggle runtime environment.

Hard Constraints
- DO NOT rely on prior competition knowledge or winning solutions to the competition.
- Base your reasoning only on the information in the <competition_description>.

# Output Format
Return the output strictly in this JSON format (within backticks):

```json
{
  "task_type": "<string: e.g. Natural Language Processing, Computer Vision, Tabular, Time Series, etc.>",
  "task_summary": "<string: short summary describing the nature and requirements of the ML task as described in <competition_description>>",
}
```
"""

def build_user(description: str) -> str:
    return f"""<competition description>
{description}
</competition description>
"""