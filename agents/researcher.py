import json
import os
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from tools.researcher import ask_domain_expert, ask_eda, get_tools
from tools.helpers import call_llm_with_retry


def _safe_read(path: str) -> str:
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception:
        return ""


class ResearcherAgent:
    """Lead Research Strategist with tool-calling.

    Uses OpenAI tool calling to decide when/what to ask via ask_domain_expert.
    Continues executing tool calls until the model returns a final plan
    (no further tool calls), or the tool-call budget is exhausted.
    """

    def __init__(self, slug: str):
        load_dotenv()
        self.slug = slug
        os.environ["TASK_SLUG"] = slug
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.messages: List[dict] = []

    def _compose_system(self) -> str:
        base_dir = os.path.join("task", self.slug)
        overview = _safe_read(os.path.join(base_dir, "overview.md"))
        data_description = _safe_read(os.path.join(base_dir, "data_description.md"))
        return f"""
You are an experienced Kaggle Competitions Grandmaster. Your strength is not in writing code, but in quickly identifying the 'game within the game.' You excel at finding hidden nuances in the data description, evaluation metric, and community discussions to form a winning strategy. You are methodical, prioritizing foundational understanding before exploring complex solutions.

Your goal is to design a winning strategy for this competition.

You have access to two function tools 
- `ask_domain_expert(question)` which is trained on Kaggle Discussions related to the competition. There could be hints on how to achieve high rankings in the competitions, be it data processing, modelling approaches, or postprocessing.
- `ask_eda(question)` which can answer exploratory questions about the data.

Call the function as needed and finish with one high-level instruction for the Developer.

Guidelines:
- Ask at most one clarifying question at a time via the tool
- Every tool question must be concise (1-2 lines), end with a question mark
- The question should guide you towards the goal of winning the competition
- Stop calling tools once you can state the final instruction
- The final instruction should be a high-level overview for the Developer to implement
- Do not specify particular technologies or libraries in your final instruction

Competition Overview:
{overview}

Data Description:
{data_description}
"""

    def build_plan(self, max_steps: int = 16) -> str:
        system_prompt = self._compose_system()
        self.messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Begin your investigation. Use the ask_domain_expert tool as needed. Please win the competition. All the best."
                ),
            },
        ]
        

        tools = get_tools()

        for step in range(max_steps):
            print(self.messages[-2:])
            print("--"*50)
            print(f"\n[Step {step + 1}/{max_steps}]")
            if step == max_steps - 1:
                self.messages.append({"role": "user", "content": "This is your FINAL step. Output the final plan now!"})

            llm_params = {
                "model": "gpt-5",
                "messages": self.messages,
                "tools": tools,
                "tool_choice": "auto",
                "max_completion_tokens": 8192,
            }

            completion = call_llm_with_retry(self.client, **llm_params)

            msg = completion.choices[0].message
            self.messages.append(msg.model_dump())

            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    function_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                    print(f"Tool #{step + 1}: {function_name}({arguments})")
                    if function_name == "ask_domain_expert":
                        question = arguments.get("question", "")
                        if len(question) == 0:
                            tool_output = "Your question cannot be answered based on the competition discussion threads."
                        else:
                            tool_output = ask_domain_expert(question)

                        print(f"Tool #{step + 1}: {function_name}({tool_output})")

                        self.messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": tool_output or "Your question cannot be answered based on the competition discussion threads.",
                            }
                        )
                continue

            # No tool calls -> final plan
            final_content = msg.content or ""
            print(f"Step {step + 1} final content: \n{final_content}")
            if len(final_content) == 0:
                raise RuntimeError("No plan produced. Please review the docs and craft a plan.")
            return final_content
