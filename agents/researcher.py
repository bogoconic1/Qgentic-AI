import json
import os
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from tools.researcher import ask_eda, get_tools
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
        self.client = OpenAI(api_key=os.environ.get("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")
        self.messages: List[dict] = []

    def _compose_system(self) -> str:
        base_dir = os.path.join("task", self.slug)
        self.description = _safe_read(os.path.join(base_dir, "description.md"))
        return f"""
You are an experienced Kaggle Competitions Grandmaster. Your strength is not in writing code, but in quickly identifying the 'game within the game.' You excel at finding hidden nuances in the data description, evaluation metric, and community discussions to form a winning strategy. You are methodical, prioritizing foundational understanding before exploring complex solutions.

Your goal is to design a winning strategy for this competition.

You have access to one function tool to help you:
- `ask_eda(question)` which can answer exploratory questions about the data.

Call the function as needed and finish with a technical plan for the Developer.

Guidelines:
- Ask at most one clarifying question at a time via the tool
- Every tool question must be concise (1-2 lines)
- Stop calling tools once you can state the final instruction
- The final instruction should be a technical plan for the Developer to implement

Competition Description:
{self.description}

For your benefit, here are some baselines that can help you get started:
{_safe_read(os.path.join(base_dir, "public_insights.md"))}

IMPORTANT: DO NOT OPTIMIZE FOR THE EFFICIENCY PRIZE
"""

    def build_plan(self, max_steps: int = 16) -> str:
        system_prompt = self._compose_system()
        self.messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Begin your investigation. Use the ask_eda tool as needed. Please win the competition. All the best."
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
                "model": "qwen/qwen3-next-80b-a3b-instruct",
                "messages": self.messages,
                "tools": tools,
                "tool_choice": "auto",
            }

            msg_content = "<tool_call>"

            while "<tool_call>" in msg_content:
                completion = call_llm_with_retry(self.client, **llm_params)
                msg = completion.choices[0].message
                msg_content = msg.content
                
            self.messages.append(msg.model_dump())

            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    function_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                    print(f"Tool #{step + 1}: {function_name}({arguments})")
                    if function_name == "ask_eda":
                        question = arguments.get("question", "")
                        if len(question) == 0:
                            tool_output = "Your question cannot be answered based on the competition discussion threads."
                        else:
                            tool_output = ask_eda(question, self.description, data_path=f"task/{self.slug}")

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
            print("=="*50)
            print(f"Step {step + 1} final content: \n{final_content}")
            if len(final_content) == 0:
                raise RuntimeError("No plan produced. Please review the docs and craft a plan.")
            return final_content
