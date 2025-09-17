import json
import os
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from tools.researcher import ask_domain_expert, get_tools
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
        return (
            "You are the Researcher agent (Lead Research Strategist).\n"
            "You have access to a function tool `ask_domain_expert(question)` which \n"
            "answers questions using the provided docs and threads.\n"
            "Goal: produce a clear, summarized step-by-step coding plan for the Developer. Each step should not be more than 2 lines.\n"
            "- Ask one clarifying question at a time via the tool when needed.\n"
            "- Stop calling tools once you can write the final plan.\n"
            "- Output the final plan as Markdown with concrete, actionable steps.\n\n"
            "- The Developer is expected to write all code within ONE python script.\n"
            f"Competition Overview:\n{overview}\n\n"
            f"Data Description:\n{data_description}\n"
        )

    def build_plan(self, max_steps: int = 16) -> str:
        system_prompt = self._compose_system()
        winning_approaches_summary = ask_domain_expert("What are the possible winning approaches to the competition? Summarize them.")
        print(f"Winning approaches summary: \n{winning_approaches_summary}")
        self.messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Begin your investigation. Use the ask_domain_expert tool as needed "
                    "When you no longer need tools, provide the final "
                    "step-by-step instructions for the Developer in Markdown."
                ),
            },
            {
                "role": "user",
                "content": (
                    "This is a short brief on possible winning approaches to the competition. "
                    "Please use this to guide your investigation. "
                    f"Possible winning approaches: \n{winning_approaches_summary}"
                )
            },
        ]

        tools = get_tools()

        for step in range(max_steps):
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
                            tool_output = "I don't know"
                        else:
                            tool_output = ask_domain_expert(question)

                        print(f"Tool #{step + 1}: {function_name}({tool_output})")

                        self.messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": tool_output or "I don't know",
                            }
                        )
                continue

            # No tool calls -> final plan
            final_content = msg.content or ""
            print(f"Step {step + 1} final content: \n{final_content}")
            if len(final_content) == 0:
                raise RuntimeError("No plan produced. Please review the docs and craft a plan.")
            return final_content

