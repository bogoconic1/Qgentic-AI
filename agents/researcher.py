import json
import logging
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

from tools.researcher import ask_eda, download_external_datasets, get_tools
from tools.helpers import call_llm_with_retry


def _safe_read(path: str) -> str:
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception:
        return ""

logger = logging.getLogger(__name__)


class ResearcherAgent:
    """Lead Research Strategist with tool-calling.

    Uses OpenAI tool calling to decide when/what to ask via ask_domain_expert.
    Continues executing tool calls until the model returns a final plan
    (no further tool calls), or the tool-call budget is exhausted.
    """

    def __init__(self, slug: str, iteration: int):
        load_dotenv()
        self.slug = slug
        self.iteration = iteration
        os.environ["TASK_SLUG"] = slug
        self.client = OpenAI(api_key=os.environ.get("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")
        self.messages: List[dict] = []
        self.base_dir = Path("task") / self.slug
        self.outputs_dir = self.base_dir / "outputs" / str(self.iteration)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.researcher_log_path = self.outputs_dir / "researcher.txt"
        self._configure_logger()

    def _configure_logger(self) -> None:
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.DEBUG,
                format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            )

        # Prevent duplicate handlers
        existing_paths = {
            getattr(handler, "baseFilename", None)
            for handler in logger.handlers
        }
        if str(self.researcher_log_path) not in existing_paths:
            file_handler = logging.FileHandler(self.researcher_log_path)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
            )
            logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)
        logger.info(
            "ResearcherAgent initialized for slug=%s iteration=%s", self.slug, self.iteration
        )

    def _compose_system(self) -> str:
        base_dir = os.path.join("task", self.slug)
        self.description = _safe_read(os.path.join(base_dir, "description.md"))
        return f"""
You are the lead research strategist for a machine-learning competition team. You never write production code yourself; instead you uncover how the dataset behaves so the developer can build a winning solution with evidence on hand.

Primary mandate:
- Build a deep understanding of the dataset before recommending any modeling steps.
- Alternate between forming hypotheses and confirming them with the `ask_eda` tool.
- Do not rely on memory or intuition when data can be inspected.

Tooling:
- `ask_eda(question)` executes Python against the local dataset. Use it to quantify distributions, check data quality, look for leakage, and validate every key assumption.
- `download_external_datasets(query)` downloads relevant external datasets in task/{self.slug} for further analysis and usefulness in aiding model improvements. You can call ask_eda on these downloaded datasets as well.

Operating principles (competition-agnostic):
1. Start by clarifying the target variable(s), feature space(s), and evaluation metric(s) from the competition description.
2. Investigate label balance or distribution, missing values, feature ranges, and dataset size.
3. Examine the structure of the input data (e.g., length distributions, numerical scales, categorical cardinalities, sequence lengths, or image dimensions).
4. Probe for potential leakage, ordering effects (temporal, spatial, or otherwise), or train/test distribution shifts.
5. Only stop calling tools once each major recommendation in your final plan cites concrete evidence from tool outputs.
6. Keep each tool call focused on one hypothesis at a time and follow up if results are inconclusive.
7. List all paths to the external datasets which you want the developer to consider, and also tell how to use them.

Deliverable:
- A step-by-step technical plan for the Developer. For every recommendation, reference the dataset insight that motivated it (e.g., “Class distribution is skewed toward label 6 per Tool Run #2; therefore we…”) and highlight remaining risks or unanswered questions.

Competition Description:
{self.description}

Supporting baseline notes (may be useful to see what other competitors are trying):
{_safe_read(os.path.join(base_dir, "public_insights.md"))}

IMPORTANT: DO NOT OPTIMIZE FOR THE EFFICIENCY PRIZE
"""

    def build_plan(self, max_steps: int = 32) -> str:
        system_prompt = self._compose_system()
        self.messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Remember: If you feel that external data is useful, use download_external_datasets and use ask_eda to analyze these as well. Form hypotheses about the available data and validate each one with ask_eda. Do not produce the final plan until you have evidence-backed insights covering distribution, data quality, and any potential leakage or shift."
                ),
            },
        ]

        logger.info("Researcher run started with max_steps=%s", max_steps)
        tools = get_tools()

        for step in range(max_steps):
            # logger.debug("Current conversation tail: %s", self.messages[-2:])
            logger.info("[Researcher] Step %s/%s", step + 1, max_steps)
            if step == max_steps - 1:
                self.messages.append({"role": "user", "content": "This is your FINAL step. Output the final plan now!"})
                logger.info("Reached final step; forcing plan output prompt")

            llm_params = {
                "model": "qwen/qwen3-next-80b-a3b-thinking",
                "messages": self.messages,
                "tools": tools,
                "tool_choice": "auto",
            }

            msg_content = "<tool_call>"

            while "<tool_call>" in msg_content or msg_content == "" or """{"name":""" in msg_content:
                completion = call_llm_with_retry(self.client, **llm_params)
                msg = completion.choices[0].message
                msg_content = msg.content
                logger.debug("Model response content length: %s", len(msg_content or ""))
                
            self.messages.append(msg.model_dump())

            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    function_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                    logger.info(
                        "Tool call issued: %s with arguments=%s", function_name, arguments
                    )
                    if function_name == "ask_eda":
                        question = arguments.get("question", "")
                        if len(question) == 0:
                            tool_output = "Your question cannot be answered based on the competition discussion threads."
                        else:
                            tool_output = ask_eda(question, self.description, data_path=f"task/{self.slug}")

                        logger.info("Tool response length=%s", len(tool_output or ""))

                        self.messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": tool_output or "Your question cannot be answered based on the competition discussion threads.",
                            }
                        )
                    elif function_name == "download_external_datasets":
                        query = arguments.get("query", "")
                        if not query:
                            tool_output = "Search query missing; please provide a specific data need."
                        else:
                            tool_output = download_external_datasets(query, self.slug)

                        logger.info("External search response length=%s", len(tool_output or ""))

                        self.messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": tool_output or "No relevant datasets found.",
                            }
                        )
                continue

            # No tool calls -> final plan
            final_content = msg.content or ""
            logger.info("Final plan received at step %s with length=%s", step + 1, len(final_content))
            if len(final_content) == 0:
                logger.error("LLM returned empty final plan at step %s", step + 1)
                raise RuntimeError("No plan produced. Please review the docs and craft a plan.")
            logger.info("Researcher run completed successfully")
            return final_content
