import json
import logging
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

from tools.researcher import ask_eda, download_external_datasets, get_tools
from tools.helpers import call_llm_with_retry
import weave


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
            )

        # Prevent duplicate handlers
        existing_paths = {
            getattr(handler, "baseFilename", None)
            for handler in logger.handlers
        }
        if str(self.researcher_log_path) not in existing_paths:
            file_handler = logging.FileHandler(self.researcher_log_path)
            logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)
        logger.info(
            "ResearcherAgent initialized for slug=%s iteration=%s", self.slug, self.iteration
        )
        
    def _compose_system(self) -> str:
        base_dir = os.path.join("task", self.slug)
        self.description = _safe_read(os.path.join(base_dir, "description.md"))
        return f"""Role: Lead Research Strategist for Kaggle Machine-Learning Competition Team

Objective:
- Guide the developer by uncovering the underlying behavior of the dataset and providing evidence-based recommendations to support the development of a winning solution.
- Focus exclusively on research and evidence gathering; do not write production code yourself.

Instructions:
- Begin with a concise checklist (3-7 bullets) of what you will do; keep items conceptual, not implementation-level.
- Formulate and test hypotheses using available tools, alternating between asking analytical questions and confirming findings through data inspection.
- For every modeling or feature engineering suggestion, reference concrete evidence derived from data analysis.
- Do not rely on intuition or memory when data can directly inform your conclusions.
- After each tool call, validate results in 1-2 lines. If outcomes are inconclusive or incorrect, self-correct or design a follow-up investigation.

Tooling:
- Use only tools explicitly listed below. For routine read-only tasks call automatically; for destructive or state-changing operations require confirmation.
- Before any significant tool call, briefly state the purpose and specify the minimal required inputs.
- `ask_eda(question)`: Executes Python-based exploratory data analysis (EDA) on the local dataset. Use to assess distributions, data quality, leakage, and verify all critical assumptions.
- `download_external_datasets(query)`: Fetches relevant external datasets into `task/{self.slug}/` for further investigation. You may use `ask_eda` on these datasets as well.

Operating Principles (Competition-Agnostic):
1. Identify and clarify the target variable(s), feature space(s), and evaluation metric(s) based on the competition description.
2. Analyze target distribution (label balance), missing values, feature ranges, and overall dataset size.
3. Examine the structure of input data—considering properties like length distributions, numerical scales, category counts, sequence lengths, or image dimensions.
4. Probe for potential data leakage, ordering effects (temporal or spatial), and shifts between training and test distributions.
5. Ensure that every significant final recommendation is clearly motivated by previously cited tool outputs—do not assert unverified claims.
6. Restrict each tool call to investigating one hypothesis or question at a time. If results are inconclusive, follow up with more focused investigation.
7. List all relevant external datasets you recommend for the developer's consideration, and clearly state how each could be used.

Deliverable:
- Build a step-by-step technical plan for the developer. Every recommendation should be supported with specific dataset insights from tool runs (e.g., “Class distribution is skewed toward label 6 per Tool Run #2; therefore we…”). Also, highlight any open risks or unresolved questions.

Competition Description:
{self.description}

Competitor Strategies:
Refer to the following for what other competitors have tried—this may inform your approach:
{_safe_read(os.path.join(base_dir, "public_insights.md"))}

Note: DO NOT optimize for the efficiency prize.
"""

    @weave.op()
    def build_plan(self, max_steps: int = 512) -> str:
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
                # "model": "qwen/qwen3-next-80b-a3b-thinking",
                "model": "gpt-5",
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
                if msg.tool_calls:
                    break
                
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
                        logger.info(f"{question}")
                        if len(question) == 0:
                            tool_output = "Your question cannot be answered based on the competition discussion threads."
                        else:
                            tool_output = ask_eda(question, self.description, data_path=f"task/{self.slug}")

                        logger.info("```tool")
                        logger.info(f"{tool_output}")
                        logger.info("```")

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