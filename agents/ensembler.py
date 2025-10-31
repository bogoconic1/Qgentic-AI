"""EnsemblerAgent: Combines baseline model predictions using iterative strategy testing."""

import json
import logging
import re
from pathlib import Path

import weave
from dotenv import load_dotenv

from project_config import get_config
from tools.ensembler import test_ensemble_strategy, get_tools
from tools.researcher import ask_eda
from prompts.ensembler_agent import (
    build_system as prompt_build_system,
    build_initial_user as prompt_initial_user,
)
from tools.helpers import call_llm_with_retry

logger = logging.getLogger(__name__)

_CONFIG = get_config()
_LLM_CFG = _CONFIG.get("llm")
_PATH_CFG = _CONFIG.get("paths")
_RUNTIME_CFG = _CONFIG.get("runtime")

_ENSEMBLER_MODEL = _LLM_CFG.get("ensembler_model")
_TASK_ROOT = Path(_PATH_CFG.get("task_root"))
_OUTPUTS_DIRNAME = _PATH_CFG.get("outputs_dirname")
_DEFAULT_MAX_STEPS = _RUNTIME_CFG.get("ensembler_max_steps")


class EnsemblerAgent:
    """Ensemble Strategy Developer with tool-calling.

    Analyzes baseline model predictions, tests ensemble strategies,
    and produces optimal combination approach using iterative LLM-guided
    exploration similar to ResearcherAgent.
    """

    def __init__(self, slug: str, iteration: int):
        """
        Initialize EnsemblerAgent.

        Args:
            slug: Competition slug
            iteration: Current iteration number
        """
        load_dotenv()
        self.slug = slug
        self.iteration = iteration

        # Setup paths
        self.base_dir = _TASK_ROOT / slug
        self.outputs_dir = self.base_dir / _OUTPUTS_DIRNAME / str(iteration)
        self.ensemble_folder = self.outputs_dir / "ensemble"

        # Ensure ensemble folder exists
        if not self.ensemble_folder.exists():
            raise FileNotFoundError(
                f"Ensemble folder not found at {self.ensemble_folder}. "
                "Please run move_best_code_to_ensemble_folder() first."
            )

        # Load metadata
        self.metadata_path = self.ensemble_folder / "ensemble_metadata.json"
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"ensemble_metadata.json not found at {self.metadata_path}"
            )

        with open(self.metadata_path) as f:
            self.metadata = json.load(f)

        # Load competition description
        with open(self.base_dir / "description.md") as f:
            self.description = f.read()

        # Track tested strategies
        self.tested_strategies = []  # List of dicts: {step, score, code}
        self.best_score = None

        self._configure_logger()

    def _configure_logger(self):
        """Setup logging for this agent."""
        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.DEBUG)

        log_dir = self.ensemble_folder / "logs"
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / "ensembler.txt"

        # Prevent duplicate handlers
        existing_paths = {
            getattr(h, "baseFilename", None) for h in logger.handlers
        }
        if str(log_path) not in existing_paths:
            handler = logging.FileHandler(log_path)
            handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
            )
            logger.addHandler(handler)

        logger.setLevel(logging.DEBUG)
        logger.info(
            "EnsemblerAgent initialized for slug=%s iteration=%s",
            self.slug, self.iteration
        )
        logger.info(f"Ensemble folder: {self.ensemble_folder}")
        logger.info(f"Models to ensemble: {list(self.metadata.keys())}")

    @weave.op()
    def run(self, max_steps: int | None = None) -> dict:
        """
        Main ensembler loop with tool calling.

        Iteratively uses ask_eda and test_ensemble_strategy tools to:
        1. Analyze model correlations and diversity
        2. Test different ensemble strategies
        3. Generate final Python code that creates submission_ens.csv

        Args:
            max_steps: Maximum number of tool-calling steps (default from config)

        Returns:
            dict with keys:
                - success: bool indicating if ensemble was created successfully
                - final_score: float score of final ensemble
                - final_submission_path: str path to submission_ens.csv
                - strategies_tested: list of all tested strategies
                - error: str error message if failed
        """
        max_steps = max_steps or _DEFAULT_MAX_STEPS
        logger.info("Starting EnsemblerAgent with max_steps=%s", max_steps)

        # Build prompts
        system_prompt = prompt_build_system(
            description=self.description,
            metadata=self.metadata,
            ensemble_folder=str(self.ensemble_folder),
        )

        input_list = [
            {"role": "user", "content": prompt_initial_user(self.metadata)}
        ]

        tools = get_tools()

        for step in range(max_steps):
            logger.info("[Ensembler] Step %s/%s", step + 1, max_steps)

            # Force finalization on last step
            if step == max_steps - 1:
                input_list.append({
                    "role": "user",
                    "content": "This is your FINAL step. Output the final ensemble Python code now!"
                })
                logger.info("Reached final step; forcing finalization")

            # Call LLM with tools
            response = call_llm_with_retry(
                model=_ENSEMBLER_MODEL,
                instructions=system_prompt,
                tools=tools,
                messages=input_list
            )

            input_list += response.output
            tool_calls = False

            # Process tool calls
            for item in response.output:
                if item.type == "function_call":
                    tool_calls = True

                    if item.name == "ask_eda":
                        try:
                            question = json.loads(item.arguments)["question"]
                        except Exception as e:
                            logger.error("Failed to parse ask_eda arguments: %s", e)
                            question = ""

                        if not question:
                            result = "Error: No question provided. Please specify an EDA question."
                        else:
                            logger.info(f"EDA Question: {question}")

                            result = ask_eda(
                                question=question,
                                description=self.description,
                                data_path=str(self.ensemble_folder)
                            )

                        input_list.append({
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": json.dumps({"insights": result})
                        })

                    elif item.name == "test_ensemble_strategy":
                        try:
                            strategy = json.loads(item.arguments)["strategy"]
                        except Exception as e:
                            logger.error("Failed to parse test_ensemble_strategy arguments: %s", e)
                            strategy = ""

                        if not strategy:
                            result = "Error: No strategy provided. Please describe an ensemble strategy to test."
                        else:
                            logger.info("Testing strategy: %s", strategy)

                            result = test_ensemble_strategy(
                                strategy=strategy,
                                ensemble_folder=self.ensemble_folder,
                                slug=self.slug
                            )

                        input_list.append({
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": json.dumps({"result": result})
                        })

            # If no tool calls, LLM has output final plan
            if not tool_calls:
                final_content = response.output_text or ""
                logger.info("Final ensemble plan received at step %s with length=%s", step + 1, len(final_content))

                if len(final_content) == 0:
                    logger.error("LLM returned empty final plan at step %s", step + 1)
                    raise RuntimeError("No ensemble plan produced. Please review the analysis and produce a final plan.")

                logger.info("Ensembler run completed successfully")
                return final_content

        # Exhausted steps without explicit finalization
        logger.error("Exhausted max_steps=%s without finalization", max_steps)
        raise RuntimeError(f"Exhausted {max_steps} steps without producing final ensemble plan.")
