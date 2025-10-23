"""Model Recommender Agent for generating model-specific strategies."""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
import weave

from project_config import get_config
from tools.helpers import call_llm_with_retry
from prompts.model_recommender_agent import (
    build_unified_system_prompt,
)


logger = logging.getLogger(__name__)


_CONFIG = get_config()
_LLM_CFG = _CONFIG.get("llm")
_PATH_CFG = _CONFIG.get("paths")
_MODEL_REC_CFG = _CONFIG.get("model_recommender")

_MODEL_RECOMMENDER_MODEL = _LLM_CFG.get("model_recommender_model")
_DEFAULT_MODELS = _MODEL_REC_CFG.get("default_models", ["deberta-v3-large"])
_ENABLE_WEB_SEARCH = _MODEL_REC_CFG.get("enable_web_search", True)

_TASK_ROOT = Path(_PATH_CFG.get("task_root"))
_OUTPUTS_DIRNAME = _PATH_CFG.get("outputs_dirname")


class ModelRecommenderAgent:
    """Generates complete, coherent ML strategies with NOW/LATER prioritization.

    For each model in the provided list, generates a unified strategy covering:
    - Preprocessing & Data Preparation (NOW: foundations, LATER: advanced)
    - Loss Function (NOW: standard, LATER: custom variants)
    - Hyperparameters & Architecture (NOW: simple baseline, LATER: enhancements)
    - Inference Strategy (NOW: direct prediction, LATER: TTA/ensembling/calibration)

    Uses a single unified LLM call to ensure internal consistency and coherence.
    NOW recommendations form the baseline, LATER provide an optimization roadmap.

    Outputs to: task/{slug}/outputs/{iteration}/model_recommendations.json
    """

    def __init__(self, slug: str, iteration: int):
        load_dotenv()
        self.slug = slug
        self.iteration = iteration

        self.task_root = _TASK_ROOT
        self.outputs_dirname = _OUTPUTS_DIRNAME
        self.base_dir = self.task_root / slug
        self.outputs_dir = self.base_dir / self.outputs_dirname / str(iteration)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        # Output paths
        self.json_path = self.outputs_dir / "model_recommendations.json"
        self.log_path = self.outputs_dir / "model_recommender.log"

        # Configure logger
        self._configure_logger()

        # Load inputs once
        self.inputs = self._load_inputs()

        logger.info(
            "ModelRecommenderAgent initialized for slug=%s iteration=%s",
            self.slug,
            self.iteration,
        )

    def _configure_logger(self) -> None:
        """Configure file logging for model recommender."""
        existing_paths = {getattr(h, "baseFilename", None) for h in logger.handlers}
        if str(self.log_path) not in existing_paths:
            try:
                fh = logging.FileHandler(self.log_path)
                fh.setFormatter(
                    logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
                )
                logger.addHandler(fh)
            except Exception:
                pass
        logger.setLevel(logging.DEBUG)

    def _load_inputs(self) -> Dict[str, Any]:
        """Load all necessary inputs for model recommendations.

        Returns:
            dict with keys: description, task_type, task_summary, plan (optional)
        """
        inputs = {}

        # Load competition description
        description_path = self.base_dir / "description.md"
        if description_path.exists():
            with open(description_path, "r") as f:
                inputs["description"] = f.read()
        else:
            logger.warning("No description.md found at %s", description_path)
            inputs["description"] = ""

        # Load task_type and task_summary from starter_suggestions.json
        starter_path = self.outputs_dir / "starter_suggestions.json"
        if starter_path.exists():
            try:
                with open(starter_path, "r") as f:
                    starter_data = json.load(f)
                inputs["task_type"] = starter_data.get("task_type", "")
                inputs["task_summary"] = starter_data.get("task_summary", "")
                logger.info(
                    "Loaded task_type=%s from starter_suggestions.json",
                    inputs["task_type"],
                )
            except Exception as e:
                logger.warning("Failed to parse starter_suggestions.json: %s", e)
                inputs["task_type"] = ""
                inputs["task_summary"] = ""
        else:
            logger.warning("No starter_suggestions.json found at %s", starter_path)
            inputs["task_type"] = ""
            inputs["task_summary"] = ""

        # Load research plan (optional)
        plan_path = self.outputs_dir / "plan.md"
        if plan_path.exists():
            with open(plan_path, "r") as f:
                inputs["plan"] = f.read()
            logger.info("Loaded plan.md with %d characters", len(inputs["plan"]))
        else:
            logger.info("No plan.md found (optional)")
            inputs["plan"] = None

        return inputs

    @staticmethod
    def _extract_json_block(text: str) -> Optional[str]:
        """Extract JSON block from LLM response.

        Tries to find ```json ... ``` block first, then falls back to
        finding outermost braces.

        Args:
            text: LLM response text

        Returns:
            JSON string or None if not found
        """
        if not text:
            return None

        # Prefer fenced JSON block
        try:
            m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        except Exception:
            pass

        # Fallback: find outermost braces
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            return text[start:end]
        except Exception:
            return None

    @weave.op()
    def _recommend_complete_strategy(self, model_name: str) -> Dict[str, Any]:
        """Generate complete unified strategy for a single model using NOW/LATER approach.

        Args:
            model_name: Model name (e.g., "deberta-v3-large")

        Returns:
            Dict containing all recommendation types with NOW/LATER categorization
        """
        logger.info("[%s] Starting unified strategy generation", model_name)

        # Build unified system prompt
        system_prompt = build_unified_system_prompt(
            task_type=self.inputs.get("task_type", ""),
            description=self.inputs.get("description", ""),
            research_plan=self.inputs.get("plan", ""),
            model_name=model_name,
        )

        # Simple user message to trigger generation
        messages = [{"role": "user", "content": "Generate the complete ML strategy."}]

        try:
            response = call_llm_with_retry(
                model=_MODEL_RECOMMENDER_MODEL,
                instructions=system_prompt,
                tools=[],
                messages=messages,
                web_search_enabled=_ENABLE_WEB_SEARCH,
            )
            content = response.output_text or ""

            # Extract JSON
            json_text = self._extract_json_block(content)
            if not json_text:
                logger.warning("[%s] No JSON block found in unified strategy response", model_name)
                return self._empty_strategy()

            # Parse JSON
            strategy = json.loads(json_text)
            logger.info("[%s] Successfully parsed unified strategy", model_name)

            # Validate structure
            expected_keys = ["preprocessing", "loss_function", "hyperparameters", "inference"]
            for key in expected_keys:
                if key not in strategy:
                    logger.warning("[%s] Missing '%s' in strategy, adding empty", model_name, key)
                    strategy[key] = {"NOW": [], "LATER": []}

            return strategy

        except json.JSONDecodeError as e:
            logger.warning("[%s] JSON parsing failed for unified strategy: %s", model_name, e)
            return self._empty_strategy()
        except Exception as e:
            logger.error("[%s] Error getting unified strategy: %s", model_name, e)
            return self._empty_strategy()

    def _empty_strategy(self) -> Dict[str, Any]:
        """Return empty strategy structure as fallback."""
        return {
            "preprocessing": {"NOW": [], "LATER": []},
            "loss_function": {"NOW": {"loss": "", "explanation": ""}, "LATER": []},
            "hyperparameters": {
                "NOW": {"core_hyperparameters": {}, "architecture": ""},
                "LATER": {"training_enhancements": [], "architecture_enhancements": []}
            },
            "inference": {"NOW": [], "LATER": []}
        }

    @weave.op()
    def run(self, model_list: Optional[list[str]] = None) -> Dict[str, Dict[str, Any]]:
        """Run model recommender for all models in the list.

        Args:
            model_list: List of model names. If None, uses default_models from config.

        Returns:
            Dict mapping model names to their complete strategies (with NOW/LATER categorization)
        """
        if model_list is None:
            model_list = _DEFAULT_MODELS
            logger.info("Using default models from config: %s", model_list)

        if not model_list:
            logger.error("No models provided for recommendations")
            raise ValueError("model_list cannot be empty")

        logger.info("Processing %d model(s): %s", len(model_list), model_list)

        all_strategies = {}

        for model_name in model_list:
            try:
                strategy = self._recommend_complete_strategy(model_name)
                all_strategies[model_name] = strategy
            except Exception as e:
                logger.error("[%s] Failed to get strategy: %s", model_name, e)
                # Continue with other models
                continue

        if not all_strategies:
            logger.error("Failed to get strategies for any model")
            raise RuntimeError("No model strategies were generated successfully")

        # Persist to JSON
        try:
            with open(self.json_path, "w") as f:
                json.dump(all_strategies, f, indent=2)
            logger.info("Model strategies saved to %s", self.json_path)
        except Exception as e:
            logger.error("Failed to save model strategies: %s", e)

        logger.info(
            "ModelRecommenderAgent completed for %d model(s)", len(all_strategies)
        )
        return all_strategies
