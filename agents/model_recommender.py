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
    model_selector_system_prompt,
    preprocessing_system_prompt,
    loss_function_system_prompt,
    hyperparameter_tuning_system_prompt,
    inference_strategy_system_prompt,
    build_user_prompt,
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
    """Recommends model-specific strategies (preprocessing, loss, hyperparameters, inference).

    For each model in the provided list, generates:
    - Preprocessing strategies (preprocessing, feature_creation, feature_selection, etc.)
    - Loss function recommendation
    - Hyperparameter and architecture recommendations
    - Inference strategies

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

    def _call_llm_for_recommendation(
        self, system_prompt: str, model_name: str, recommendation_type: str
    ) -> Optional[Dict[str, Any]]:
        """Call LLM to get a specific type of recommendation.

        Args:
            system_prompt: System prompt function output
            model_name: Model name (e.g., "deberta-v3-large")
            recommendation_type: Type of recommendation for logging

        Returns:
            Parsed JSON dict or None on failure
        """
        user_prompt = build_user_prompt(
            description=self.inputs["description"],
            task_type=self.inputs["task_type"],
            task_summary=self.inputs["task_summary"],
            model_name=model_name,
            research_plan=self.inputs.get("plan"),
        )

        messages = [{"role": "user", "content": user_prompt}]

        logger.info(
            "[%s] Requesting %s recommendations", model_name, recommendation_type
        )

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
                logger.warning(
                    "[%s] No JSON block found in %s response",
                    model_name,
                    recommendation_type,
                )
                return None

            # Parse JSON
            parsed = json.loads(json_text)
            logger.info(
                "[%s] Successfully parsed %s recommendations",
                model_name,
                recommendation_type,
            )
            return parsed

        except json.JSONDecodeError as e:
            logger.warning(
                "[%s] JSON parsing failed for %s: %s", model_name, recommendation_type, e
            )
            return None
        except Exception as e:
            logger.error(
                "[%s] Error getting %s recommendations: %s",
                model_name,
                recommendation_type,
                e,
            )
            return None

    def _recommend_preprocessing(self, model_name: str) -> Dict[str, Any]:
        """Get preprocessing recommendations for a model."""
        # Build user prompt with categories
        user_prompt = build_user_prompt(
            description=self.inputs["description"],
            task_type=self.inputs["task_type"],
            task_summary=self.inputs["task_summary"],
            model_name=model_name,
            research_plan=self.inputs.get("plan"),
        )

        # Call LLM
        system_prompt = preprocessing_system_prompt()
        messages = [{"role": "user", "content": user_prompt}]

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
                logger.warning("[%s] No JSON block found in preprocessing response", model_name)
                return {"preprocessing": [], "feature_creation": [], "feature_selection": [], "feature_transformation": [], "tokenization": [], "data_augmentation": []}

            # Parse JSON
            result = json.loads(json_text)
            logger.info("[%s] Successfully parsed preprocessing recommendations", model_name)

            return result

        except json.JSONDecodeError as e:
            logger.warning("[%s] JSON parsing failed for preprocessing: %s", model_name, e)
            return {"preprocessing": [], "feature_creation": [], "feature_selection": [], "feature_transformation": [], "tokenization": [], "data_augmentation": []}
        except Exception as e:
            logger.error("[%s] Error getting preprocessing recommendations: %s", model_name, e)
            return {"preprocessing": [], "feature_creation": [], "feature_selection": [], "feature_transformation": [], "tokenization": [], "data_augmentation": []}

    def _recommend_loss_function(self, model_name: str) -> Dict[str, Any]:
        """Get loss function recommendation for a model."""
        result = self._call_llm_for_recommendation(
            loss_function_system_prompt(), model_name, "loss_function"
        )
        if result is None:
            logger.warning("[%s] Using empty loss function recommendation", model_name)
            return {"loss_function": "", "explanation": ""}
        return result

    def _recommend_hyperparameters(self, model_name: str) -> Dict[str, Any]:
        """Get hyperparameter and architecture recommendations for a model."""
        result = self._call_llm_for_recommendation(
            hyperparameter_tuning_system_prompt(), model_name, "hyperparameters"
        )
        if result is None:
            logger.warning(
                "[%s] Using empty hyperparameter recommendations", model_name
            )
            return {"hyperparameters": [], "architectures": []}
        return result

    def _recommend_inference(self, model_name: str) -> Dict[str, Any]:
        """Get inference strategy recommendations for a model."""
        result = self._call_llm_for_recommendation(
            inference_strategy_system_prompt(), model_name, "inference"
        )
        if result is None:
            logger.warning("[%s] Using empty inference recommendations", model_name)
            return {"inference_strategies": []}
        return result

    @weave.op()
    def _recommend_for_model(self, model_name: str) -> Dict[str, Any]:
        """Generate all recommendations for a single model.

        Args:
            model_name: Model name (e.g., "deberta-v3-large")

        Returns:
            Dict containing all recommendation types
        """
        logger.info("[%s] Starting recommendations", model_name)

        recommendations = {
            "preprocessing": self._recommend_preprocessing(model_name),
            "loss_function": self._recommend_loss_function(model_name),
            "hyperparameters": self._recommend_hyperparameters(model_name),
            "inference_strategies": self._recommend_inference(model_name),
        }

        logger.info("[%s] All recommendations completed", model_name)
        return recommendations

    @weave.op()
    def select_models(self) -> list[str]:
        """Use LLM to dynamically select suitable models for the competition.

        Returns:
            List of up to 5 selected model names
        """
        logger.info("Starting dynamic model selection")

        system_prompt = model_selector_system_prompt()

        user_prompt = build_user_prompt(
            description=self.inputs["description"],
            task_type=self.inputs["task_type"],
            task_summary=self.inputs["task_summary"],
            model_name="",  # Not needed for model selection
            research_plan=self.inputs.get("plan"),
        )

        messages = [{"role": "user", "content": user_prompt}]

        logger.info("Requesting model selection from LLM")

        response = call_llm_with_retry(
            model=_MODEL_RECOMMENDER_MODEL,
            instructions=system_prompt,
            tools=[],
            messages=messages,
            web_search_enabled=_ENABLE_WEB_SEARCH,
        )

        content = response.output_text or ""
        logger.debug("Model selection response length: %d", len(content))

        # Extract JSON block
        json_str = self._extract_json_block(content)
        if not json_str:
            logger.warning("No JSON block found in model selection response, using default models")
            return _DEFAULT_MODELS

        # Parse JSON
        try:
            data = json.loads(json_str)
            recommended_models = data.get("recommended_models", [])

            if not recommended_models:
                logger.warning("No models in recommended_models array, using default models")
                return _DEFAULT_MODELS

            # Extract model names
            model_names = []
            for item in recommended_models:
                if isinstance(item, dict):
                    name = item.get("name", "").strip()
                    if name:
                        model_names.append(name)

            if not model_names:
                logger.warning("No valid model names extracted, using default models")
                return _DEFAULT_MODELS

            logger.info("Selected %d models: %s", len(model_names), model_names)

            # Save selected models to file for reference
            selected_models_path = self.outputs_dir / "selected_models.json"
            try:
                with open(selected_models_path, "w") as f:
                    json.dump({
                        "selected_models": model_names,
                        "raw_response": content
                    }, f, indent=2)
                logger.info("Selected models saved to %s", selected_models_path)
            except Exception as e:
                logger.warning("Failed to save selected models: %s", e)

            return model_names

        except json.JSONDecodeError as e:
            logger.warning("Failed to parse model selection JSON: %s, using default models", e)
            return _DEFAULT_MODELS

    @weave.op()
    def run(self, model_list: Optional[list[str]] = None, use_dynamic_selection: bool = False) -> Dict[str, Dict[str, Any]]:
        """Run model recommender for all models in the list.

        Args:
            model_list: List of model names. If None and use_dynamic_selection=False, uses default_models from config.
                       If use_dynamic_selection=True, dynamically selects models using LLM.
            use_dynamic_selection: If True, use LLM to select models dynamically. Ignores model_list parameter.

        Returns:
            Dict mapping model names to their recommendations
        """
        if use_dynamic_selection:
            logger.info("Using dynamic model selection")
            model_list = self.select_models()
        elif model_list is None:
            model_list = _DEFAULT_MODELS
            logger.info("Using default models from config: %s", model_list)

        if not model_list:
            logger.error("No models provided for recommendations")
            raise ValueError("model_list cannot be empty")

        logger.info("Processing %d model(s): %s", len(model_list), model_list)

        all_recommendations = {}

        for model_name in model_list:
            try:
                recommendations = self._recommend_for_model(model_name)
                all_recommendations[model_name] = recommendations
            except Exception as e:
                logger.error("[%s] Failed to get recommendations: %s", model_name, e)
                # Continue with other models
                continue

        if not all_recommendations:
            logger.error("Failed to get recommendations for any model")
            raise RuntimeError("No model recommendations were generated successfully")

        # Persist to JSON
        try:
            with open(self.json_path, "w") as f:
                json.dump(all_recommendations, f, indent=2)
            logger.info("Model recommendations saved to %s", self.json_path)
        except Exception as e:
            logger.error("Failed to save model recommendations: %s", e)

        logger.info(
            "ModelRecommenderAgent completed for %d model(s)", len(all_recommendations)
        )
        return all_recommendations
