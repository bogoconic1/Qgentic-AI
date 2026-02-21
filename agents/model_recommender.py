"""Model Recommender Agent for generating model-specific strategies."""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
import weave
from weave.trace.util import ThreadPoolExecutor

from project_config import get_config, get_instructions
from tools.helpers import call_llm
from tools.generate_paper_summary import PaperSummaryClient
from utils.llm_utils import extract_text_from_response, append_message
from prompts.model_recommender_agent import (
    model_selector_system_prompt,
    model_refiner_system_prompt,
    build_refiner_user_prompt,
    preprocessing_system_prompt,
    loss_function_system_prompt,
    hyperparameter_tuning_system_prompt,
    inference_strategy_system_prompt,
    build_user_prompt,
)
from schemas.model_recommender import (
    ModelSelection,
    RefinedModelSelection,
    PreprocessingRecommendations,
    LossFunctionRecommendations,
    HyperparameterRecommendations,
    InferenceStrategyRecommendations,
)


logger = logging.getLogger(__name__)


_CONFIG = get_config()
_LLM_CFG = _CONFIG.get("llm")
_PATH_CFG = _CONFIG.get("paths")
_MODEL_REC_CFG = _CONFIG.get("model_recommender")
_RUNTIME_CFG = _CONFIG.get("runtime")

_MODEL_SELECTOR_MODEL = _LLM_CFG.get("model_selector_model")
_MODEL_RECOMMENDER_MODEL = _LLM_CFG.get("model_recommender_model")
_HITL_MODELS = get_instructions()["# Models"]
_ENABLE_WEB_SEARCH = _MODEL_REC_CFG.get("enable_web_search", True)

_TASK_ROOT = Path(_PATH_CFG.get("task_root"))
_OUTPUTS_DIRNAME = _PATH_CFG.get("outputs_dirname")

# Time limits from config (in seconds)
_BASELINE_CODE_TIMEOUT = _RUNTIME_CFG.get("baseline_code_timeout")


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

        self.json_path = self.outputs_dir / "model_recommendations.json"
        self.log_path = self.outputs_dir / "model_recommender.log"

        self._configure_logger()

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

        description_path = self.base_dir / "description.md"
        if description_path.exists():
            with open(description_path, "r") as f:
                inputs["description"] = f.read()
        else:
            logger.warning("No description.md found at %s", description_path)
            inputs["description"] = ""

        starter_path = self.outputs_dir / "starter_suggestions.json"
        if starter_path.exists():
            try:
                with open(starter_path, "r") as f:
                    starter_data = json.load(f)
                task_types = starter_data.get("task_types", [])
                # Pass as list to support multimodal competitions
                inputs["task_type"] = task_types
                inputs["task_summary"] = starter_data.get("task_summary", "")
                logger.info(
                    "Loaded task_types=%s from starter_suggestions.json",
                    inputs["task_type"],
                )
            except Exception as e:
                logger.warning("Failed to parse starter_suggestions.json: %s", e)
                inputs["task_type"] = []
                inputs["task_summary"] = ""
        else:
            logger.warning("No starter_suggestions.json found at %s", starter_path)
            inputs["task_type"] = []
            inputs["task_summary"] = ""

        plan_path = self.outputs_dir / "plan.md"
        if plan_path.exists():
            with open(plan_path, "r") as f:
                inputs["plan"] = f.read()
            logger.info("Loaded plan.md with %d characters", len(inputs["plan"]))
        else:
            logger.info("No plan.md found (optional)")
            inputs["plan"] = None

        return inputs

    def _call_model_selector(self, system_prompt: str, messages: list, text_format=None):
        """Call the model selector.

        Args:
            system_prompt: System instruction
            messages: List of message dicts
            text_format: Optional Pydantic model for structured outputs

        Returns:
            LLM response
        """
        return call_llm(
            model=_MODEL_SELECTOR_MODEL,
            system_instruction=system_prompt,
            messages=messages,
            text_format=text_format,
            enable_google_search=_ENABLE_WEB_SEARCH,
        )

    def _get_structured_output(self, response):
        """Extract structured output from response.

        Args:
            response: LLM response object

        Returns:
            Pydantic model instance or None
        """
        return response if response else None

    @weave.op()
    def _recommend_preprocessing(self, model_name: str) -> Dict[str, Any]:
        """Get preprocessing recommendations for a model."""
        user_prompt = build_user_prompt(
            description=self.inputs["description"],
            task_type=self.inputs["task_type"],
            task_summary=self.inputs["task_summary"],
            model_name=model_name,
            research_plan=self.inputs.get("plan"),
        )

        time_limit_minutes = int(_BASELINE_CODE_TIMEOUT / 60)
        system_prompt = preprocessing_system_prompt(time_limit_minutes=time_limit_minutes)
        messages = [append_message("user", user_prompt)]

        try:
            response = self._call_model_selector(system_prompt, messages)

            response_text = extract_text_from_response(response)

            if not response_text:
                logger.warning("[%s] No response for preprocessing", model_name)
                return {}

            json_pattern = r'```json\s*(.*?)\s*```'
            matches = re.findall(json_pattern, response_text, re.DOTALL)

            if not matches:
                logger.warning("[%s] No JSON block found in preprocessing response", model_name)
                return {}

            result = json.loads(matches[0])
            logger.info("[%s] Successfully parsed preprocessing recommendations", model_name)

            return result

        except Exception as e:
            logger.error("[%s] Error getting preprocessing recommendations: %s", model_name, e)
            return {}

    @weave.op()
    def _recommend_loss_function(self, model_name: str) -> Dict[str, Any]:
        """Get loss function recommendation for a model."""
        user_prompt = build_user_prompt(
            description=self.inputs["description"],
            task_type=self.inputs["task_type"],
            task_summary=self.inputs["task_summary"],
            model_name=model_name,
            research_plan=self.inputs.get("plan"),
        )

        time_limit_minutes = int(_BASELINE_CODE_TIMEOUT / 60)
        system_prompt = loss_function_system_prompt(time_limit_minutes=time_limit_minutes)
        messages = [append_message("user", user_prompt)]

        try:
            response = self._call_model_selector(
                system_prompt, messages, text_format=LossFunctionRecommendations
            )

            parsed = self._get_structured_output(response)
            if not parsed:
                logger.warning("[%s] No structured output for loss function", model_name)
                return {}

            result = parsed.model_dump()
            logger.info("[%s] Successfully parsed loss function recommendations", model_name)
            return result

        except Exception as e:
            logger.error("[%s] Error getting loss function recommendations: %s", model_name, e)
            return {}

    @weave.op()
    def _recommend_hyperparameters(self, model_name: str) -> Dict[str, Any]:
        """Get hyperparameter and architecture recommendations for a model."""
        user_prompt = build_user_prompt(
            description=self.inputs["description"],
            task_type=self.inputs["task_type"],
            task_summary=self.inputs["task_summary"],
            model_name=model_name,
            research_plan=self.inputs.get("plan"),
        )

        time_limit_minutes = int(_BASELINE_CODE_TIMEOUT / 60)
        system_prompt = hyperparameter_tuning_system_prompt(time_limit_minutes=time_limit_minutes)
        messages = [append_message("user", user_prompt)]

        try:
            response = self._call_model_selector(
                system_prompt, messages, text_format=HyperparameterRecommendations
            )

            parsed = self._get_structured_output(response)
            if not parsed:
                logger.warning("[%s] No structured output for hyperparameters", model_name)
                return {}

            result = parsed.model_dump()
            logger.info("[%s] Successfully parsed hyperparameter recommendations", model_name)
            return result

        except Exception as e:
            logger.error("[%s] Error getting hyperparameter recommendations: %s", model_name, e)
            return {}

    @weave.op()
    def _recommend_inference(self, model_name: str) -> Dict[str, Any]:
        """Get inference strategy recommendations for a model."""
        user_prompt = build_user_prompt(
            description=self.inputs["description"],
            task_type=self.inputs["task_type"],
            task_summary=self.inputs["task_summary"],
            model_name=model_name,
            research_plan=self.inputs.get("plan"),
        )

        inference_time_limit_minutes = int(_BASELINE_CODE_TIMEOUT / 60)
        system_prompt = inference_strategy_system_prompt(inference_time_limit_minutes=inference_time_limit_minutes)
        messages = [append_message("user", user_prompt)]

        try:
            response = self._call_model_selector(
                system_prompt, messages, text_format=InferenceStrategyRecommendations
            )

            parsed = self._get_structured_output(response)
            if not parsed:
                logger.warning("[%s] No structured output for inference strategies", model_name)
                return {}

            result = parsed.model_dump()
            logger.info("[%s] Successfully parsed inference strategy recommendations", model_name)
            return result

        except Exception as e:
            logger.error("[%s] Error getting inference strategy recommendations: %s", model_name, e)
            return {}

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
        """Three-stage model selection process:
        Stage 1: Select 16 candidate models using LLM with web search
        Stage 2: Fetch paper summaries for all 16 candidates
        Stage 3: Refine to 8 final models using Gemini 2.5 Pro with paper summaries

        Returns:
            List of 8 final selected model names
        """
        # STAGE 1: Select 16 candidate models
        time_limit_minutes = int(_BASELINE_CODE_TIMEOUT / 60)
        system_prompt = model_selector_system_prompt(time_limit_minutes=time_limit_minutes)

        user_prompt = build_user_prompt(
            description=self.inputs["description"],
            task_type=self.inputs["task_type"],
            task_summary=self.inputs["task_summary"],
            model_name="",  # Not needed for model selection
            research_plan=self.inputs.get("plan"),
        )

        messages = [append_message("user", user_prompt)]

        response = self._call_model_selector(
            system_prompt, messages, text_format=ModelSelection
        )

        try:
            parsed = self._get_structured_output(response)
            if not parsed:
                logger.warning("No structured output received, using default models")
                return []

            if not parsed.recommended_models:
                logger.warning("No models in recommended_models array, using default models")
                return []

            candidate_models = [model.name.strip() for model in parsed.recommended_models if model.name.strip()]

            if not candidate_models:
                logger.warning("No valid model names extracted, using default models")
                return []

            logger.info("Stage 1: Selected %d candidate models", len(candidate_models))

            candidate_models_path = self.outputs_dir / "candidate_models.json"
            try:
                with open(candidate_models_path, "w") as f:
                    json.dump({"candidate_models": candidate_models}, f, indent=2)
            except Exception:
                pass

            # STAGE 2: Fetch paper summaries
            logger.info("Stage 2: Fetching paper summaries for %d candidates", len(candidate_models))
            summaries = self._fetch_paper_summaries(candidate_models)

            # STAGE 3: Refine to 8 final models
            logger.info("Stage 3: Refining to 8 final models")
            final_models = self._refine_model_selection(candidate_models, summaries)

            selected_models_path = self.outputs_dir / "selected_models.json"
            try:
                with open(selected_models_path, "w") as f:
                    json.dump({"selected_models": final_models}, f, indent=2)
            except Exception:
                pass

            logger.info("Three-stage selection complete: %d final models", len(final_models))
            return final_models

        except Exception as e:
            logger.error("Error in model selection: %s, using default models", e)
            return []

    @weave.op()
    def _fetch_paper_summaries(self, model_names: list[str]) -> Dict[str, str]:
        """Fetch paper summaries for all models in parallel using Gemini.

        Args:
            model_names: List of model names to fetch summaries for

        Returns:
            Dict mapping model names to their paper summaries
        """
        logger.info("Starting paper summary retrieval for %d models", len(model_names))

        client = PaperSummaryClient(is_model=True)
        summaries = {}

        with ThreadPoolExecutor(max_workers=len(model_names)) as executor:
            future_to_model = {
                executor.submit(client.generate_summary, model_name): model_name
                for model_name in model_names
            }

            for future in future_to_model:
                model_name = future_to_model[future]
                try:
                    summary = future.result(timeout=60)  # 60 second timeout per model
                    summaries[model_name] = summary
                    logger.info("[%s] Successfully retrieved paper summary", model_name)
                except Exception as e:
                    logger.warning("[%s] Failed to retrieve paper summary: %s", model_name, e)
                    summaries[model_name] = "Summary unavailable - no paper found or retrieval failed"

        summaries_path = self.outputs_dir / "candidate_model_summaries.json"
        try:
            with open(summaries_path, "w") as f:
                json.dump(summaries, f, indent=2)
            logger.info("Paper summaries saved to %s", summaries_path)
        except Exception as e:
            logger.warning("Failed to save paper summaries: %s", e)

        logger.info(
            "Paper summary retrieval completed: %d successful, %d failed",
            sum(1 for s in summaries.values() if not s.startswith("Summary unavailable")),
            sum(1 for s in summaries.values() if s.startswith("Summary unavailable")),
        )

        return summaries

    @weave.op()
    def _refine_model_selection(
        self,
        candidate_models: list[str],
        summaries: Dict[str, str]
    ) -> list[str]:
        """Refine 16 candidate models down to 8 using Gemini 2.5 Pro with paper summaries.

        Args:
            candidate_models: List of 16 candidate model names
            summaries: Dict mapping model names to their paper summaries

        Returns:
            List of 8 refined model names
        """
        logger.info("Starting model refinement: %d candidates -> 8 final models", len(candidate_models))

        time_limit_minutes = int(_BASELINE_CODE_TIMEOUT / 60)

        user_prompt = build_refiner_user_prompt(
            description=self.inputs["description"],
            task_type=self.inputs["task_type"],
            task_summary=self.inputs["task_summary"],
            research_plan=self.inputs.get("plan"),
            candidate_models=candidate_models,
            summaries=summaries,
            time_limit_minutes=time_limit_minutes,
        )

        system_instruction = model_refiner_system_prompt(time_limit_minutes=time_limit_minutes)

        try:
            messages = [append_message("user", user_prompt)]
            refined_selection = call_llm(
                model=_MODEL_RECOMMENDER_MODEL,
                system_instruction=system_instruction,
                messages=messages,
                text_format=RefinedModelSelection,
            )

            final_models = [model.name for model in refined_selection.final_models]

            logger.info("Model refinement completed: selected %d models", len(final_models))

            refined_path = self.outputs_dir / "refined_model_selection.json"
            try:
                with open(refined_path, "w") as f:
                    json.dump(refined_selection.model_dump(), f, indent=2)
                logger.info("Refined model selection saved to %s", refined_path)
            except Exception as e:
                logger.warning("Failed to save refined selection: %s", e)

            return final_models

        except Exception as e:
            logger.error("Model refinement failed: %s", e)
            logger.warning("Falling back to first 8 candidates")
            return candidate_models[:8]

    @weave.op()
    def run(self, model_list: Optional[list[str]] = None, use_dynamic_selection: bool = False) -> Dict[str, Dict[str, Any]]:
        """Run model recommender for all models in the list.

        Args:
            model_list: List of model names. If None and use_dynamic_selection=False, uses hitl_models from config.
                       If use_dynamic_selection=True, dynamically selects models using LLM.
            use_dynamic_selection: If True, use LLM to select models dynamically. Ignores model_list parameter.

        Returns:
            Dict mapping model names to their recommendations
        """
        if use_dynamic_selection:
            logger.info("Using dynamic model selection")
            model_list = self.select_models()
        elif model_list is None:
            model_list = _HITL_MODELS
            logger.info("Using HITL models from config: %s", model_list)

        if not model_list:
            logger.error("No models provided for recommendations")
            raise ValueError("model_list cannot be empty")

        logger.info("Processing %d model(s) in parallel: %s", len(model_list), model_list)

        all_recommendations = {}

        with ThreadPoolExecutor(max_workers=len(model_list)) as executor:
            future_to_model = {
                executor.submit(self._recommend_for_model, model_name): model_name
                for model_name in model_list
            }

            for future in future_to_model:
                model_name = future_to_model[future]
                try:
                    recommendations = future.result()
                    all_recommendations[model_name] = recommendations
                    logger.info("[%s] Recommendations generated successfully", model_name)

                    try:
                        with open(self.json_path, "w") as f:
                            json.dump(all_recommendations, f, indent=2)
                        logger.info("Model recommendations saved to %s", self.json_path)
                    except Exception as e:
                        logger.error("Failed to save model recommendations: %s", e)

                except Exception as e:
                    logger.error("[%s] Failed to get recommendations: %s", model_name, e)
                    # Continue with other models
                    continue

        if not all_recommendations:
            logger.error("Failed to get recommendations for any model")
            raise RuntimeError("No model recommendations were generated successfully")

        try:
            with open(self.json_path, "w") as f:
                json.dump(all_recommendations, f, indent=2)
            logger.info("Final model recommendations saved to %s", self.json_path)
        except Exception as e:
            logger.error("Failed to save final model recommendations: %s", e)

        logger.info(
            "ModelRecommenderAgent completed for %d model(s)", len(all_recommendations)
        )
        return all_recommendations