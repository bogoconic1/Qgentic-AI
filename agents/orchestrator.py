from typing import Tuple
from pathlib import Path
import json

from agents.researcher import ResearcherAgent
from agents.developer import DeveloperAgent
from agents.starter import StarterAgent
from agents.model_recommender import ModelRecommenderAgent
from project_config import get_config
import weave


_CONFIG = get_config()
_RUNTIME_CFG = _CONFIG.get("runtime")
_PATH_CFG = _CONFIG.get("paths")
_TASK_ROOT = Path(_PATH_CFG.get("task_root"))
_OUTPUTS_DIRNAME = _PATH_CFG.get("outputs_dirname")
_DEFAULT_PARALLEL = int(_RUNTIME_CFG.get("researcher_parallel_runs"))

@weave.op()
def _run_researcher_once(slug: str, iteration: int, run_id: int) -> tuple[int, str, int]:
    agent = ResearcherAgent(slug, iteration, run_id=run_id)
    plan = agent.build_plan()
    outputs_dir = _TASK_ROOT / slug / _OUTPUTS_DIRNAME / str(iteration)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    if run_id == 1:
        plan_path = outputs_dir / "plan.md"
    else:
        plan_path = outputs_dir / f"plan_{run_id}.md"
    try:
        with open(plan_path, "w") as f:
            f.write(plan)
    except Exception:
        pass
    return run_id, str(plan_path), len(plan or "")

@weave.op()
def _run_developer_baseline(slug: str, iteration_suffix: str, model_name: str, now_recommendations: dict, key: str):
    """Run a single baseline DeveloperAgent and return (key, best_score, best_code).

    Args:
        slug: Competition slug
        iteration_suffix: Iteration identifier (e.g., "1_1")
        model_name: Model name (e.g., "deberta-v3-large")
        now_recommendations: NOW-only recommendations dict for this model
        key: Key for tracking results
    """
    # Format NOW recommendations for developer
    formatted_recommendations = _format_recommendations_for_developer(now_recommendations)

    dev = DeveloperAgent(slug, iteration_suffix, model_name=model_name, model_recommendations=formatted_recommendations)
    best_score, best_code, blacklisted_ideas = dev.run(max_time_seconds=9000)
    return key, best_score, best_code, blacklisted_ideas

def _extract_now_recommendations(recommendations: dict) -> dict:
    """Extract only NOW recommendations from model recommendations.

    Filters out citations and est_runtime_minutes_gpu fields.
    Returns a dict with the same structure but only NOW sections.
    """
    now_only = {}

    # Preprocessing - each category has NOW/LATER structure
    preprocessing = recommendations.get("preprocessing", {})
    if preprocessing:
        now_only["preprocessing"] = {}
        for category, content in preprocessing.items():
            if isinstance(content, dict) and "NOW" in content:
                now_items = content["NOW"]
                if isinstance(now_items, list):
                    now_only["preprocessing"][category] = {
                        "NOW": now_items
                    }

    # Loss function - NOW is an object, LATER is an array
    loss_fn = recommendations.get("loss_function", {})
    if loss_fn and "NOW" in loss_fn:
        now_only["loss_function"] = {
            "NOW": loss_fn["NOW"]
        }

    # Hyperparameters - NOW has hyperparameters and architectures arrays
    hyperparams = recommendations.get("hyperparameters", {})
    if hyperparams and "NOW" in hyperparams:
        now_section = hyperparams["NOW"]
        cleaned_now = {}
        if "hyperparameters" in now_section:
            cleaned_now["hyperparameters"] = now_section["hyperparameters"]
        if "architectures" in now_section:
            cleaned_now["architectures"] = now_section["architectures"]
        now_only["hyperparameters"] = {"NOW": cleaned_now}

    # Inference strategies - NOW has inference_strategies array
    inference = recommendations.get("inference_strategies", {})
    if inference and "NOW" in inference:
        now_section = inference["NOW"]
        if "inference_strategies" in now_section:
            now_only["inference_strategies"] = {
                "NOW": {
                    "inference_strategies": now_section["inference_strategies"]
                }
            }

    return now_only

def _extract_later_recommendations(recommendations: dict) -> dict:
    """Extract only LATER recommendations from model recommendations.

    Filters out citations and est_runtime_minutes_gpu fields.
    Returns a dict with the same structure but only LATER sections.
    """
    later_only = {}

    # Preprocessing - each category has NOW/LATER structure
    preprocessing = recommendations.get("preprocessing", {})
    if preprocessing:
        later_only["preprocessing"] = {}
        for category, content in preprocessing.items():
            if isinstance(content, dict) and "LATER" in content:
                later_items = content["LATER"]
                if isinstance(later_items, list):
                    later_only["preprocessing"][category] = {
                        "LATER": later_items
                    }

    # Loss function - LATER is an array
    loss_fn = recommendations.get("loss_function", {})
    if loss_fn and "LATER" in loss_fn:
        later_only["loss_function"] = {
            "LATER": loss_fn["LATER"]
        }

    # Hyperparameters - LATER has hyperparameters and architectures arrays
    hyperparams = recommendations.get("hyperparameters", {})
    if hyperparams and "LATER" in hyperparams:
        later_section = hyperparams["LATER"]
        cleaned_later = {}
        if "hyperparameters" in later_section:
            cleaned_later["hyperparameters"] = later_section["hyperparameters"]
        if "architectures" in later_section:
            cleaned_later["architectures"] = later_section["architectures"]
        later_only["hyperparameters"] = {"LATER": cleaned_later}

    # Inference strategies - LATER has inference_strategies array
    inference = recommendations.get("inference_strategies", {})
    if inference and "LATER" in inference:
        later_section = inference["LATER"]
        if "inference_strategies" in later_section:
            later_only["inference_strategies"] = {
                "LATER": {
                    "inference_strategies": later_section["inference_strategies"]
                }
            }

    return later_only

def _format_recommendations_for_developer(recommendations: dict) -> str:
    """Format model recommendations for DeveloperAgent.

    Formats the NOW recommendations into a readable string that can be passed
    to DeveloperAgent as guidance.
    """
    details = []

    # Preprocessing
    preprocessing = recommendations.get("preprocessing", {})
    if preprocessing:
        details.append("## Preprocessing Strategies")
        for category, content in preprocessing.items():
            if isinstance(content, dict) and "NOW" in content:
                now_items = content["NOW"]
                if now_items and isinstance(now_items, list):
                    details.append(f"\n### {category.replace('_', ' ').title()}")
                    for item in now_items:
                        if isinstance(item, dict):
                            strategy = item.get("strategy", "")
                            explanation = item.get("explanation", "")
                            if strategy:
                                details.append(f"- {strategy}: {explanation}")

    # Loss function
    loss_fn = recommendations.get("loss_function", {})
    if loss_fn and "NOW" in loss_fn:
        details.append("\n## Loss Function")
        now_loss = loss_fn["NOW"]
        loss_name = now_loss.get("loss_function", "")
        loss_exp = now_loss.get("explanation", "")
        if loss_name:
            details.append(f"- Use {loss_name}: {loss_exp}")

    # Hyperparameters
    hyperparams = recommendations.get("hyperparameters", {})
    if hyperparams and "NOW" in hyperparams:
        now_section = hyperparams["NOW"]

        details.append("\n## Hyperparameters")
        hp_list = now_section.get("hyperparameters", [])
        for item in hp_list:
            if isinstance(item, dict):
                hp = item.get("hyperparameter", "")
                explanation = item.get("explanation", "")
                if hp:
                    details.append(f"- {hp}: {explanation}")

        # Architectures
        arch_list = now_section.get("architectures", [])
        if arch_list:
            details.append("\n### Architecture Recommendations")
            for item in arch_list:
                if isinstance(item, dict):
                    arch = item.get("architecture", "")
                    explanation = item.get("explanation", "")
                    if arch:
                        details.append(f"- {arch}: {explanation}")

    # Inference strategies
    inference = recommendations.get("inference_strategies", {})
    if inference and "NOW" in inference:
        now_section = inference["NOW"]
        details.append("\n## Inference Strategies")
        strategies = now_section.get("inference_strategies", [])
        for item in strategies:
            if isinstance(item, dict):
                strategy = item.get("strategy", "")
                explanation = item.get("explanation", "")
                if strategy:
                    details.append(f"- {strategy}: {explanation}")

    return "\n".join(details) if details else "No specific recommendations available."

class Orchestrator:
    def __init__(self, slug: str, iteration: int):
        self.slug = slug
        self.iteration = iteration
        self.base_dir = _TASK_ROOT / slug
        self.outputs_dir = self.base_dir / _OUTPUTS_DIRNAME / str(iteration)

    @weave.op()
    def run(self, max_time_seconds: int | None = 6 * 3600) -> Tuple[bool, str]:

        # Phase 1: Starter Agent - Get task type and summary
        starter_suggestion_path = self.outputs_dir / "starter_suggestions.json"
        if starter_suggestion_path.exists():
            with open(starter_suggestion_path, "r") as f:
                starter_data = json.load(f)
        else:
            starter = StarterAgent(self.slug, self.iteration)
            starter.run()

            if starter_suggestion_path.exists():
                with open(starter_suggestion_path, "r") as f:
                    starter_data = json.load(f)
            else:
                raise RuntimeError("No starter suggestions found")

        # Phase 2: Researcher Agent - Generate research plan
        plan_path = self.outputs_dir / "plan.md"
        if not plan_path.exists():
            _run_researcher_once(self.slug, self.iteration, 1)
            if not plan_path.exists():
                raise RuntimeError("No plan found")

        # Phase 3: Model Recommender Agent - Get model-specific recommendations
        # (ModelRecommender uses plan.md internally via research_plan input)
        model_rec_path = self.outputs_dir / "model_recommendations.json"
        if model_rec_path.exists():
            with open(model_rec_path, "r") as f:
                model_recommendations = json.load(f)
        else:
            model_rec_agent = ModelRecommenderAgent(self.slug, self.iteration)
            model_recommendations = model_rec_agent.run()  # Uses default_models from config

            if model_rec_path.exists():
                with open(model_rec_path, "r") as f:
                    model_recommendations = json.load(f)
            else:
                raise RuntimeError("No model recommendations found")

        # Extract NOW and LATER recommendations for each model
        now_recommendations_all = {}
        later_recommendations_all = {}

        for model_name, recommendations in model_recommendations.items():
            now_recommendations_all[model_name] = _extract_now_recommendations(recommendations)
            later_recommendations_all[model_name] = _extract_later_recommendations(recommendations)
        
        # Persist NOW recommendations for future use
        now_rec_path = self.outputs_dir / "now_recommendations.json"
        with open(now_rec_path, "w") as f:
            json.dump(now_recommendations_all, f, indent=2)

        # Persist LATER recommendations for future use
        later_rec_path = self.outputs_dir / "later_recommendations.json"
        with open(later_rec_path, "w") as f:
            json.dump(later_recommendations_all, f, indent=2)

        # Phase 4: Baseline Developer Stage - Evaluate models with NOW recommendations
        baseline_results = {}

        # Run developer agents sequentially
        for idx, (model_name, now_recommendations) in enumerate(now_recommendations_all.items(), start=1):
            key = model_name  # Use model name as key
            dev_iter = f"{self.iteration}_{idx}"

            try:
                key, best_score, best_code, blacklisted_ideas = _run_developer_baseline(
                    self.slug,
                    dev_iter,
                    model_name,
                    now_recommendations,
                    key
                )
                baseline_results[key] = {
                    "model_name": key,
                    "best_score": best_score,
                    "best_code": best_code or "",
                    "blacklisted_ideas": blacklisted_ideas,
                    "now_recommendations": now_recommendations_all.get(key, {}),
                    "later_recommendations": later_recommendations_all.get(key, {})
                }
            except Exception as e:
                # Log error but continue with other models
                import logging
                logging.error(f"Failed to run developer for model {key}: {e}")
                continue

        if not baseline_results:
            raise RuntimeError("All developer baseline runs failed")

        # Persist baseline results
        baseline_path = self.outputs_dir / "baseline_results.json"
        with open(baseline_path, "w") as f:
            json.dump(baseline_results, f, indent=2)

        return True, str(baseline_path)
    