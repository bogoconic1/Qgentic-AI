from typing import Tuple
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
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
def _run_developer_baseline(slug: str, iteration_suffix: str, model_name: str, model_recommendations: dict, key: str):
    """Run a single baseline DeveloperAgent and return (key, best_score, best_code).

    Args:
        slug: Competition slug
        iteration_suffix: Iteration identifier (e.g., "1_1")
        model_name: Model name (e.g., "deberta-v3-large")
        model_recommendations: Full recommendations dict for this model
        key: Key for tracking results
    """
    # Format recommendations for developer
    formatted_recommendations = _format_recommendations_for_developer(model_recommendations)

    dev = DeveloperAgent(slug, iteration_suffix, model_name=model_name, model_recommendations=formatted_recommendations)
    best_score, best_code, blacklisted_ideas = dev.run(max_time_seconds=9000)
    return key, best_score, best_code, blacklisted_ideas

def _format_recommendations_for_developer(recommendations: dict) -> str:
    """Format model recommendations for DeveloperAgent.

    Formats the recommendations into a readable string that can be passed
    to DeveloperAgent as guidance. Includes ALL recommendations without filtering.
    """
    details = []

    # Preprocessing
    preprocessing = recommendations.get("preprocessing", {})
    if preprocessing:
        details.append("## Preprocessing Strategies")
        for category, strategies in preprocessing.items():
            if strategies and isinstance(strategies, list):
                details.append(f"\n### {category.replace('_', ' ').title()}")
                for item in strategies:  # ALL strategies, no limit
                    if isinstance(item, dict):
                        strategy = item.get("strategy", "")
                        explanation = item.get("explanation", "")
                        if strategy:
                            details.append(f"- {strategy}: {explanation}")

    # Loss function
    loss_fn = recommendations.get("loss_function", {})
    if loss_fn:
        details.append("\n## Loss Function")
        loss_name = loss_fn.get("loss_function", "")
        loss_exp = loss_fn.get("explanation", "")
        if loss_name:
            details.append(f"- Use {loss_name}: {loss_exp}")

    # Hyperparameters
    hyperparams = recommendations.get("hyperparameters", {})
    if hyperparams:
        details.append("\n## Hyperparameters")
        hp_list = hyperparams.get("hyperparameters", [])
        for item in hp_list:  # ALL hyperparameters, no limit
            if isinstance(item, dict):
                hp = item.get("hyperparameter", "")
                explanation = item.get("explanation", "")
                if hp:
                    details.append(f"- {hp}: {explanation}")

        # Architectures
        arch_list = hyperparams.get("architectures", [])
        if arch_list:
            details.append("\n### Architecture Recommendations")
            for item in arch_list:  # ALL architectures, no limit
                if isinstance(item, dict):
                    arch = item.get("architecture", "")
                    explanation = item.get("explanation", "")
                    if arch:
                        details.append(f"- {arch}: {explanation}")

    # Inference strategies
    inference = recommendations.get("inference_strategies", {})
    if inference:
        details.append("\n## Inference Strategies")
        strategies = inference.get("inference_strategies", [])
        for item in strategies:  # ALL strategies, no limit
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

        # Phase 4: Baseline Developer Stage - Evaluate models with recommendations
        baseline_results = {}
        tasks = []

        with ProcessPoolExecutor(max_workers=min(len(model_recommendations), 5)) as ex:
            for idx, (model_name, recommendations) in enumerate(model_recommendations.items(), start=1):
                key = model_name  # Use model name as key instead of model_1, model_2, etc.
                dev_iter = f"{self.iteration}_{idx}"
                fut = ex.submit(
                    _run_developer_baseline,
                    self.slug,
                    dev_iter,
                    model_name,
                    recommendations,
                    key
                )
                tasks.append(fut)

            for fut in as_completed(tasks):
                try:
                    key, best_score, best_code, blacklisted_ideas = fut.result()
                    baseline_results[key] = {
                        "model_name": key,
                        "best_score": best_score,
                        "best_code": best_code or "",
                        "blacklisted_ideas": blacklisted_ideas,
                        "recommendations": model_recommendations.get(key, {})
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
    