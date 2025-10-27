from typing import Tuple
from pathlib import Path
import json
import os
import subprocess
import re
from queue import Queue

from agents.researcher import ResearcherAgent
from agents.developer import DeveloperAgent
from agents.starter import StarterAgent
from agents.model_recommender import ModelRecommenderAgent
from project_config import get_config
from weave.trace.util import ThreadPoolExecutor
import weave


_CONFIG = get_config()
_RUNTIME_CFG = _CONFIG.get("runtime")
_PATH_CFG = _CONFIG.get("paths")
_TASK_ROOT = Path(_PATH_CFG.get("task_root"))
_OUTPUTS_DIRNAME = _PATH_CFG.get("outputs_dirname")
_DEFAULT_PARALLEL = int(_RUNTIME_CFG.get("researcher_parallel_runs"))

def _get_mig_uuids(gpu_id: int = 0) -> list[str]:
    """Parse MIG device UUIDs from nvidia-smi -L output.

    Returns:
        List of MIG UUIDs (e.g., ["MIG-17331e1a-f2f0-500d-86f0-acf8289655ad", ...])
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            return []

        # Parse output for MIG UUIDs
        # Example line: "  MIG 2g.20gb     Device  0: (UUID: MIG-17331e1a-f2f0-500d-86f0-acf8289655ad)"
        mig_uuids = []
        lines = result.stdout.strip().split("\n")
        in_target_gpu = False

        for line in lines:
            # Check if this is the target GPU line
            if line.startswith(f"GPU {gpu_id}:"):
                in_target_gpu = True
                continue

            # Check if we've moved to next GPU
            if line.startswith("GPU ") and not line.startswith(f"GPU {gpu_id}:"):
                in_target_gpu = False
                continue

            # Parse MIG UUID if we're in the target GPU section
            if in_target_gpu and "MIG" in line and "UUID:" in line:
                match = re.search(r'UUID:\s+(MIG-[a-f0-9\-]+)', line)
                if match:
                    mig_uuids.append(match.group(1))

        return mig_uuids
    except Exception:
        return []

def _get_available_gpus() -> list[int]:
    """Detect available CUDA GPUs using nvidia-smi.

    Returns:
        List of GPU IDs (e.g., [0, 1, 2, 3] for 4 GPUs)
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            gpu_ids = [int(line.strip()) for line in result.stdout.strip().split("\n") if line.strip()]
            return gpu_ids
        return []
    except Exception:
        return []

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
def _run_developer_baseline(slug: str, iteration_suffix: str, model_name: str, now_recommendations: dict, later_recommendations: dict, key: str, cpu_core_pool: Queue | None = None, gpu_pool: Queue | None = None, gpu_isolation_mode: str = "none"):
    """Run a single baseline DeveloperAgent and return results.

    Args:
        slug: Competition slug
        iteration_suffix: Iteration identifier (e.g., "1_1")
        model_name: Model name (e.g., "deberta-v3-large")
        now_recommendations: NOW-only recommendations dict for this model
        later_recommendations: LATER-only recommendations dict for this model
        key: Key for tracking results
        cpu_core_pool: Queue of CPU core ranges to grab from (None = no affinity)
        gpu_pool: Queue of GPU identifiers (MIG UUIDs or GPU IDs) to grab from (None = use GPU 0)
        gpu_isolation_mode: Type of GPU isolation ("mig", "multi-gpu", or "none")
    """
    # Acquire resources from pools (blocks until available)
    cpu_core_range = cpu_core_pool.get() if cpu_core_pool else None
    gpu_identifier = gpu_pool.get() if gpu_pool else None

    try:
        # Format NOW recommendations for developer
        formatted_recommendations = _format_recommendations_for_developer(now_recommendations)

        dev = DeveloperAgent(
            slug,
            iteration_suffix,
            model_name=model_name,
            model_recommendations=formatted_recommendations,
            later_recommendations=later_recommendations,
            cpu_core_range=cpu_core_range,
            gpu_identifier=gpu_identifier,
            gpu_isolation_mode=gpu_isolation_mode
        )
        best_score, best_code_file, blacklisted_ideas, successful_ideas = dev.run(max_time_seconds=10800)  # 3 hours
        return key, best_score, best_code_file, blacklisted_ideas, successful_ideas
    finally:
        # Return resources to pools for next task
        if cpu_core_pool and cpu_core_range is not None:
            cpu_core_pool.put(cpu_core_range)
        if gpu_pool and gpu_identifier is not None:
            gpu_pool.put(gpu_identifier)

def _extract_now_recommendations(recommendations: dict) -> dict:
    """Extract only MUST_HAVE recommendations from model recommendations.

    Filters out citations and est_runtime_minutes_gpu fields.
    Returns a dict with the same structure but only MUST_HAVE sections.
    """
    now_only = {}

    # Preprocessing - each category has MUST_HAVE/NICE_TO_HAVE structure
    preprocessing = recommendations.get("preprocessing", {})
    if preprocessing:
        now_only["preprocessing"] = {}
        for category, content in preprocessing.items():
            if isinstance(content, dict) and "MUST_HAVE" in content:
                now_items = content["MUST_HAVE"]
                if isinstance(now_items, list):
                    now_only["preprocessing"][category] = {
                        "MUST_HAVE": now_items
                    }

    # Loss function - MUST_HAVE is an object, NICE_TO_HAVE is an array
    loss_fn = recommendations.get("loss_function", {})
    if loss_fn and "MUST_HAVE" in loss_fn:
        now_only["loss_function"] = {
            "MUST_HAVE": loss_fn["MUST_HAVE"]
        }

    # Hyperparameters - MUST_HAVE has hyperparameters and architectures arrays
    hyperparams = recommendations.get("hyperparameters", {})
    if hyperparams and "MUST_HAVE" in hyperparams:
        now_section = hyperparams["MUST_HAVE"]
        cleaned_now = {}
        if "hyperparameters" in now_section:
            cleaned_now["hyperparameters"] = now_section["hyperparameters"]
        if "architectures" in now_section:
            cleaned_now["architectures"] = now_section["architectures"]
        now_only["hyperparameters"] = {"MUST_HAVE": cleaned_now}

    # Inference strategies - MUST_HAVE has inference_strategies array
    inference = recommendations.get("inference_strategies", {})
    if inference and "MUST_HAVE" in inference:
        now_section = inference["MUST_HAVE"]
        if "inference_strategies" in now_section:
            now_only["inference_strategies"] = {
                "MUST_HAVE": {
                    "inference_strategies": now_section["inference_strategies"]
                }
            }

    return now_only

def _extract_later_recommendations(recommendations: dict) -> dict:
    """Extract only NICE_TO_HAVE recommendations from model recommendations.

    Filters out citations and est_runtime_minutes_gpu fields.
    Returns a dict with the same structure but only NICE_TO_HAVE sections.
    """
    later_only = {}

    # Preprocessing - each category has MUST_HAVE/NICE_TO_HAVE structure
    preprocessing = recommendations.get("preprocessing", {})
    if preprocessing:
        later_only["preprocessing"] = {}
        for category, content in preprocessing.items():
            if isinstance(content, dict) and "NICE_TO_HAVE" in content:
                later_items = content["NICE_TO_HAVE"]
                if isinstance(later_items, list):
                    later_only["preprocessing"][category] = {
                        "NICE_TO_HAVE": later_items
                    }

    # Loss function - NICE_TO_HAVE is an array
    loss_fn = recommendations.get("loss_function", {})
    if loss_fn and "NICE_TO_HAVE" in loss_fn:
        later_only["loss_function"] = {
            "NICE_TO_HAVE": loss_fn["NICE_TO_HAVE"]
        }

    # Hyperparameters - NICE_TO_HAVE has hyperparameters and architectures arrays
    hyperparams = recommendations.get("hyperparameters", {})
    if hyperparams and "NICE_TO_HAVE" in hyperparams:
        later_section = hyperparams["NICE_TO_HAVE"]
        cleaned_later = {}
        if "hyperparameters" in later_section:
            cleaned_later["hyperparameters"] = later_section["hyperparameters"]
        if "architectures" in later_section:
            cleaned_later["architectures"] = later_section["architectures"]
        later_only["hyperparameters"] = {"NICE_TO_HAVE": cleaned_later}

    # Inference strategies - NICE_TO_HAVE has inference_strategies array
    inference = recommendations.get("inference_strategies", {})
    if inference and "NICE_TO_HAVE" in inference:
        later_section = inference["NICE_TO_HAVE"]
        if "inference_strategies" in later_section:
            later_only["inference_strategies"] = {
                "NICE_TO_HAVE": {
                    "inference_strategies": later_section["inference_strategies"]
                }
            }

    return later_only

def _format_recommendations_for_developer(recommendations: dict) -> str:
    """Format model recommendations for DeveloperAgent.

    Formats the MUST_HAVE recommendations into a readable string that can be passed
    to DeveloperAgent as guidance.
    """
    details = []

    # Fold split strategy
    fold_split = recommendations.get("fold_split_strategy", {})
    if fold_split:
        strategy = fold_split.get("strategy", "")
        if strategy:
            details.append("## Cross-Validation Strategy")
            details.append(f"- Use {strategy}")

    # Preprocessing
    preprocessing = recommendations.get("preprocessing", {})
    if preprocessing:
        details.append("## Preprocessing Strategies")
        for category, content in preprocessing.items():
            if isinstance(content, dict) and "MUST_HAVE" in content:
                now_items = content["MUST_HAVE"]
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
    if loss_fn and "MUST_HAVE" in loss_fn:
        details.append("\n## Loss Function")
        now_loss = loss_fn["MUST_HAVE"]
        loss_name = now_loss.get("loss_function", "")
        loss_exp = now_loss.get("explanation", "")
        if loss_name:
            details.append(f"- Use {loss_name}: {loss_exp}")

    # Hyperparameters
    hyperparams = recommendations.get("hyperparameters", {})
    if hyperparams and "MUST_HAVE" in hyperparams:
        now_section = hyperparams["MUST_HAVE"]

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
    if inference and "MUST_HAVE" in inference:
        now_section = inference["MUST_HAVE"]
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
        # First dynamically selects suitable models, then generates recommendations for each
        model_rec_path = self.outputs_dir / "model_recommendations.json"
        if model_rec_path.exists():
            with open(model_rec_path, "r") as f:
                model_recommendations = json.load(f)
        else:
            model_rec_agent = ModelRecommenderAgent(self.slug, self.iteration)
            model_recommendations = model_rec_agent.run(use_dynamic_selection=True)

            if model_rec_path.exists():
                with open(model_rec_path, "r") as f:
                    model_recommendations = json.load(f)
            else:
                raise RuntimeError("No model recommendations found")

        # Load fold split strategy (single strategy for all models)
        fold_split_strategy_path = self.outputs_dir / "fold_split_strategy.json"
        fold_split_strategy = {}
        if fold_split_strategy_path.exists():
            try:
                with open(fold_split_strategy_path, "r") as f:
                    fold_split_strategy = json.load(f)
            except Exception:
                pass

        # Extract NOW and LATER recommendations for each model
        now_recommendations_all = {}
        later_recommendations_all = {}

        for model_name, recommendations in model_recommendations.items():
            now_recs = _extract_now_recommendations(recommendations)
            # Add the fold split strategy to each model's recommendations
            if fold_split_strategy:
                now_recs["fold_split_strategy"] = fold_split_strategy
            now_recommendations_all[model_name] = now_recs
            later_recommendations_all[model_name] = _extract_later_recommendations(recommendations)
        
        # Persist NOW recommendations for future use
        now_rec_path = self.outputs_dir / "now_recommendations.json"
        with open(now_rec_path, "w") as f:
            json.dump(now_recommendations_all, f, indent=2)

        # Persist LATER recommendations for future use
        later_rec_path = self.outputs_dir / "later_recommendations.json"
        with open(later_rec_path, "w") as f:
            json.dump(later_recommendations_all, f, indent=2)

        # Phase 4: Baseline Developer Stage - Evaluate models in parallel with NOW recommendations
        baseline_results = {}

        # Get parallel execution configuration
        enable_cpu_affinity = _RUNTIME_CFG.get("enable_cpu_affinity", False)
        enable_mig = _RUNTIME_CFG.get("enable_mig", False)
        enable_multi_gpu = _RUNTIME_CFG.get("enable_multi_gpu", False)

        # Sanity check: both MIG and multi-GPU cannot be enabled simultaneously
        if enable_mig and enable_multi_gpu:
            raise ValueError(
                "Conflicting GPU isolation modes: Cannot enable both MIG and multi-GPU simultaneously. "
                "Please set only one of: enable_mig or enable_multi_gpu to true in config.yaml"
            )

        # Determine max_parallel_workers and GPU pool based on isolation mode
        gpu_pool = None
        gpu_isolation_mode = "none"
        detected_mig_uuids = []
        available_gpus = []

        if enable_mig:
            # MIG mode: Auto-detect MIG instances
            gpu_isolation_mode = "mig"
            detected_mig_uuids = _get_mig_uuids(gpu_id=0)
            if len(detected_mig_uuids) > 0:
                max_parallel_workers = len(detected_mig_uuids)
                print(f"MIG enabled: Auto-detected {max_parallel_workers} MIG instances")
            else:
                print("WARNING: enable_mig=true but no MIG instances detected")
                print("Falling back to baseline_max_parallel_workers from config")
                max_parallel_workers = int(_RUNTIME_CFG.get("baseline_max_parallel_workers", 1))

        elif enable_multi_gpu:
            # Multi-GPU mode: Auto-detect available GPUs
            gpu_isolation_mode = "multi-gpu"
            available_gpus = _get_available_gpus()
            if len(available_gpus) > 0:
                max_parallel_workers = len(available_gpus)
                print(f"Multi-GPU enabled: Auto-detected {max_parallel_workers} GPUs: {available_gpus}")
            else:
                print("WARNING: enable_multi_gpu=true but no GPUs detected")
                print("Falling back to baseline_max_parallel_workers from config")
                max_parallel_workers = int(_RUNTIME_CFG.get("baseline_max_parallel_workers", 1))

        else:
            # No GPU isolation: Use configured value
            max_parallel_workers = int(_RUNTIME_CFG.get("baseline_max_parallel_workers", 1))
            print(f"GPU isolation disabled: Using baseline_max_parallel_workers={max_parallel_workers}")

        # Create resource pools for dynamic allocation
        total_cores = os.cpu_count() or 1
        num_models = len(now_recommendations_all)

        # CPU core pool
        cpu_core_pool = None
        if enable_cpu_affinity and num_models > 1:
            cpu_core_pool = Queue()
            cores_per_worker = total_cores // max_parallel_workers
            for i in range(max_parallel_workers):
                start_core = i * cores_per_worker
                # Last worker gets remaining cores
                end_core = total_cores if i == max_parallel_workers - 1 else (i + 1) * cores_per_worker
                cpu_core_pool.put(list(range(start_core, end_core)))
            print(f"CPU core pool created with {max_parallel_workers} core ranges")

        # GPU pool (unified for both MIG and multi-GPU)
        if num_models > 1:
            if enable_mig and len(detected_mig_uuids) > 0:
                gpu_pool = Queue()
                for mig_uuid in detected_mig_uuids:
                    gpu_pool.put(mig_uuid)
                print(f"MIG pool created with {len(detected_mig_uuids)} instances")
            elif enable_multi_gpu and len(available_gpus) > 0:
                gpu_pool = Queue()
                for gpu_id in available_gpus:
                    gpu_pool.put(str(gpu_id))  # Store as string for consistency
                print(f"GPU pool created with {len(available_gpus)} GPUs")

        # Prepare tasks for parallel execution (no pre-assignment of resources)
        tasks = []
        for idx, (model_name, now_recommendations) in enumerate(now_recommendations_all.items(), start=1):
            key = model_name
            dev_iter = f"{self.iteration}_{idx}"
            later_recs = later_recommendations_all.get(model_name, {})

            tasks.append((
                self.slug,
                dev_iter,
                model_name,
                now_recommendations,
                later_recs,
                key,
                cpu_core_pool,
                gpu_pool,
                gpu_isolation_mode
            ))

        # Run developer agents in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_parallel_workers) as executor:
            futures = [
                executor.submit(_run_developer_baseline, *task_args)
                for task_args in tasks
            ]

            for future in futures:
                try:
                    result_key, best_score, best_code_file, blacklisted_ideas, successful_ideas = future.result()
                    baseline_results[result_key] = {
                        "model_name": result_key,
                        "best_score": best_score,
                        "best_code_file": best_code_file or "",
                        "blacklisted_ideas": blacklisted_ideas,
                        "successful_ideas": successful_ideas,
                        "fold_split_strategy": fold_split_strategy,
                        "now_recommendations": now_recommendations_all.get(result_key, {}),
                        "later_recommendations": later_recommendations_all.get(result_key, {})
                    }
                    baseline_path = self.outputs_dir / "baseline_results.json"
                    with open(baseline_path, "w") as f:
                        json.dump(baseline_results, f, indent=2)
                except Exception:
                    continue

        # Persist baseline results
        if baseline_results:
            baseline_path = self.outputs_dir / "baseline_results.json"
            with open(baseline_path, "w") as f:
                json.dump(baseline_results, f, indent=2)
        else:
            raise RuntimeError("All developer baseline runs failed")

        return True, str(baseline_path)
    