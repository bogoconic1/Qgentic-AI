from typing import Tuple
from pathlib import Path
import json
import os
import subprocess
import re
from queue import Queue

from agents.researcher import ResearcherAgent
from agents.developer import DeveloperAgent
from agents.ensembler import EnsemblerAgent
from agents.starter import StarterAgent
from agents.model_recommender import ModelRecommenderAgent
from project_config import get_config
from weave.trace.util import ThreadPoolExecutor
import weave
from utils.ensembler import move_best_code_to_ensemble_folder, recommend_ensemble_strategies


_CONFIG = get_config()
_RUNTIME_CFG = _CONFIG.get("runtime")
_PATH_CFG = _CONFIG.get("paths")
_TASK_ROOT = Path(_PATH_CFG.get("task_root"))
_OUTPUTS_DIRNAME = _PATH_CFG.get("outputs_dirname")
_DEFAULT_PARALLEL = int(_RUNTIME_CFG.get("researcher_parallel_runs"))

def _get_mig_uuids() -> list[str]:
    """Parse MIG device UUIDs from nvidia-smi -L output for all GPUs.

    Returns:
        List of MIG UUIDs from all GPUs (e.g., ["MIG-17331e1a-f2f0-500d-86f0-acf8289655ad", ...])
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

        # Parse output for MIG UUIDs from all GPUs
        # Example line: "  MIG 2g.20gb     Device  0: (UUID: MIG-17331e1a-f2f0-500d-86f0-acf8289655ad)"
        mig_uuids = []
        lines = result.stdout.strip().split("\n")

        for line in lines:
            # Parse MIG UUID from any line that contains both "MIG" and "UUID:"
            if "MIG" in line and "UUID:" in line:
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

def _calculate_max_parallel_workers_and_pools() -> tuple[int, list[str], str]:
    """Calculate max parallel workers and create GPU pool based on configuration.

    Returns:
        Tuple of (max_parallel_workers, gpu_pool, gpu_isolation_mode)
        - max_parallel_workers: Number of parallel workers
        - gpu_pool: List of GPU identifiers (MIG UUIDs or GPU IDs as strings)
        - gpu_isolation_mode: "mig", "multi-gpu", or "none"
    """
    enable_mig = _RUNTIME_CFG.get("enable_mig", False)
    enable_multi_gpu = _RUNTIME_CFG.get("enable_multi_gpu", False)

    # Sanity check: both MIG and multi-GPU cannot be enabled simultaneously
    if enable_mig and enable_multi_gpu:
        raise ValueError(
            "Conflicting GPU isolation modes: Cannot enable both MIG and multi-GPU simultaneously. "
            "Please set only one of: enable_mig or enable_multi_gpu to true in config.yaml"
        )

    if enable_mig:
        # MIG mode: Auto-detect MIG instances from all GPUs
        detected_mig_uuids = _get_mig_uuids()
        if len(detected_mig_uuids) > 0:
            return len(detected_mig_uuids), detected_mig_uuids, "mig"
        else:
            fallback = int(_RUNTIME_CFG.get("baseline_max_parallel_workers", 1))
            return fallback, [], "mig"

    elif enable_multi_gpu:
        # Multi-GPU mode: Auto-detect available GPUs
        available_gpus = _get_available_gpus()

        # Filter by allowed_gpu_ids if specified in config
        allowed_gpu_ids = _RUNTIME_CFG.get("allowed_gpu_ids", None)
        if allowed_gpu_ids is not None and len(allowed_gpu_ids) > 0:
            available_gpus = [gpu_id for gpu_id in available_gpus if gpu_id in allowed_gpu_ids]

        if len(available_gpus) > 0:
            gpu_pool = [str(gpu_id) for gpu_id in available_gpus]
            return len(available_gpus), gpu_pool, "multi-gpu"
        else:
            fallback = int(_RUNTIME_CFG.get("baseline_max_parallel_workers", 1))
            return fallback, [], "multi-gpu"

    else:
        # No GPU isolation: Use configured value
        max_workers = int(_RUNTIME_CFG.get("baseline_max_parallel_workers", 1))
        return max_workers, [], "none"


def _create_cpu_core_pool(max_parallel_workers: int) -> list[list[int]]:
    """Create CPU core pool for parallel execution with affinity.

    Args:
        max_parallel_workers: Number of parallel workers

    Returns:
        List of CPU core ranges, one per worker
    """
    enable_cpu_affinity = _RUNTIME_CFG.get("enable_cpu_affinity", False)

    if not enable_cpu_affinity or max_parallel_workers <= 1:
        return []

    total_cores = os.cpu_count() or 1
    cores_per_worker = total_cores // max_parallel_workers
    cpu_core_pool = []

    for i in range(max_parallel_workers):
        start_core = i * cores_per_worker
        # Last worker gets remaining cores
        end_core = total_cores if i == max_parallel_workers - 1 else (i + 1) * cores_per_worker
        cpu_core_pool.append(list(range(start_core, end_core)))

    return cpu_core_pool


def _ensure_conda_environments(num_workers: int) -> None:
    """Create isolated conda environments for parallel baseline execution.

    Args:
        num_workers: Number of conda environments to create (one per parallel worker)
    """
    # Get the base environment name (current active environment)
    base_env = os.environ.get('CONDA_DEFAULT_ENV', 'qgentic-ai')

    # Check if environments should be reset on each run
    reset_per_run = _RUNTIME_CFG.get("reset_conda_envs_per_run", True)

    if reset_per_run:
        print(f"Creating {num_workers} fresh conda environments for isolated execution (reset mode enabled)...")
        print(f"Base environment: {base_env}")
    else:
        print(f"Ensuring {num_workers} conda environments for isolated package installation (reuse mode enabled)...")
        print(f"Base environment: {base_env}")

    # Check which environments exist
    try:
        result = subprocess.run(
            ["conda", "env", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        existing_envs = result.stdout
    except Exception as e:
        print(f"Warning: Could not check conda environments: {e}")
        print("Skipping conda environment creation")
        return

    # Create or recreate environments based on config
    for i in range(1, num_workers + 1):
        env_name = f"qgentic-model-{i}"

        if reset_per_run:
            # Reset mode: Delete and recreate for clean slate
            if env_name in existing_envs:
                print(f"  Removing existing {env_name}...")
                try:
                    subprocess.run(
                        ["conda", "env", "remove", "-n", env_name, "-y"],
                        capture_output=True,
                        text=True,
                        check=True
                    )
                except subprocess.CalledProcessError as e:
                    print(f"  Warning: Failed to remove {env_name}: {e}")

            print(f"  Creating {env_name} (cloning from {base_env})...")
            try:
                subprocess.run(
                    ["conda", "create", "--name", env_name, "--clone", base_env, "-y", "--quiet"],
                    check=True,
                    capture_output=True,
                    text=True
                )
                print(f"  ✓ Created {env_name}")
            except subprocess.CalledProcessError as e:
                print(f"  Warning: Failed to create {env_name}: {e}")
                print(f"  stderr: {e.stderr}")
        else:
            # Reuse mode: Only create if missing
            if env_name in existing_envs:
                print(f"  ✓ {env_name} already exists (reusing)")
            else:
                print(f"  Creating {env_name} (cloning from {base_env})...")
                try:
                    subprocess.run(
                        ["conda", "create", "--name", env_name, "--clone", base_env, "-y", "--quiet"],
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    print(f"  ✓ Created {env_name}")
                except subprocess.CalledProcessError as e:
                    print(f"  Warning: Failed to create {env_name}: {e}")
                    print(f"  stderr: {e.stderr}")

    print("Conda environment setup complete!")
    print()

@weave.op()
def _run_researcher_once(slug: str, iteration: int, run_id: int) -> tuple[int, str, int]:
    # Calculate max_parallel_workers and resource pools for AB test parallelization
    max_parallel_workers, gpu_pool, _ = _calculate_max_parallel_workers_and_pools()
    cpu_core_pool = _create_cpu_core_pool(max_parallel_workers)

    print(f"Researcher Agent: max_parallel_workers={max_parallel_workers}, CPU pools={len(cpu_core_pool)}, GPU pools={len(gpu_pool)}")

    agent = ResearcherAgent(slug, iteration, run_id=run_id, max_parallel_workers=max_parallel_workers, cpu_core_pool=cpu_core_pool, gpu_pool=gpu_pool)
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
def _run_developer_baseline(slug: str, iteration_suffix: str, model_name: str, now_recommendations: dict, later_recommendations: dict, key: str, external_data_listing: str, plan_content: str, cpu_core_pool: Queue | None = None, gpu_pool: Queue | None = None, gpu_isolation_mode: str = "none", conda_env: str | None = None):
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
        conda_env: Conda environment name to use for code execution (None = use current env)
    """
    # Acquire resources from pools (blocks until available)
    cpu_core_range = cpu_core_pool.get() if cpu_core_pool else None
    gpu_identifier = gpu_pool.get() if gpu_pool else None

    try:
        # Format NOW recommendations for developer
        formatted_recommendations = _format_recommendations_for_developer(now_recommendations)

        baseline_time_limit = _RUNTIME_CFG["baseline_time_limit"]
        dev = DeveloperAgent(
            slug,
            iteration_suffix,
            model_name=model_name,
            model_recommendations=formatted_recommendations,
            later_recommendations=later_recommendations,
            external_data_listing=external_data_listing,
            plan_content=plan_content,
            cpu_core_range=cpu_core_range,
            gpu_identifier=gpu_identifier,
            gpu_isolation_mode=gpu_isolation_mode,
            conda_env=conda_env
        )
        best_score, best_code_file, blacklisted_ideas, successful_ideas = dev.run(max_time_seconds=baseline_time_limit)
        return key, best_score, best_code_file, blacklisted_ideas, successful_ideas
    finally:
        # Return resources to pools for next task
        if cpu_core_pool and cpu_core_range is not None:
            cpu_core_pool.put(cpu_core_range)
        if gpu_pool and gpu_identifier is not None:
            gpu_pool.put(gpu_identifier)

@weave.op()
def _run_ensembler_single(slug: str, iteration: int, strategy_index: int, strategy: dict, baseline_metadata: dict, external_data_listing: str, plan_content: str, cpu_core_pool: Queue | None = None, gpu_pool: Queue | None = None, gpu_isolation_mode: str = "none", conda_env: str | None = None):
    """Run a single EnsemblerAgent for one ensemble strategy and return results.

    Args:
        slug: Competition slug
        iteration: Iteration number
        strategy_index: Index of this strategy (1-8)
        strategy: Strategy dict with "strategy" and "models_needed" keys
        baseline_metadata: Metadata from ensemble_metadata.json
        cpu_core_pool: Queue of CPU core ranges to grab from (None = no affinity)
        gpu_pool: Queue of GPU identifiers (MIG UUIDs or GPU IDs) to grab from (None = use GPU 0)
        gpu_isolation_mode: Type of GPU isolation ("mig", "multi-gpu", or "none")
        conda_env: Conda environment name to use for code execution (None = use current env)
    """
    # Acquire resources from pools (blocks until available)
    cpu_core_range = cpu_core_pool.get() if cpu_core_pool else None
    gpu_identifier = gpu_pool.get() if gpu_pool else None

    try:
        ensemble_time_limit = _RUNTIME_CFG["ensemble_time_limit"]
        agent = EnsemblerAgent(
            slug=slug,
            iteration=iteration,
            strategy_index=strategy_index,
            strategy=strategy,
            baseline_metadata=baseline_metadata,
            external_data_listing=external_data_listing,
            plan_content=plan_content,
            cpu_core_range=cpu_core_range,
            gpu_identifier=gpu_identifier,
            gpu_isolation_mode=gpu_isolation_mode,
            conda_env=conda_env
        )
        best_score, best_code_file, blacklisted_ideas, successful_ideas = agent.run(max_time_seconds=ensemble_time_limit)
        strategy_key = f"EnsembleStrategy{strategy_index}"
        return strategy_key, best_score, best_code_file, blacklisted_ideas, successful_ideas
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

    def _get_external_data_listing(self) -> str:
        """Get directory listing of external_data_* folders in outputs/<iteration>."""
        from tools.helpers import _build_directory_listing

        if not self.outputs_dir.exists():
            return "No external data directories found."

        external_data_dirs = [d for d in self.outputs_dir.iterdir() if d.is_dir() and d.name.startswith("external_data_")]

        if not external_data_dirs:
            return "No external data directories found."

        lines = []
        for ext_dir in sorted(external_data_dirs):
            lines.append(f"\n{ext_dir.name}/")
            dir_listing = _build_directory_listing(str(ext_dir))
            lines.append(dir_listing)

        return "\n".join(lines)

    def _get_plan_content(self) -> str:
        """Get content of plan.md if it exists."""
        plan_path = self.outputs_dir / "plan.md"
        if plan_path.exists():
            with open(plan_path, "r") as f:
                return f.read()
        return "No plan.md found."

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

        # Get parallel execution configuration using shared helper functions
        max_parallel_workers, gpu_pool_list, gpu_isolation_mode = _calculate_max_parallel_workers_and_pools()
        cpu_core_pool_list = _create_cpu_core_pool(max_parallel_workers)

        # Print configuration summary
        if gpu_isolation_mode == "mig":
            if len(gpu_pool_list) > 0:
                print(f"MIG enabled: Auto-detected {max_parallel_workers} MIG instances")
            else:
                print("WARNING: enable_mig=true but no MIG instances detected")
                print("Falling back to baseline_max_parallel_workers from config")
        elif gpu_isolation_mode == "multi-gpu":
            if len(gpu_pool_list) > 0:
                print(f"Multi-GPU enabled: Auto-detected {max_parallel_workers} GPUs: {gpu_pool_list}")
            else:
                print("WARNING: enable_multi_gpu=true but no GPUs detected")
                print("Falling back to baseline_max_parallel_workers from config")
        else:
            print(f"GPU isolation disabled: Using baseline_max_parallel_workers={max_parallel_workers}")

        # Create resource queues for dynamic allocation (baseline developer uses Queues for flexibility)
        num_models = len(now_recommendations_all)

        # CPU core pool (convert list to Queue for dynamic acquisition/release)
        cpu_core_pool = None
        if len(cpu_core_pool_list) > 0 and num_models > 1:
            cpu_core_pool = Queue()
            for core_range in cpu_core_pool_list:
                cpu_core_pool.put(core_range)
            print(f"CPU core pool created with {max_parallel_workers} core ranges")

        # GPU pool (convert list to Queue for dynamic acquisition/release)
        gpu_pool = None
        if num_models > 1 and len(gpu_pool_list) > 0:
            gpu_pool = Queue()
            for gpu_id in gpu_pool_list:
                gpu_pool.put(gpu_id)
            if gpu_isolation_mode == "mig":
                print(f"MIG pool created with {len(gpu_pool_list)} instances")
            elif gpu_isolation_mode == "multi-gpu":
                print(f"GPU pool created with {len(gpu_pool_list)} GPUs")

        # Prepare tasks for parallel execution (no pre-assignment of resources)
        # Load existing baseline results if available for selective reruns
        existing_baseline_results = {}
        baseline_path = self.outputs_dir / "baseline_results.json"
        if baseline_path.exists():
            try:
                with open(baseline_path, 'r') as f:
                    existing_baseline_results = json.load(f)
                print(f"Loaded existing baseline results with {len(existing_baseline_results)} models")
            except Exception as e:
                print(f"Warning: Could not load existing baseline results: {e}")

        # Read external data listing and plan content (shared across all models)
        external_data_listing = self._get_external_data_listing()
        plan_content = self._get_plan_content()
        print(f"External data listing: {len(external_data_listing)} chars")
        print(f"Plan content: {len(plan_content)} chars")

        if not os.path.exists(self.outputs_dir / "baseline_results.json") or len(existing_baseline_results) < len(now_recommendations_all):
            # Ensure conda environments exist for isolated package installation
            _ensure_conda_environments(max_parallel_workers)

            tasks = []
            for idx, (model_name, now_recommendations) in enumerate(now_recommendations_all.items(), start=1):
                key = model_name
                dev_iter = f"{self.iteration}_{idx}"
                later_recs = later_recommendations_all.get(model_name, {})

                # Assign isolated conda environment for this model
                conda_env = f"qgentic-model-{idx}"

                # Skip if this model already has results (unless you want to force rerun)
                # To force rerun specific models, delete them from baseline_results.json
                if key in existing_baseline_results:
                    print(f"Skipping {key} (iteration {dev_iter}): already completed")
                    continue

                print(f"Queueing {key} (iteration {dev_iter}) for execution with conda env: {conda_env}")
                tasks.append((
                    self.slug,
                    dev_iter,
                    model_name,
                    now_recommendations,
                    later_recs,
                    key,
                    external_data_listing,
                    plan_content,
                    cpu_core_pool,
                    gpu_pool,
                    gpu_isolation_mode,
                    conda_env
                ))

            # Start with existing results (for incremental updates)
            baseline_results = existing_baseline_results.copy()

            # Run developer agents in parallel using ThreadPoolExecutor
            if tasks:
                print(f"Running {len(tasks)} baseline task(s) in parallel")
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
                            # Incrementally persist after each completion
                            baseline_path = self.outputs_dir / "baseline_results.json"
                            with open(baseline_path, "w") as f:
                                json.dump(baseline_results, f, indent=2)
                        except Exception as e:
                            print(f"Error in baseline task: {e}")
                            continue
            else:
                print("No new baseline tasks to run (all models already completed)")

            # Persist baseline results (final)
            if baseline_results:
                baseline_path = self.outputs_dir / "baseline_results.json"
                with open(baseline_path, "w") as f:
                    json.dump(baseline_results, f, indent=2)
            else:
                raise RuntimeError("All developer baseline runs failed")

        # Phase 5: Ensemble Phase - Combine baseline models using ensemble strategies
        print("\n" + "="*80)
        print("PHASE 5: ENSEMBLE PHASE")
        print("="*80)

        # Check if ensemble metadata already exists with strategies
        ensemble_folder = self.outputs_dir / "ensemble"
        ensemble_metadata_path = ensemble_folder / "ensemble_metadata.json"

        if ensemble_metadata_path.exists():
            with open(ensemble_metadata_path, "r") as f:
                ensemble_metadata = json.load(f)

            if "strategies" in ensemble_metadata:
                print("Ensemble metadata with strategies already exists, skipping generation...")
                ensemble_strategies = ensemble_metadata["strategies"]
                print(f"Loaded {len(ensemble_strategies)} existing ensemble strategies")
            else:
                print("Ensemble metadata exists but no strategies found, regenerating...")
                # Step 1: Move best code to ensemble folder
                print("Moving best baseline code to ensemble folder...")
                ensemble_folder = move_best_code_to_ensemble_folder(self.slug, self.iteration)
                print(f"Ensemble folder created: {ensemble_folder}")

                # Step 2: Generate ensemble strategies
                print("Generating ensemble strategies...")
                ensemble_strategies = recommend_ensemble_strategies(self.slug, self.iteration)
                print(f"Generated {len(ensemble_strategies)} ensemble strategies")

                # Reload metadata after generation
                with open(ensemble_metadata_path, "r") as f:
                    ensemble_metadata = json.load(f)
        else:
            print("No ensemble metadata found, creating from scratch...")
            # Step 1: Move best code to ensemble folder
            print("Moving best baseline code to ensemble folder...")
            ensemble_folder = move_best_code_to_ensemble_folder(self.slug, self.iteration)
            print(f"Ensemble folder created: {ensemble_folder}")

            # Step 2: Generate ensemble strategies
            print("Generating ensemble strategies...")
            ensemble_strategies = recommend_ensemble_strategies(self.slug, self.iteration)
            print(f"Generated {len(ensemble_strategies)} ensemble strategies")

            # Load ensemble metadata with strategies
            with open(ensemble_metadata_path, "r") as f:
                ensemble_metadata = json.load(f)

        # Step 3: Re-initialize conda environments if configured
        if _RUNTIME_CFG.get("reset_conda_envs_per_run", False):
            print("Re-initializing conda environments for ensemble phase...")
            num_strategies = len(ensemble_strategies)
            num_ensemble_workers = min(num_strategies, max_parallel_workers)
            _ensure_conda_environments(num_ensemble_workers)

        # Step 4: Prepare ensemble tasks for parallel execution
        ensemble_tasks = []
        for strategy_index, strategy in enumerate(ensemble_strategies, start=1):
            conda_env = f"qgentic-model-{strategy_index}"
            print(f"Queueing EnsembleStrategy{strategy_index} with conda env: {conda_env}")
            ensemble_tasks.append((
                self.slug,
                self.iteration,
                strategy_index,
                strategy,
                ensemble_metadata,
                external_data_listing,
                plan_content,
                cpu_core_pool,
                gpu_pool,
                gpu_isolation_mode,
                conda_env
            ))

        # Step 5: Run ensemble agents in parallel
        ensemble_results = {}
        if ensemble_tasks:
            print(f"Running {len(ensemble_tasks)} ensemble strategies in parallel")
            with ThreadPoolExecutor(max_workers=max_parallel_workers) as executor:
                futures = [
                    executor.submit(_run_ensembler_single, *task_args)
                    for task_args in ensemble_tasks
                ]

                for future in futures:
                    try:
                        strategy_key, best_score, best_code_file, blacklisted_ideas, successful_ideas = future.result()
                        ensemble_results[strategy_key] = {
                            "strategy_name": strategy_key,
                            "best_score": best_score,
                            "best_code_file": best_code_file or "",
                            "blacklisted_ideas": blacklisted_ideas,
                            "successful_ideas": successful_ideas
                        }
                        # Incrementally persist after each completion
                        ensemble_results_path = self.outputs_dir / "ensemble_results.json"
                        with open(ensemble_results_path, "w") as f:
                            json.dump(ensemble_results, f, indent=2)
                    except Exception as e:
                        print(f"Error in ensemble task: {e}")
                        continue

            # Persist ensemble results (final)
            if ensemble_results:
                ensemble_results_path = self.outputs_dir / "ensemble_results.json"
                with open(ensemble_results_path, "w") as f:
                    json.dump(ensemble_results, f, indent=2)
                print(f"Ensemble results saved to {ensemble_results_path}")
            else:
                print("Warning: All ensemble strategies failed")

        # Return baseline results path
        baseline_path = self.outputs_dir / "baseline_results.json"
        return True, str(baseline_path)
    