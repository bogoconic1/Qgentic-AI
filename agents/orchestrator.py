from typing import Tuple
from pathlib import Path
from datetime import datetime
import json
import os
import shutil
import subprocess
import re
from queue import Queue

from agents.researcher import ResearcherAgent
from agents.developer import DeveloperAgent
from agents.starter import StarterAgent
from agents.model_recommender import ModelRecommenderAgent
from project_config import get_config, get_instructions
from tools.helpers import _build_directory_listing
from utils.checkpoint import create_db as _create_checkpoint_db, delete_checkpoints_after
from weave.trace.util import ThreadPoolExecutor
import weave


_CONFIG = get_config()
_RUNTIME_CFG = _CONFIG.get("runtime")
_PATH_CFG = _CONFIG.get("paths")
_DEVELOPER_CFG = _CONFIG.get("developer")
_TASK_ROOT = Path(_PATH_CFG.get("task_root"))
_OUTPUTS_DIRNAME = _PATH_CFG.get("outputs_dirname")
_HITL_SOTA = bool(_DEVELOPER_CFG.get("hitl_sota"))

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

        # Example line: "  MIG 2g.20gb     Device  0: (UUID: MIG-17331e1a-f2f0-500d-86f0-acf8289655ad)"
        mig_uuids = []
        lines = result.stdout.strip().split("\n")

        for line in lines:
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
        detected_mig_uuids = _get_mig_uuids()
        if len(detected_mig_uuids) > 0:
            return len(detected_mig_uuids), detected_mig_uuids, "mig"
        else:
            fallback = int(_RUNTIME_CFG.get("baseline_max_parallel_workers", 1))
            return fallback, [], "mig"

    elif enable_multi_gpu:
        available_gpus = _get_available_gpus()

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
    # Get conda executable path (resolves "conda: command not found" in subprocess)
    conda_exe = os.environ.get('CONDA_EXE', 'conda')

    base_env = os.environ.get('CONDA_DEFAULT_ENV', 'qgentic-ai')

    reset_per_run = _RUNTIME_CFG.get("reset_conda_envs_per_run", True)

    if reset_per_run:
        print(f"Creating {num_workers} fresh conda environments for isolated execution (reset mode enabled)...")
        print(f"Base environment: {base_env}")
    else:
        print(f"Ensuring {num_workers} conda environments for isolated package installation (reuse mode enabled)...")
        print(f"Base environment: {base_env}")

    result = subprocess.run(
        [conda_exe, "env", "list"],
        capture_output=True,
        text=True,
        check=True
    )
    
    existing_envs = result.stdout

    for i in range(1, num_workers + 1):
        env_name = f"qgentic-model-{i}"

        if reset_per_run:
            # Reset mode: Delete and recreate for clean slate
            if env_name in existing_envs:
                print(f"  Removing existing {env_name}...")
                subprocess.run(
                    [conda_exe, "env", "remove", "-n", env_name, "-y"],
                    capture_output=True,
                    text=True,
                    check=True
                )

            print(f"  Creating {env_name} (cloning from {base_env})...")
            subprocess.run(
                [conda_exe, "create", "--name", env_name, "--clone", base_env, "-y", "--quiet"],
                check=True,
                capture_output=True,
                text=True
            )
            print(f"  ✓ Created {env_name}")
        else:
            # Reuse mode: Only create if missing
            if env_name in existing_envs:
                print(f"  ✓ {env_name} already exists (reusing)")
            else:
                print(f"  Creating {env_name} (cloning from {base_env})...")
                subprocess.run(
                    [conda_exe, "create", "--name", env_name, "--clone", base_env, "-y", "--quiet"],
                    check=True,
                    capture_output=True,
                    text=True
                )
                print(f"  ✓ Created {env_name}")


    print("Conda environment setup complete!")
    print()

@weave.op()
def _run_researcher_once(slug: str, iteration: int, run_id: int) -> tuple[int, str, int]:
    max_parallel_workers, gpu_pool, _ = _calculate_max_parallel_workers_and_pools()
    cpu_core_pool = _create_cpu_core_pool(max_parallel_workers)

    print(f"Researcher Agent: max_parallel_workers={max_parallel_workers}, CPU pools={len(cpu_core_pool)}, GPU pools={len(gpu_pool)}")

    agent = ResearcherAgent(slug, iteration, run_id=run_id, cpu_core_pool=cpu_core_pool, gpu_pool=gpu_pool)
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

def _extract_now_recommendations(recommendations: dict) -> dict:
    """Extract only MUST_HAVE recommendations from model recommendations.

    Filters out citations and est_runtime_minutes_gpu fields.
    Returns a dict with the same structure but only MUST_HAVE sections.
    """
    now_only = {}

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

    loss_fn = recommendations.get("loss_function", {})
    if loss_fn and "MUST_HAVE" in loss_fn:
        now_only["loss_function"] = {
            "MUST_HAVE": loss_fn["MUST_HAVE"]
        }

    hyperparams = recommendations.get("hyperparameters", {})
    if hyperparams and "MUST_HAVE" in hyperparams:
        now_section = hyperparams["MUST_HAVE"]
        cleaned_now = {}
        if "hyperparameters" in now_section:
            cleaned_now["hyperparameters"] = now_section["hyperparameters"]
        if "architectures" in now_section:
            cleaned_now["architectures"] = now_section["architectures"]
        now_only["hyperparameters"] = {"MUST_HAVE": cleaned_now}

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

    loss_fn = recommendations.get("loss_function", {})
    if loss_fn and "NICE_TO_HAVE" in loss_fn:
        later_only["loss_function"] = {
            "NICE_TO_HAVE": loss_fn["NICE_TO_HAVE"]
        }

    hyperparams = recommendations.get("hyperparameters", {})
    if hyperparams and "NICE_TO_HAVE" in hyperparams:
        later_section = hyperparams["NICE_TO_HAVE"]
        cleaned_later = {}
        if "hyperparameters" in later_section:
            cleaned_later["hyperparameters"] = later_section["hyperparameters"]
        if "architectures" in later_section:
            cleaned_later["architectures"] = later_section["architectures"]
        later_only["hyperparameters"] = {"NICE_TO_HAVE": cleaned_later}

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

    loss_fn = recommendations.get("loss_function", {})
    if loss_fn and "MUST_HAVE" in loss_fn:
        details.append("\n## Loss Function")
        now_loss = loss_fn["MUST_HAVE"]
        loss_name = now_loss.get("loss_function", "")
        loss_exp = now_loss.get("explanation", "")
        if loss_name:
            details.append(f"- Use {loss_name}: {loss_exp}")

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

        arch_list = now_section.get("architectures", [])
        if arch_list:
            details.append("\n### Architecture Recommendations")
            for item in arch_list:
                if isinstance(item, dict):
                    arch = item.get("architecture", "")
                    explanation = item.get("explanation", "")
                    if arch:
                        details.append(f"- {arch}: {explanation}")

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
    def __init__(self, slug: str, iteration: int, rollback_to_version: int | None = None):
        self.slug = slug
        self.iteration = iteration
        self.rollback_to_version = rollback_to_version
        self.base_dir = _TASK_ROOT / slug
        self.outputs_dir = self.base_dir / _OUTPUTS_DIRNAME / str(iteration)

    def _get_external_data_listing(self) -> str:
        """Get directory listing of external_data_* folders in outputs/<iteration>."""
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

    def _rollback(self, model_iterations: list[str]) -> None:
        """Roll back all models to self.rollback_to_version.

        Moves version folders beyond N into _trash/<timestamp>/ and deletes checkpoint rows beyond N.
        Also removes baseline_results.json so affected models get re-run.
        """
        target = self.rollback_to_version
        parent_outputs = self.base_dir / _OUTPUTS_DIRNAME

        found = False
        for iter_suffix in model_iterations:
            model_dir = parent_outputs / iter_suffix
            if (model_dir / str(target)).exists():
                found = True
                break
        if not found:
            raise ValueError(
                f"Rollback failed: version {target} not found in any model outputs ({model_iterations})"
            )

        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        conn = _create_checkpoint_db()

        for iter_suffix in model_iterations:
            model_dir = parent_outputs / iter_suffix
            if not model_dir.exists():
                continue

            trash_dir = model_dir / "_trash" / timestamp
            moved_any = False

            for entry in sorted(model_dir.iterdir()):
                if not entry.is_dir() or entry.name.startswith("_"):
                    continue
                try:
                    version_num = int(entry.name)
                except ValueError:
                    continue
                if version_num > target:
                    if not moved_any:
                        trash_dir.mkdir(parents=True, exist_ok=True)
                        moved_any = True
                    dest = trash_dir / entry.name
                    entry.rename(dest)
                    print(f"Rollback: moved {entry} -> {dest}")

            delete_checkpoints_after(conn, self.slug, iter_suffix, target)
            print(f"Rollback: deleted checkpoints after v{target} for iteration {iter_suffix}")

        # Remove baseline_results.json so models get re-queued
        baseline_path = self.outputs_dir / "baseline_results.json"
        if baseline_path.exists():
            baseline_path.unlink()
            print("Rollback: removed baseline_results.json")

        conn.close()
        print(f"Rollback to version {target} complete")

    @weave.op()
    def run(self) -> Tuple[bool, str]:

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
        model_rec_path = self.outputs_dir / "model_recommendations.json"
        if model_rec_path.exists():
            with open(model_rec_path, "r") as f:
                model_recommendations = json.load(f)
        else:
            model_rec_agent = ModelRecommenderAgent(self.slug, self.iteration)

            hitl_models = get_instructions()["# Models"]

            if hitl_models and len(hitl_models) > 0:
                # HITL mode: Use human-specified models, skip LLM selection
                print(f"HITL mode: Using manually specified models: {hitl_models}")
                model_recommendations = model_rec_agent.run(model_list=hitl_models, use_dynamic_selection=False)
            else:
                # Auto mode: Let LLM select models dynamically
                print("Auto mode: Using LLM dynamic model selection")
                model_recommendations = model_rec_agent.run(use_dynamic_selection=True)

            if model_rec_path.exists():
                with open(model_rec_path, "r") as f:
                    model_recommendations = json.load(f)
            else:
                raise RuntimeError("No model recommendations found")

        now_recommendations_all = {}
        later_recommendations_all = {}

        for model_name, recommendations in model_recommendations.items():
            now_recs = _extract_now_recommendations(recommendations)
            now_recommendations_all[model_name] = now_recs
            later_recommendations_all[model_name] = _extract_later_recommendations(recommendations)
        
        now_rec_path = self.outputs_dir / "now_recommendations.json"
        with open(now_rec_path, "w") as f:
            json.dump(now_recommendations_all, f, indent=2)

        later_rec_path = self.outputs_dir / "later_recommendations.json"
        with open(later_rec_path, "w") as f:
            json.dump(later_recommendations_all, f, indent=2)

        # Phase 4: Baseline Developer Stage - Evaluate models in parallel with NOW recommendations

        if self.rollback_to_version is not None:
            model_iterations = [
                f"{self.iteration}_{idx}"
                for idx in range(1, len(now_recommendations_all) + 1)
            ]
            self._rollback(model_iterations)

        baseline_results = {}

        max_parallel_workers, gpu_pool_list, gpu_isolation_mode = _calculate_max_parallel_workers_and_pools()

        # Force single worker in HITL mode
        if _HITL_SOTA and max_parallel_workers > 1:
            print("[HITL] Warning: HITL mode enabled, overriding parallel workers to 1")
            max_parallel_workers = 1
            gpu_pool_list = gpu_pool_list[:1] if gpu_pool_list else []

        cpu_core_pool_list = _create_cpu_core_pool(max_parallel_workers)

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

        cpu_core_pool = None
        if len(cpu_core_pool_list) > 0 and num_models > 1:
            cpu_core_pool = Queue()
            for core_range in cpu_core_pool_list:
                cpu_core_pool.put(core_range)
            print(f"CPU core pool created with {max_parallel_workers} core ranges")

        gpu_pool = None
        if num_models > 1 and len(gpu_pool_list) > 0:
            gpu_pool = Queue()
            for gpu_id in gpu_pool_list:
                gpu_pool.put(gpu_id)
            if gpu_isolation_mode == "mig":
                print(f"MIG pool created with {len(gpu_pool_list)} instances")
            elif gpu_isolation_mode == "multi-gpu":
                print(f"GPU pool created with {len(gpu_pool_list)} GPUs")

        existing_baseline_results = {}
        baseline_path = self.outputs_dir / "baseline_results.json"
        if baseline_path.exists():
            try:
                with open(baseline_path, 'r') as f:
                    existing_baseline_results = json.load(f)
                print(f"Loaded existing baseline results with {len(existing_baseline_results)} models")
            except Exception as e:
                print(f"Warning: Could not load existing baseline results: {e}")

        external_data_listing = self._get_external_data_listing()
        plan_content = self._get_plan_content()
        print(f"External data listing: {len(external_data_listing)} chars")
        print(f"Plan content: {len(plan_content)} chars")

        if not os.path.exists(self.outputs_dir / "baseline_results.json") or len(existing_baseline_results) < len(now_recommendations_all):
            num_models = len(now_recommendations_all)
            _ensure_conda_environments(num_models)

            tasks = []
            for idx, (model_name, now_recommendations) in enumerate(now_recommendations_all.items(), start=1):
                key = model_name
                dev_iter = f"{self.iteration}_{idx}"
                later_recs = later_recommendations_all.get(model_name, {})

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

            baseline_results = existing_baseline_results.copy()

            # Run developer agents — directly on main thread for HITL (stdin required),
            # otherwise in parallel using ThreadPoolExecutor
            if tasks:
                if _HITL_SOTA:
                    print(f"Running {len(tasks)} baseline task(s) sequentially (HITL mode)")
                    for task_args in tasks:
                        try:
                            result_key, best_score, best_code_file, blacklisted_ideas, successful_ideas = _run_developer_baseline(*task_args)
                            baseline_results[result_key] = {
                                "model_name": result_key,
                                "best_score": best_score,
                                "best_code_file": best_code_file or "",
                                "blacklisted_ideas": blacklisted_ideas,
                                "successful_ideas": successful_ideas,
                                "now_recommendations": now_recommendations_all.get(result_key, {}),
                                "later_recommendations": later_recommendations_all.get(result_key, {})
                            }
                            baseline_path = self.outputs_dir / "baseline_results.json"
                            with open(baseline_path, "w") as f:
                                json.dump(baseline_results, f, indent=2)
                        except Exception as e:
                            print(f"Error in baseline task: {e}")
                            continue
                else:
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
                                    "now_recommendations": now_recommendations_all.get(result_key, {}),
                                    "later_recommendations": later_recommendations_all.get(result_key, {})
                                }
                                baseline_path = self.outputs_dir / "baseline_results.json"
                                with open(baseline_path, "w") as f:
                                    json.dump(baseline_results, f, indent=2)
                            except Exception as e:
                                print(f"Error in baseline task: {e}")
                                continue
            else:
                print("No new baseline tasks to run (all models already completed)")

            if baseline_results:
                baseline_path = self.outputs_dir / "baseline_results.json"
                with open(baseline_path, "w") as f:
                    json.dump(baseline_results, f, indent=2)
            else:
                raise RuntimeError("All developer baseline runs failed")

        baseline_path = self.outputs_dir / "baseline_results.json"
        return True, str(baseline_path)
    