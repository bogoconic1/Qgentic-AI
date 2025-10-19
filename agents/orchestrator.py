from typing import Tuple
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import json

from agents.researcher import ResearcherAgent
from agents.developer import DeveloperAgent
from agents.starter import StarterAgent
from project_config import get_config
import weave
import wandb


_CONFIG = get_config()
_RUNTIME_CFG = _CONFIG.get("runtime", {}) if isinstance(_CONFIG, dict) else {}
_PATH_CFG = _CONFIG.get("paths", {}) if isinstance(_CONFIG, dict) else {}
_TASK_ROOT = Path(_PATH_CFG.get("task_root", "task"))
_OUTPUTS_DIRNAME = _PATH_CFG.get("outputs_dirname", "outputs")
_DEFAULT_PARALLEL = int(_RUNTIME_CFG.get("researcher_parallel_runs", 1) or 1)

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
def _run_developer_baseline(slug: str, iteration_suffix: str, plan: str, model_name: str, example_code: str, key: str):
    """Run a single baseline DeveloperAgent and return (key, best_score, best_code)."""
    dev = DeveloperAgent(slug, iteration_suffix, model_name=model_name, example_code=example_code)
    best_score, best_code = dev.run(plan, max_time_seconds=3600)
    return key, best_score, best_code

class Orchestrator:
    def __init__(self, slug: str, iteration: int):
        self.slug = slug
        self.iteration = iteration
        self.researcher = ResearcherAgent(slug, iteration)
        self.developer = DeveloperAgent(slug, iteration)
        self.base_dir = _TASK_ROOT / slug
        self.outputs_dir = self.base_dir / _OUTPUTS_DIRNAME / str(iteration)

    @weave.op()
    def run(self, max_time_seconds: int | None = 6 * 3600) -> Tuple[bool, str]:
        # if plan exists, don't run the researcher agent
        plan_path = self.outputs_dir / "plan.md"
        if plan_path.exists():
            with open(plan_path, "r") as f:
                plan = f.read()
        else:
            # starter = StarterAgent(self.slug, self.iteration)
            # starter_summary = starter.run()
            parallel = int(os.environ.get("RESEARCHER_PARALLEL_RUNS", _DEFAULT_PARALLEL) or _DEFAULT_PARALLEL)
            results: list[tuple[int, str, int]] = []
            _, _, _ = _run_researcher_once(self.slug, self.iteration, 1)
            '''
            with ProcessPoolExecutor(max_workers=parallel) as ex:
                futures = [ex.submit(_run_researcher_once, self.slug, self.iteration, i + 1) for i in range(parallel)]
                for fut in as_completed(futures):
                    try:
                        results.append(fut.result())
                    except Exception:
                        continue'''
            # Prefer the first run's plan.md; fall back to any available plan file; else single-run
            plan_md_path = self.outputs_dir / "plan.md"
            if plan_md_path.exists():
                with open(plan_md_path, "r") as f:
                    plan = f.read()
            else:
                raise RuntimeError("No plan found")
        # Baseline stage: evaluate 5 starter suggestions with constrained developer runs
        try:
            suggestions_path = self.outputs_dir / "starter_suggestions.json"
            if suggestions_path.exists():
                with open(suggestions_path, "r") as f:
                    suggestions = json.load(f) or {}

                # Dispatch runs in parallel using ProcessPoolExecutor
                tasks = []
                with ProcessPoolExecutor(max_workers=5) as ex:
                    for i in range(1, 6):
                        key = f"model_{i}"
                        entry = suggestions.get(key)
                        if not isinstance(entry, dict):
                            continue
                        model_name = entry.get("suggestion", "")
                        example_code = entry.get("code", "")
                        dev_iter = f"{self.iteration}_{i}"
                        fut = ex.submit(_run_developer_baseline, self.slug, dev_iter, plan, model_name, example_code, key)
                        tasks.append(fut)

                    for fut in as_completed(tasks):
                        try:
                            key, best_score, best_code = fut.result()
                            if isinstance(suggestions.get(key), dict):
                                suggestions[key]["best_score"] = best_score
                                suggestions[key]["best_code"] = best_code or ""
                        except Exception:
                            continue

                # Persist combined results alongside original suggestions
                baseline_path = self.outputs_dir / "baseline_results.json"
                try:
                    with open(baseline_path, "w") as f:
                        json.dump(suggestions, f, indent=2)
                except Exception:
                    pass
        except Exception:
            pass
    
