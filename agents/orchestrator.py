from typing import Tuple
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

from agents.researcher import ResearcherAgent
from agents.developer import DeveloperAgent
from agents.starter import StarterAgent
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
def _run_developer_baseline(slug: str, iteration_suffix: str, plan: str, model_name: str, example_details: str, key: str):
    """Run a single baseline DeveloperAgent and return (key, best_score, best_code)."""
    dev = DeveloperAgent(slug, iteration_suffix, model_name=model_name, example_details=example_details)
    best_score, best_code, blacklisted_ideas = dev.run(plan, max_time_seconds=3600)
    return key, best_score, best_code, blacklisted_ideas

class Orchestrator:
    def __init__(self, slug: str, iteration: int):
        self.slug = slug
        self.iteration = iteration
        self.base_dir = _TASK_ROOT / slug
        self.outputs_dir = self.base_dir / _OUTPUTS_DIRNAME / str(iteration)

    @weave.op()
    def run(self, max_time_seconds: int | None = 6 * 3600) -> Tuple[bool, str]:

        # starter suggestions
        starter_suggestion_path = self.outputs_dir / "starter_suggestions.json"
        if starter_suggestion_path.exists():
            with open(starter_suggestion_path, "r") as f:
                suggestions = json.load(f)
        else:
            starter = StarterAgent(self.slug, self.iteration)
            starter.run()

            if starter_suggestion_path.exists():
                with open(starter_suggestion_path, "r") as f:
                    suggestions = json.load(f)
            else:
                raise RuntimeError("No starter suggestions found")
        
        # research plan
        plan_path = self.outputs_dir / "plan.md"
        if plan_path.exists():
            with open(plan_path, "r") as f:
                plan = f.read()
        else:
            _run_researcher_once(self.slug, self.iteration, 1)

            if plan_path.exists():
                with open(plan_path, "r") as f:
                    plan = f.read()
            else:
                raise RuntimeError("No plan found")
            
        # Baseline stage: evaluate 5 starter suggestions with constrained developer runs
        tasks = []
        with ProcessPoolExecutor(max_workers=5) as ex:
            for i in range(1, 6):
                key = f"model_{i}"
                entry = suggestions.get(key)
                if not isinstance(entry, dict):
                    continue
                model_name = entry.get("suggestion", "")
                example_details = entry.get("details", "")
                dev_iter = f"{self.iteration}_{i}"
                fut = ex.submit(_run_developer_baseline, self.slug, dev_iter, plan, model_name, example_details, key)
                tasks.append(fut)

            for fut in as_completed(tasks):
                try:
                    key, best_score, best_code, blacklisted_ideas = fut.result()
                    if isinstance(suggestions.get(key), dict):
                        suggestions[key]["best_score"] = best_score
                        suggestions[key]["best_code"] = best_code or ""
                        suggestions[key]["blacklisted_ideas"] = blacklisted_ideas
                except Exception:
                    continue
        '''
        for i in range(1, 2):
            key = f"model_{i}"
            entry = suggestions.get(key)
            if not isinstance(entry, dict):
                continue
            model_name = entry.get("suggestion", "")
            example_details = entry.get("details", "")
            dev_iter = f"{self.iteration}_{i}"
            try:
                _, best_score, best_code, blacklisted_ideas = _run_developer_baseline(self.slug, dev_iter, plan, model_name, example_details, key)
                if isinstance(suggestions.get(key), dict):
                    suggestions[key]["best_score"] = best_score
                    suggestions[key]["best_code"] = best_code or ""
                    suggestions[key]["blacklisted_ideas"] = blacklisted_ideas or []
            except Exception:
                continue'''

        # Persist combined results alongside original suggestions
        baseline_path = self.outputs_dir / "baseline_results.json"
        with open(baseline_path, "w") as f:
            json.dump(suggestions, f, indent=2)
    