from typing import Tuple
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

from agents.researcher import ResearcherAgent
from agents.developer import DeveloperAgent
from project_config import get_config
import weave
import wandb


_CONFIG = get_config()
_RUNTIME_CFG = _CONFIG.get("runtime", {}) if isinstance(_CONFIG, dict) else {}
_PATH_CFG = _CONFIG.get("paths", {}) if isinstance(_CONFIG, dict) else {}
_TASK_ROOT = Path(_PATH_CFG.get("task_root", "task"))
_OUTPUTS_DIRNAME = _PATH_CFG.get("outputs_dirname", "outputs")
_DEFAULT_PARALLEL = int(_RUNTIME_CFG.get("researcher_parallel_runs", 1) or 1)


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
            parallel = int(os.environ.get("RESEARCHER_PARALLEL_RUNS", _DEFAULT_PARALLEL) or _DEFAULT_PARALLEL)
            results: list[tuple[int, str, int]] = []
            with ProcessPoolExecutor(max_workers=parallel) as ex:
                futures = [ex.submit(_run_researcher_once, self.slug, self.iteration, i + 1) for i in range(parallel)]
                for fut in as_completed(futures):
                    try:
                        results.append(fut.result())
                    except Exception:
                        continue
            # Prefer the first run's plan.md; fall back to any available plan file; else single-run
            plan_md_path = self.outputs_dir / "plan.md"
            if plan_md_path.exists():
                with open(plan_md_path, "r") as f:
                    plan = f.read()
            else:
                raise RuntimeError("No plan found")
        # success = self.developer.run(plan, max_time_seconds=max_time_seconds)

        # return success, plan
    
