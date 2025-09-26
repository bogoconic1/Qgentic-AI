from typing import Tuple
from pathlib import Path

from agents.researcher import ResearcherAgent
from agents.developer import DeveloperAgent
from project_config import get_config
import weave
import wandb


_CONFIG = get_config()
_PATH_CFG = _CONFIG.get("paths", {}) if isinstance(_CONFIG, dict) else {}
_TASK_ROOT = Path(_PATH_CFG.get("task_root", "task"))
_OUTPUTS_DIRNAME = _PATH_CFG.get("outputs_dirname", "outputs")

class Orchestrator:
    def __init__(self, slug: str, iteration: int):
        self.slug = slug
        self.iteration = iteration
        self.researcher = ResearcherAgent(slug, iteration)
        self.developer = DeveloperAgent(slug, iteration)
        self.base_dir = _TASK_ROOT / slug
        self.outputs_dir = self.base_dir / _OUTPUTS_DIRNAME / str(iteration)

    @weave.op()
    def run(self, max_code_tries: int = 50) -> Tuple[bool, str]:
        # if plan exists, don't run the researcher agent
        plan_path = self.outputs_dir / "plan.md"
        if plan_path.exists():
            with open(plan_path, "r") as f:
                plan = f.read()
        else:
            plan = self.researcher.build_plan()
            with open(plan_path, "w") as f:
                f.write(plan)
        
        success = self.developer.run(plan, max_tries=max_code_tries)

        return success, plan
    
