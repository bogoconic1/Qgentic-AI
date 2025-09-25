from typing import Tuple
from pathlib import Path

from agents.researcher import ResearcherAgent
from agents.developer import DeveloperAgent
import weave
import wandb

class Orchestrator:
    def __init__(self, slug: str, iteration: int):
        self.slug = slug
        self.iteration = iteration
        self.researcher = ResearcherAgent(slug, iteration)
        self.developer = DeveloperAgent(slug, iteration)

    @weave.op()
    def run(self, max_code_tries: int = 50) -> Tuple[bool, str]:
        # if plan exists, don't run the researcher agent
        if Path(f"task/{self.slug}/outputs/{self.iteration}/plan.md").exists():
            with open(f"task/{self.slug}/outputs/{self.iteration}/plan.md", "r") as f:
                plan = f.read()
        else:
            plan = self.researcher.build_plan()
            with open(f"task/{self.slug}/outputs/{self.iteration}/plan.md", "w") as f:
                f.write(plan)
        
        success = self.developer.run(plan, max_tries=max_code_tries)

        return success, plan
    
