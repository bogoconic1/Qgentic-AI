from typing import Tuple
import subprocess
from pathlib import Path

from agents.researcher import ResearcherAgent
from agents.developer import DeveloperAgent

class Orchestrator:
    def __init__(self, slug: str, iteration: int):
        self.slug = slug
        self.iteration = iteration
        self.researcher = ResearcherAgent(slug)
        self.developer = DeveloperAgent(slug, iteration)

    def run(self, max_code_tries: int = 20) -> Tuple[bool, str]:
        # if plan exists, don't run the researcher agent
        if Path(f"task/{self.slug}/outputs/{self.iteration}/plan.md").exists():
            with open(f"task/{self.slug}/outputs/{self.iteration}/plan.md", "r") as f:
                plan = f.read()
        else:
            plan = self.researcher.build_plan()
            with open(f"task/{self.slug}/outputs/{self.iteration}/plan.md", "w") as f:
                f.write(plan)
        success = self.developer.run(plan, max_tries=max_code_tries)
        if success:
            # mlebench grade-sample /workspace/gstar-project/task/learning-agency-lab-automated-essay-scoring-2/outputs/3/submission.csv learning-agency-lab-automated-essay-scoring-2
            subprocess.run([
                "mlebench", "grade-sample",
                f"/workspace/gstar-project/task/{self.slug}/outputs/{self.iteration}/submission.csv",
                self.slug
            ])
  
        return success, plan
    

