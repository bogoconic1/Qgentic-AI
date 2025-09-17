from typing import Tuple

from agents.researcher import ResearcherAgent
from agents.developer import DeveloperAgent


class Orchestrator:
    def __init__(self, slug: str, iteration: int):
        self.slug = slug
        self.iteration = int(iteration)
        self.researcher = ResearcherAgent(slug)
        self.developer = DeveloperAgent(slug, iteration)

    def run(self, max_code_tries: int = 3) -> Tuple[bool, str]:
        plan = self.researcher.build_plan()
        success = self.developer.run(plan, max_tries=max_code_tries)
        return success, plan


