from pathlib import Path

import weave

from project_config import get_config
from utils.competition_data import download_competition_data


_CONFIG = get_config()
_TASK_ROOT = Path(_CONFIG["paths"]["task_root"])


class Orchestrator:
    def __init__(self, slug: str, run_id: str):
        self.slug = slug
        self.run_id = run_id
        self.base_dir = _TASK_ROOT / slug
        self.outputs_dir = self.base_dir / self.run_id

    @weave.op()
    def run(self) -> None:
        download_competition_data(self.slug, self.base_dir)
        raise NotImplementedError("Main Agent not yet wired")
