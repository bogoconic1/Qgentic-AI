from pathlib import Path
from datetime import datetime
import json

from agents.developer import DeveloperAgent
from project_config import get_config
from tools.research import deep_research
from tools.helpers import _build_directory_listing
from utils.checkpoint import (
    create_db as _create_checkpoint_db,
    delete_checkpoints_after,
)
from utils.competition_data import download_competition_data
import weave


_CONFIG = get_config()
_RUNTIME_CFG = _CONFIG["runtime"]
_PATH_CFG = _CONFIG["paths"]
_TASK_ROOT = Path(_PATH_CFG["task_root"])


class Orchestrator:
    def __init__(
        self, slug: str, run_id: str, rollback_to_version: int | None = None
    ):
        self.slug = slug
        self.run_id = run_id
        self.rollback_to_version = rollback_to_version
        self.base_dir = _TASK_ROOT / slug
        self.outputs_dir = self.base_dir / self.run_id

    def _get_external_data_listing(self) -> str:
        """Get directory listing of external_data_* folders under the run dir."""
        if not self.outputs_dir.exists():
            return "No external data directories found."

        external_data_dirs = [
            d
            for d in self.outputs_dir.iterdir()
            if d.is_dir() and d.name == "external_data"
        ]

        if not external_data_dirs:
            return "No external data directories found."

        lines = []
        for ext_dir in sorted(external_data_dirs):
            lines.append(f"\n{ext_dir.name}/")
            dir_listing = _build_directory_listing(str(ext_dir))
            lines.append(dir_listing)

        return "\n".join(lines)

    def _rollback(self) -> None:
        """Roll back the current run's developer outputs to self.rollback_to_version."""
        target = self.rollback_to_version
        model_dir = self.outputs_dir

        if not (model_dir / str(target)).exists():
            raise ValueError(
                f"Rollback failed: version {target} not found in {model_dir}"
            )

        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        conn = _create_checkpoint_db()

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

        delete_checkpoints_after(conn, self.slug, self.run_id, target)
        print(f"Rollback: deleted checkpoints after v{target} for run {self.run_id}")

        baseline_path = self.outputs_dir / "baseline_results.json"
        if baseline_path.exists():
            baseline_path.unlink()
            print("Rollback: removed baseline_results.json")

        conn.close()
        print(f"Rollback to version {target} complete")

    @weave.op()
    def run(self) -> tuple[bool, str]:
        # Phase 0: Ensure raw competition data is on disk before any agent runs.
        download_competition_data(self.slug, self.base_dir)

        # Phase 1: Deep Research — produce PLAN_1.md via the researcher subagent.
        description = (self.base_dir / "description.md").read_text()
        research_iter = 1
        instruction = (
            f"Research Kaggle competition '{self.slug}'. Use web_research + "
            f"web_fetch to study the domain, prior art, and top approaches. "
            f"Return a markdown PLAN that a developer can execute on — "
            f"include domain context, validated hypotheses, data strategy, "
            f"feature engineering priorities, and modeling direction.\n\n"
            f"<competition_description>\n{description}\n</competition_description>"
        )
        plan_content = deep_research(instruction, self.slug, self.run_id, research_iter)

        external_data_listing = self._get_external_data_listing()
        print(f"External data listing: {len(external_data_listing)} chars")
        print(f"Plan content: {len(plan_content)} chars")

        if self.rollback_to_version is not None:
            self._rollback()

        # Phase 2: Developer — one baseline run per invocation.
        baseline_path = self.outputs_dir / "baseline_results.json"
        if baseline_path.exists():
            with open(baseline_path, "r") as f:
                existing = json.load(f)
            print(f"Baseline already completed: {existing}")
            return True, str(baseline_path)

        baseline_time_limit = _RUNTIME_CFG["baseline_time_limit"]
        dev = DeveloperAgent(
            self.slug,
            self.run_id,
            "1",
            external_data_listing=external_data_listing,
            plan_content=plan_content,
        )
        best_score, best_code_file = dev.run(max_time_seconds=baseline_time_limit)

        baseline_results = {
            "best_score": best_score,
            "best_code_file": best_code_file or "",
        }
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        with open(baseline_path, "w") as f:
            json.dump(baseline_results, f, indent=2)
        return True, str(baseline_path)
