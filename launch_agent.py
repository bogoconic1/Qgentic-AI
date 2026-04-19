"""CLI entry for a Qgentic-AI run.

Loads `GOAL.md`, downloads the Kaggle competition data if needed, and hands
control to `MainAgent.run()` for the rest of the session. No termination in
software — SIGKILL when satisfied.
"""

import argparse
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import wandb
import weave

from agents.main_agent import MainAgent
from project_config import get_config, get_config_value
from utils.competition_data import download_competition_data


_TASK_ROOT = Path(get_config()["paths"]["task_root"])


def _resolve_wandb_target(
    cli_entity: Optional[str], cli_project: Optional[str]
) -> Tuple[Optional[str], Optional[str]]:
    """Determine the wandb entity/project from CLI, env vars, or config."""
    entity = next(
        (
            value
            for value in (
                cli_entity,
                os.environ.get("WANDB_ENTITY"),
                get_config_value("tracking", "wandb", "entity", default=None),
            )
            if value
        ),
        None,
    )
    project = next(
        (
            value
            for value in (
                cli_project,
                os.environ.get("WANDB_PROJECT"),
                get_config_value("tracking", "wandb", "project", default=None),
            )
            if value
        ),
        None,
    )
    return entity, project


def _init_tracking(args: argparse.Namespace) -> None:
    """Initialise wandb and weave using the best available configuration."""
    entity, project = _resolve_wandb_target(args.wandb_entity, args.wandb_project)
    run_name = getattr(args, "wandb_run_name", None) or f"{args.run_id}-{args.slug}"

    if not project:
        wandb.init(mode="disabled", name=run_name)
        return

    wandb_kwargs = {"project": project, "name": run_name}
    if entity:
        wandb_kwargs["entity"] = entity
    wandb.init(**wandb_kwargs)
    weave_project = f"{entity}/{project}" if entity else project
    weave.init(project_name=weave_project)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch a Qgentic-AI Main Agent session against a Kaggle competition.",
    )
    parser.add_argument(
        "--slug",
        type=str,
        required=True,
        help="Kaggle competition slug; data will be downloaded into task/<slug>/.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run identifier (default: current timestamp %%Y%%m%%d_%%H%%M%%S).",
    )
    parser.add_argument(
        "--goal-file",
        type=Path,
        default=None,
        help="Path to GOAL.md (default: task/<slug>/GOAL.md).",
    )
    parser.add_argument("--wandb-entity", type=str, help="Weights & Biases entity name")
    parser.add_argument("--wandb-project", type=str, help="Weights & Biases project name")
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        help="Optional wandb run name override (defaults to '<run_id>-<slug>').",
    )
    args = parser.parse_args()

    if args.run_id is None:
        args.run_id = time.strftime("%Y%m%d_%H%M%S")

    base_dir = _TASK_ROOT / args.slug
    base_dir.mkdir(parents=True, exist_ok=True)

    goal_file = args.goal_file or base_dir / "GOAL.md"
    if not goal_file.exists():
        raise FileNotFoundError(
            f"Goal file not found: {goal_file}. Create one, or pass --goal-file."
        )
    goal_text = goal_file.read_text()

    os.environ["TASK_SLUG"] = args.slug
    _init_tracking(args)

    try:
        download_competition_data(args.slug, base_dir)
        MainAgent(slug=args.slug, run_id=args.run_id, goal_text=goal_text).run()
    finally:
        try:
            weave.finish()
        except Exception:
            pass
        try:
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
