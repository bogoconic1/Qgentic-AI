"""Top-level entry point for the standalone goal-mode developer agent.

Usage:
    python goal_mode.py
    python goal_mode.py --goal-file GOAL.md --run-id abc --max-versions 30
"""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import wandb
import weave

from agents.goal_developer import run_goal_mode
from project_config import get_config_value


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


def _init_tracking(
    wandb_entity: Optional[str],
    wandb_project: Optional[str],
    run_name: str,
) -> None:
    """Initialise wandb and weave using the best available configuration."""
    entity, project = _resolve_wandb_target(wandb_entity, wandb_project)

    if not project:
        # Fall back to disabled mode so downstream wandb.log calls are no-ops.
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
        description="Iterate a generate-execute-review loop against a free-form GOAL.md."
    )
    parser.add_argument(
        "--goal-file",
        type=Path,
        default=Path("GOAL.md"),
        help="Path to the goal description (default: GOAL.md in the current directory).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Identifier for this run (default: timestamp).",
    )
    parser.add_argument(
        "--max-versions",
        type=int,
        default=500,
        help="Maximum number of iterations (default: 500).",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Weights & Biases entity name (overrides env / config).",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Weights & Biases project name (overrides env / config).",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Optional wandb run name override (defaults to 'goal-<run_id>').",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    if not args.goal_file.exists():
        raise FileNotFoundError(
            f"Goal file not found: {args.goal_file}. "
            f"Create one at {args.goal_file.resolve()} or pass --goal-file."
        )

    goal_text = args.goal_file.read_text()
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path("goal_runs") / run_id

    run_name = args.wandb_run_name or f"goal-{run_id}"
    _init_tracking(args.wandb_entity, args.wandb_project, run_name)

    try:
        history = run_goal_mode(
            goal_text=goal_text,
            run_dir=run_dir,
            max_versions=args.max_versions,
        )
    finally:
        # Gracefully close tracking backends to avoid hanging background threads.
        try:
            weave.finish()
        except Exception:
            pass
        try:
            wandb.finish()
        except Exception:
            pass

    final = history[-1] if history else None
    final_done = bool(final and final.review and final.review.done)
    final_score = final.review.score if final and final.review else None
    print(
        f"goal_mode finished: {len(history)} iterations, done={final_done}, "
        f"final_score={final_score}, run_dir={run_dir}"
    )


if __name__ == "__main__":
    main()
