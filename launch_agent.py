import argparse
import os
from typing import Optional, Tuple

from agents.orchestrator import Orchestrator
from project_config import get_config_value
import weave
import wandb


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
    run_name = getattr(args, "wandb_run_name", None) or f"{args.iteration}-{args.slug}"

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


def main():
    parser = argparse.ArgumentParser(description="Run Researcher+Developer pipeline")
    parser.add_argument("--slug", type=str, help="Competition slug under task/<slug>")
    parser.add_argument("--iteration", type=int, help="Iteration number (e.g., 1)")
    parser.add_argument("--wandb-entity", type=str, help="Weights & Biases entity name")
    parser.add_argument("--wandb-project", type=str, help="Weights & Biases project name")
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        help="Optional wandb run name override (defaults to '<iteration>-<slug>')",
    )
    parser.add_argument(
        "--rollback-to-version",
        type=int,
        default=None,
        help="Rollback to a specific version: moves later version folders to _trash/ and deletes their checkpoints",
    )
    args = parser.parse_args()

    os.environ["TASK_SLUG"] = args.slug
    _init_tracking(args)

    orchestrator = Orchestrator(args.slug, args.iteration, rollback_to_version=args.rollback_to_version)
    orchestrator.run()

    # Gracefully close tracking backends to avoid hanging background threads
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
