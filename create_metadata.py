from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create competition metadata file.")
    parser.add_argument(
        "--meta-kaggle-path",
        type=Path,
        default=Path("/workspace/meta-kaggle"),
        help="Path to the Meta Kaggle dataset directory.",
    )
    parser.add_argument(
        "--task-path",
        type=Path,
        default=Path("/workspace/Qgentic-AI/task"),
        help="Path to the task output directory.",
    )
    parser.add_argument(
        "--competition-slug",
        default="tabular-playground-series-dec-2021",
        help="Slug of the Kaggle competition to process.",
    )
    return parser.parse_args()


def get_competition_and_forum_ids(meta_kaggle_path: Path, slug: str):
    cols = ["Id", "ForumId", "Slug", "EnabledDate", "DeadlineDate"]
    df = pd.read_csv(meta_kaggle_path / "Competitions.csv", usecols=cols)
    row = df.loc[df["Slug"] == slug].iloc[0]
    competition_id = int(row["Id"]) if pd.notna(row["Id"]) else -1
    forum_id = row["ForumId"] if pd.notna(row["ForumId"]) else -1
    start_ts = pd.to_datetime(row["EnabledDate"], errors="coerce", utc=True)
    cutoff_ts = pd.to_datetime(row["DeadlineDate"], errors="coerce", utc=True)
    if pd.notna(cutoff_ts):
        start_ts = start_ts.tz_convert("UTC")
        cutoff_ts = cutoff_ts.tz_convert("UTC")
    else:
        cutoff_ts = None
    return competition_id, forum_id, start_ts, cutoff_ts


def main() -> None:
    args = parse_args()

    load_dotenv()

    task_dir = args.task_path / args.competition_slug
    os.makedirs(task_dir, exist_ok=True)

    competition_id, forum_id, start_ts, cutoff_ts = get_competition_and_forum_ids(
        args.meta_kaggle_path, args.competition_slug
    )
    print(
        f"Competition ID: {competition_id}, Forum ID: {forum_id}, "
        f"Start (UTC): {start_ts}, Deadline (UTC): {cutoff_ts}"
    )

    metadata_path = task_dir / "comp_metadata.yaml"
    metadata = {"START_DATE": start_ts.strftime("%Y-%m-%d"), "END_DATE": cutoff_ts.strftime("%Y-%m-%d")}
    with open(metadata_path, "w", encoding="utf-8") as f:
        yaml.dump(metadata, f)


if __name__ == "__main__":
    main()
