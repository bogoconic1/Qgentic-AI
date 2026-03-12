#!/usr/bin/env python3
"""Scaffold the Kaggle legal retrieval task bundle for Qgentic-AI.

This script prepares the non-data files that Qgentic expects for the
`llm-agentic-legal-information-retrieval` competition and can optionally
download the Kaggle bundle before scaffolding.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

import pandas as pd


LEGAL_RETRIEVAL_SLUG = "llm-agentic-legal-information-retrieval"
REQUIRED_DATA_FILES = (
    "train.csv",
    "val.csv",
    "test.csv",
    "sample_submission.csv",
    "laws_de.csv",
    "court_considerations.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the Qgentic task bundle for Kaggle legal retrieval."
    )
    parser.add_argument(
        "--slug",
        default=LEGAL_RETRIEVAL_SLUG,
        help="Competition slug. Defaults to the legal retrieval competition.",
    )
    parser.add_argument(
        "--task-dir",
        type=Path,
        default=Path("task") / LEGAL_RETRIEVAL_SLUG,
        help="Destination task directory.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the Kaggle competition bundle before scaffolding.",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("/tmp/omnilex"),
        help="Temporary directory for the Kaggle archive.",
    )
    parser.add_argument(
        "--kaggle-bin",
        default="kaggle",
        help="Kaggle CLI executable to use when --download is set.",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=None,
        help="Override the validation-set size if val.csv is not present yet.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing helper files.",
    )
    return parser.parse_args()


def _description_text() -> str:
    return textwrap.dedent(
        """\
        # LLM Agentic Legal Information Retrieval

        This task adapts Qgentic-AI to the Kaggle competition
        `llm-agentic-legal-information-retrieval`.

        ## Objective

        For each legal question, predict a semicolon-separated list of Swiss legal
        citations that best matches the hidden gold set. The submission must use the
        exact canonical citation strings from the provided retrieval corpus.

        ## Data layout

        - `train.csv`: Public training queries with gold citations. These queries are
          mostly non-English.
        - `val.csv`: Small English validation set with gold citations. This is the
          primary local gating split.
        - `test.csv`: English evaluation queries without labels.
        - `laws_de.csv`: German statutory corpus keyed by canonical `citation`.
        - `court_considerations.csv`: Large court-decision corpus keyed by canonical
          `citation`.

        ## Metric

        The primary metric is citation-level Macro F1. For each query, compare the
        predicted citation set against the gold citation set, compute F1, and average
        across queries. Higher is better.

        ## Submission contract

        The generated `submission.csv` must contain:

        - `query_id`
        - `predicted_citations`

        `predicted_citations` must be a semicolon-separated string. The empty string is
        allowed if no citations are predicted.

        ## Competition constraints

        - Final Kaggle notebooks must run offline with no internet access.
        - End-to-end notebook runtime must stay within Kaggle's 12-hour limit.
        - Exact citation-string matching matters; invented or reformatted citations are
          counted as false positives.
        - Favor retrieval-first systems over generic text generation.
        """
    ).strip() + "\n"


def _instructions_text() -> str:
    return textwrap.dedent(
        """\
        # Researcher Instructions

        Treat this as a multilingual legal citation retrieval problem, not a generic
        text-generation benchmark.
        Use `val.csv` as the primary local score gate because it matches the English
        query distribution of `test.csv` better than `train.csv`.
        Focus research on retrieval-first strategies: BM25, hybrid retrieval,
        cross-lingual query expansion, citation normalization, and exact string
        matching against the provided corpora.
        The final Kaggle solution must run offline, so avoid plans that depend on
        external APIs at inference time.
        For fast iteration, it is acceptable to cap `court_considerations.csv` to
        roughly 100000 rows on the first pass, then remove that cap for serious runs.

        # Developer Instructions

        Build a retrieval-first pipeline around `train.csv`, `val.csv`, `test.csv`,
        `laws_de.csv`, and `court_considerations.csv`.
        The first version should stay close to the official baseline knobs:
        `top_k_laws=40`, `top_k_courts=40`, `max_iterations=3`, `max_tokens=512`,
        `temperature=0.1`, `n_ctx=8192`.
        Never invent citation strings. Every emitted citation must exactly match a
        `citation` value from `laws_de.csv` or `court_considerations.csv`.
        `cv_splits.json` intentionally uses an empty `train` split and a full `val`
        split so local scoring tracks the English validation set. Do not "repair" this
        into a normal train/validation fold. Use `train.csv` only as auxiliary data for
        retrieval heuristics, query expansion, or few-shot exemplars.
        Use `val.csv` to compute `cv_scores`, `cv_mean`, `cv_std`, and `cv_worst` in
        `train_stats.json`. `cv_worst` is the score Qgentic should optimize.
        Even for non-gradient retrieval baselines, write all required artifacts:
        `submission.csv`, `valid_preds.csv`, `train_stats.json`, `loss_curve.png`, and
        `metric_curve.png`.
        Placeholder plots are acceptable when there is no training loop, but they must
        exist as valid image files.
        `valid_preds.csv` should include at least `query_id`, `fold`, `gold_citations`,
        and `predicted_citations`.
        Keep the local winner easy to port into an offline Kaggle notebook that reads
        only `/kaggle/input/...` assets.

        # Models
        mistral-7b-instruct-v0.2
        """
    ).strip() + "\n"


def _metric_text() -> str:
    return textwrap.dedent(
        '''\
        """Citation-level Macro F1 metric for Kaggle legal retrieval."""

        from __future__ import annotations

        import math
        import re
        from collections.abc import Iterable, Sequence


        _WS_RE = re.compile(r"\\s+")


        def _canonicalize(value: str) -> str:
            return _WS_RE.sub(" ", value.strip())


        def parse_citations(value: object) -> list[str]:
            """Return a deduplicated citation list from a string or iterable input."""
            if value is None:
                return []

            if isinstance(value, str):
                raw_items = value.split(";")
            elif isinstance(value, Iterable):
                raw_items = []
                for item in value:
                    if item is None:
                        continue
                    if isinstance(item, str):
                        raw_items.extend(item.split(";"))
                    else:
                        raw_items.append(str(item))
            else:
                if isinstance(value, float) and math.isnan(value):
                    return []
                raw_items = [str(value)]

            citations = []
            seen = set()
            for item in raw_items:
                canon = _canonicalize(item)
                if canon and canon not in seen:
                    citations.append(canon)
                    seen.add(canon)
            return citations


        def citation_f1(predicted: object, gold: object) -> float:
            """Compute set-based F1 for a single query."""
            pred_set = set(parse_citations(predicted))
            gold_set = set(parse_citations(gold))

            if not pred_set and not gold_set:
                return 1.0
            if not pred_set or not gold_set:
                return 0.0

            true_positives = len(pred_set & gold_set)
            precision = true_positives / len(pred_set)
            recall = true_positives / len(gold_set)
            if precision + recall == 0.0:
                return 0.0
            return 2.0 * precision * recall / (precision + recall)


        def score(y_true: Sequence[object], y_pred: Sequence[object]) -> float:
            """Compute citation-level Macro F1. Higher is better."""
            if len(y_true) != len(y_pred):
                raise ValueError(
                    f"Length mismatch: {len(y_true)} ground-truth rows vs {len(y_pred)} predictions"
                )

            if not y_true:
                return 0.0

            return sum(citation_f1(pred, gold) for gold, pred in zip(y_true, y_pred)) / len(y_true)
        '''
    )


def _build_cv_splits(val_size: int) -> dict[str, dict[str, list[int]]]:
    return {"fold_0": {"train": [], "val": list(range(val_size))}}


def _write_text(path: Path, content: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        print(f"Skipping existing file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"Wrote {path}")


def _write_json(path: Path, payload: dict, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        print(f"Skipping existing file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {path}")


def _download_bundle(slug: str, download_dir: Path, task_dir: Path, kaggle_bin: str) -> None:
    if shutil.which(kaggle_bin) is None:
        raise FileNotFoundError(
            f"Kaggle CLI not found: {kaggle_bin}. Install it or pass --kaggle-bin."
        )

    download_dir.mkdir(parents=True, exist_ok=True)
    cmd = [kaggle_bin, "competitions", "download", "-c", slug, "-p", str(download_dir)]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    zip_path = download_dir / f"{slug}.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Expected Kaggle archive at {zip_path}")

    task_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(task_dir)
    print(f"Extracted {zip_path} -> {task_dir}")


def _warn_on_missing_data(task_dir: Path) -> None:
    missing = [name for name in REQUIRED_DATA_FILES if not (task_dir / name).exists()]
    if missing:
        print(
            "Warning: missing competition files in "
            f"{task_dir}: {', '.join(missing)}",
            file=sys.stderr,
        )


def _infer_val_size(task_dir: Path, fallback_val_size: int | None) -> int:
    val_path = task_dir / "val.csv"
    if val_path.exists():
        return len(pd.read_csv(val_path))
    if fallback_val_size is not None:
        return fallback_val_size
    raise FileNotFoundError(
        f"{val_path} is missing. Download the competition data first or pass --val-size."
    )


def scaffold_task(task_dir: Path, overwrite: bool = False, val_size: int | None = None) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    _warn_on_missing_data(task_dir)

    resolved_val_size = _infer_val_size(task_dir, val_size)
    _write_text(task_dir / "description.md", _description_text(), overwrite)
    _write_text(task_dir / "metric.py", _metric_text(), overwrite)
    _write_text(task_dir / "INSTRUCTIONS.md", _instructions_text(), overwrite)
    _write_json(task_dir / "cv_splits.json", _build_cv_splits(resolved_val_size), overwrite)


def main() -> None:
    args = parse_args()

    if args.download:
        _download_bundle(args.slug, args.download_dir, args.task_dir, args.kaggle_bin)

    scaffold_task(args.task_dir, overwrite=args.force, val_size=args.val_size)
    print(f"Task bundle ready at {args.task_dir}")


if __name__ == "__main__":
    main()
