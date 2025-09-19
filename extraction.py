"""
Metadata extraction utilities for Kaggle Meta datasets (Pandas-only).

This module provides a `MetadataExtractor` class that can:
  - Export Kaggle forum discussions for a competition slug to Markdown files
  - Extract notebook ids (Kernels.Id) for a competition, created on/before the deadline

For each forum topic under the target competition, a Markdown file is generated at:
    <output_root>/task/<competition_slug>/discussions/<ForumTopicId>.md

Assumptions (no edge cases handled):
  - Required columns exist in all parquet files
  - Kernels.MadePublicDate uses the format "%m/%d/%Y" (e.g., "04/19/2016")
  - All timestamps are compared in UTC

Example CLI usage:
    python extraction.py --slug <competition-slug>
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import logging

import pandas as pd
import json
import shutil
import os


logger = logging.getLogger(__name__)


class MetadataExtractor:
    """Extract and export forum discussions for a competition.

    Parameters
    ----------
    competitions_path : str | Path
        Path to `Competitions.parquet`.
    topics_path : str | Path
        Path to `ForumTopics.parquet`.
    messages_path : str | Path
        Path to `ForumMessages.parquet`.
    output_root : str | Path, optional
        Root directory where the `<slug>/` folder will be created. Defaults
        to current working directory.
    indent_spaces : int, optional
        Number of spaces per reply depth level. Defaults to 4.
    sort_by : str, optional
        Field used for chronological ordering within each reply group.
        Typically `PostDate` (default) or `Id`.
    """

    def __init__(
        self,
        competitions_path: Path | str = "Competitions.parquet",
        topics_path: Path | str = "ForumTopics.parquet",
        messages_path: Path | str = "ForumMessages.parquet",
        kernel_version_competition_sources_path: Path | str = "KernelVersionCompetitionSources.parquet",
        kernels_path: Path | str = "Kernels.parquet",
        output_root: Path | str = ".",
        indent_spaces: int = 4,
        sort_by: str = "PostDate",
    ) -> None:
        self.competitions_path = Path(competitions_path)
        self.topics_path = Path(topics_path)
        self.messages_path = Path(messages_path)
        self.kernel_version_competition_sources_path = Path(kernel_version_competition_sources_path)
        self.kernels_path = Path(kernels_path)
        self.output_root = Path(output_root)
        self.indent_spaces = int(indent_spaces)
        self.sort_by = sort_by
        # Formatting toggles (default LLM-friendly):
        self.include_front_matter: bool = True
        self.include_user: bool = True
        self.include_date: bool = True

    # --------------------------- Public API ---------------------------
    def extract_for_competition(self, slug: str) -> int:
        """Export all forum topics for a competition slug.

        Creates `<slug>/<ForumTopicId>.md` files under `output_root`.

        Returns
        -------
        int
            The number of topic files written.
        """
        competition_id, forum_id, cutoff_ts = self._get_competition_and_forum_ids(slug)
        cutoff_iso = (
            cutoff_ts.strftime("%Y-%m-%dT%H:%M:%SZ") if cutoff_ts is not None else None
        )
        logger.info(
            "Resolved slug '%s' to competition_id=%s forum_id=%s cutoff=%s",
            slug,
            competition_id,
            forum_id,
            cutoff_iso,
        )

        self.export_notebook_codes(slug)

        topics = list(self._iter_topics(forum_id))
        if not topics:
            logger.warning("No topics found for forum_id=%s (slug=%s)", forum_id, slug)
            return 0

        out_dir = self.output_root / "task" / slug / "discussions"
        out_dir.mkdir(parents=True, exist_ok=True)

        num_written = 0
        for topic_id, title, created_at in topics:
            try:
                # Skip topics created strictly after the competition deadline
                if cutoff_ts is not None and created_at:
                    created_ts = pd.to_datetime(created_at, errors="coerce", utc=True)
                    if pd.notna(created_ts) and created_ts.tz_convert("UTC") > cutoff_ts:
                        logger.info(
                            "Skipping topic %s created at %s after cutoff %s",
                            topic_id,
                            created_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                            cutoff_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        )
                        continue
                df = self._load_messages_for_topic(topic_id)
                # Exclude messages after competition cutoff (if available)
                if cutoff_ts is not None and not df.empty and "PostDate" in df.columns:
                    mask = df["PostDate"].isna() | (df["PostDate"] <= cutoff_ts)
                    df = df.loc[mask].copy()
                content = self._format_markdown(
                    df=df,
                    topic_title=title,
                    slug=slug,
                    topic_id=topic_id,
                    forum_id=forum_id,
                    competition_id=competition_id,
                    topic_created_at=created_at,
                    competition_cutoff_at=cutoff_iso,
                )
                # Compute comment count (exclude the earliest root as OP)
                comment_count = 0
                if not df.empty:
                    # Identify roots
                    root_ids = [mid for mid in df["Id"].tolist() if pd.isna(df.set_index("Id").loc[mid, "ReplyToForumMessageId"]) ]
                    # Pick OP as the earliest root by existing sort order
                    op_id = root_ids[0] if root_ids else None
                    # Count all messages except OP
                    total_msgs = int(df.shape[0])
                    comment_count = total_msgs - (1 if op_id is not None else 0)
                if comment_count < 10:
                    logger.info("Skipping topic %s due to insufficient comments (%s < 10)", topic_id, comment_count)
                    continue
                self._write_topic_file(out_dir, topic_id, content)
                num_written += 1
                logger.info("Wrote topic %s (%s)", topic_id, title)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Failed to process topic_id=%s: %s", topic_id, exc)
        return num_written

    # --------------------------- Data Loading ---------------------------
    def _get_competition_and_forum_ids(self, slug: str) -> Tuple[int, int, Optional[pd.Timestamp]]:
        """Resolve competition id, forum id, and deadline (UTC) for a slug using Pandas."""
        cols = ["Id", "ForumId", "Slug", "DeadlineDate"]
        df = pd.read_parquet(self.competitions_path, columns=cols)
        row = df.loc[df["Slug"] == slug].iloc[0]
        competition_id = int(row["Id"]) if pd.notna(row["Id"]) else -1
        forum_id = int(row["ForumId"])
        cutoff_ts = pd.to_datetime(row["DeadlineDate"], errors="coerce", utc=True)
        if pd.notna(cutoff_ts):
            cutoff_ts = cutoff_ts.tz_convert("UTC")
        else:
            cutoff_ts = None
        return competition_id, forum_id, cutoff_ts

    def _iter_topics(self, forum_id: int) -> Iterable[Tuple[int, str, Optional[str]]]:
        """Yield (topic_id, title, created_at) for topics under the forum.

        Topics are loaded using predicate pushdown on `ForumId` and we only
        materialize the `Id`, `Title`, and `CreationDate` columns.
        """
        cols = ["Id", "Title", "CreationDate", "ForumId"]
        df = pd.read_parquet(self.topics_path, columns=cols)
        df = df.loc[df["ForumId"] == int(forum_id), ["Id", "Title", "CreationDate"]]
        for row in df.itertuples(index=False):
            yield int(row.Id), (row.Title or "").strip(), row.CreationDate

    def _load_messages_for_topic(self, topic_id: int) -> pd.DataFrame:
        """Load messages for a topic and prepare them for tree rendering.

        Returns a DataFrame with at least the following columns:
            - Id (int)
            - ReplyToForumMessageId (nullable int)
            - RawMarkdown (str, possibly empty)
            - PostDate (datetime64[ns, UTC] or NaT)
        """
        cols = [
            "Id",
            "ForumTopicId",
            "ReplyToForumMessageId",
            "RawMarkdown",
            "PostDate",
            "PostUserId",
        ]
        df = pd.read_parquet(self.messages_path, columns=cols)
        df = df.loc[df["ForumTopicId"] == int(topic_id)].copy()

        df["Id"] = pd.to_numeric(df["Id"], errors="coerce").astype("Int64")
        df["ReplyToForumMessageId"] = pd.to_numeric(df["ReplyToForumMessageId"], errors="coerce").astype("Int64")
        df["PostUserId"] = pd.to_numeric(df["PostUserId"], errors="coerce").astype("Int64")
        df["RawMarkdown"] = df["RawMarkdown"].astype("string").fillna("")
        df["PostDate"] = pd.to_datetime(df["PostDate"], errors="coerce", utc=True).dt.tz_convert("UTC")

        sort_cols: List[str] = [self.sort_by if self.sort_by in df.columns else "PostDate", "Id"]
        df = df.sort_values(sort_cols, kind="mergesort")

        return df

    # --------------------------- Rendering ---------------------------
    def _format_markdown(
        self,
        df: pd.DataFrame,
        topic_title: str,
        slug: str,
        topic_id: int,
        forum_id: int,
        competition_id: int,
        topic_created_at: Optional[str],
        competition_cutoff_at: Optional[str],
    ) -> str:
        """Render a topic's messages into Markdown with front matter and threaded list."""
        title = (topic_title or "").replace("\r", " ").replace("\n", " ").strip()
        if not title:
            # Fallback will be applied by caller if needed; keep non-empty here
            title = "Untitled Topic"

        # Prepare front matter metadata
        # Topic creation date to ISO8601 Z if possible
        created_iso = None
        if topic_created_at:
            try:
                ts = pd.to_datetime(topic_created_at, errors="coerce", utc=True)
                if pd.notna(ts):
                    created_iso = ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
            except Exception:  # noqa: BLE001
                created_iso = None

        message_count = int(df.shape[0]) if df is not None else 0

        lines: List[str] = []

        if self.include_front_matter:
            lines.extend(
                [
                    "---",
                    "schema_version: 1",
                    f"slug: {slug}",
                    f"competition_id: {competition_id}",
                    f"forum_id: {forum_id}",
                    f"topic_id: {topic_id}",
                    f"title: \"{title.replace('\\', '\\\\').replace('\"', '\\\"')}\"",
                    f"created_at: \"{created_iso if created_iso is not None else 'null'}\"",
                    f"cutoff_at: \"{competition_cutoff_at if competition_cutoff_at is not None else 'null'}\"",
                    f"message_count: {message_count}",
                    "---",
                    "",
                ]
            )

        # Title
        lines.extend([f"# {title}", ""])

        if df.empty:
            return "\n".join(lines) + "\n"

        # Build indices for tree traversal
        order_map: Dict[int, int] = {int(mid): idx for idx, mid in enumerate(df["Id"].tolist())}

        children_by_parent: Dict[Optional[int], List[int]] = {}
        for row in df.itertuples(index=False):
            msg_id = int(row.Id) if row.Id is not pd.NA else None
            parent_val = row.ReplyToForumMessageId
            if parent_val is pd.NA or parent_val is None:
                parent_id: Optional[int] = None
            else:
                try:
                    parent_id = int(parent_val)
                except Exception:  # noqa: BLE001
                    parent_id = None

            # If parent not present in current topic's messages, attach to root
            if parent_id is not None and parent_id not in order_map:
                parent_id = None

            children_by_parent.setdefault(parent_id, []).append(int(msg_id))

        # Sort children according to the stable order_map
        for key in list(children_by_parent.keys()):
            children_by_parent[key].sort(key=lambda mid: order_map.get(mid, 0))

        # Message text and metadata lookup
        id_to_text: Dict[int, str] = {
            int(mid): (text if isinstance(text, str) else str(text) if pd.notna(text) else "")
            for mid, text in zip(df["Id"].tolist(), df["RawMarkdown"].tolist())
        }
        id_to_user: Dict[int, Optional[int]] = {
            int(mid): (int(uid) if pd.notna(uid) else None)
            for mid, uid in zip(df["Id"].tolist(), df.get("PostUserId", pd.Series([pd.NA]*len(df))).tolist())
        }
        # Convert message dates to ISO8601 Z
        def to_iso_z(val: object) -> Optional[str]:
            try:
                if pd.isna(val):
                    return None
                if hasattr(val, "tzinfo"):
                    ts = val.tz_convert("UTC") if getattr(val, "tzinfo", None) is not None else val.tz_localize("UTC")
                else:
                    ts = pd.to_datetime(val, errors="coerce", utc=True)
                if pd.isna(ts):
                    return None
                return ts.strftime("%Y-%m-%dT%H:%M:%SZ")
            except Exception:
                return None

        id_to_date: Dict[int, Optional[str]] = {
            int(mid): to_iso_z(dt)
            for mid, dt in zip(df["Id"].tolist(), df["PostDate"].tolist())
        }

        def render_message(message_id: int, depth: int) -> None:
            indent = " " * (self.indent_spaces * depth)
            cont_indent = " " * (self.indent_spaces * depth + 2)
            raw = id_to_text.get(message_id, "")
            # Preserve message content verbatim; only add bullets/indentation
            msg_lines = raw.splitlines() or [""]
            first = msg_lines[0]
            # Build bracketed metadata
            parent_candidates = [pid for pid, kids in children_by_parent.items() if message_id in kids]
            parent_id = parent_candidates[0] if parent_candidates else None
            parent_repr = str(parent_id) if parent_id is not None else "null"
            tokens: List[str] = [f"id:{message_id}", f"parent:{parent_repr}"]
            if self.include_user:
                user_val = id_to_user.get(message_id)
                tokens.append(f"user:{user_val if user_val is not None else 'null'}")
            if self.include_date:
                date_val = id_to_date.get(message_id)
                tokens.append(f"date:{date_val if date_val is not None else 'null'}")
            prefix = "[" + " ".join(tokens) + "]"
            lines.append(f"{indent}- {prefix} {first}")
            for cont in msg_lines[1:]:
                lines.append(f"{cont_indent}{cont}")

        def walk(parent_id: Optional[int], depth: int) -> None:
            for child_id in children_by_parent.get(parent_id, []):
                render_message(child_id, depth)
                walk(child_id, depth + 1)

        walk(None, 0)

        # Trailing newline for POSIX-friendly files
        return "\n".join(lines) + "\n"

    # --------------------------- Output ---------------------------
    def _write_topic_file(self, out_dir: Path, topic_id: int, content: str) -> None:
        out_path = out_dir / f"{int(topic_id)}.md"
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)

    # --------------------------- Notebook extraction ---------------------------
    def get_notebook_ids_before_deadline(
        self,
        *,
        slug: Optional[str] = None,
        competition_id: Optional[int] = None,
    ) -> List[int]:
        """Return unique notebook ids (Kernels.Id) created on/before the competition deadline.

        Assumes required columns exist and MadePublicDate format is "%m/%d/%Y".
        Uses SourceCompetitionId and Id from KernelVersionCompetitionSources (version ids),
        and Id, CurrentKernelVersionId, MadePublicDate, Medal from Kernels.
        """

        # Resolve competition id and deadline
        if slug is not None:
            comp_id, _forum_id, cutoff_ts = self._get_competition_and_forum_ids(slug)
            target_competition_id = int(comp_id)
        else:
            target_competition_id = int(competition_id)  # type: ignore[arg-type]
            comp_df = pd.read_parquet(self.competitions_path, columns=["Id", "DeadlineDate"])
            cutoff_val = comp_df.loc[comp_df["Id"] == target_competition_id, "DeadlineDate"].iloc[0]
            cutoff_ts = pd.to_datetime(cutoff_val, errors="coerce", utc=True)
            cutoff_ts = cutoff_ts.tz_convert("UTC") if pd.notna(cutoff_ts) else None

        # Version ids for this competition
        kvcs = pd.read_parquet(
            self.kernel_version_competition_sources_path,
            columns=["Id", "KernelVersionId", "SourceCompetitionId"],
        ).copy()
        version_ids = kvcs.loc[
            kvcs["SourceCompetitionId"] == target_competition_id, "KernelVersionId"
        ].astype("Int64")

        # Join to Kernels by CurrentKernelVersionId and filter by Medal and MadePublicDate <= cutoff
        kernels = pd.read_parquet(
            self.kernels_path,
            columns=["Id", "CurrentKernelVersionId", "MadePublicDate", "Medal"],
        ).copy()
        kernels = kernels[kernels["CurrentKernelVersionId"].isin(version_ids)]
        kernels = kernels.loc[kernels["Medal"] == 1]
        kernels["MadePublicDate"] = pd.to_datetime(
            kernels["MadePublicDate"], format="%m/%d/%Y", errors="coerce", utc=True
        )
        if cutoff_ts is not None:
            kernels = kernels.loc[
                kernels["MadePublicDate"].notna() & (kernels["MadePublicDate"] <= cutoff_ts)
            ]

        notebook_ids = sorted(
            {int(x) for x in pd.to_numeric(kernels["CurrentKernelVersionId"], errors="coerce").dropna().astype(int).tolist()}
        )
        logger.info(
            "Found %s unique medalized notebook ids for competition_id=%s on/before cutoff %s",
            len(notebook_ids),
            target_competition_id,
            cutoff_ts.strftime("%Y-%m-%dT%H:%M:%SZ") if cutoff_ts is not None else "null",
        )
        return notebook_ids

    # --------------------------- Notebook files ---------------------------
    def export_notebook_codes(
        self,
        slug: str,
        datasets_root: Path | str = "../../Downloads/datasets",
    ) -> Tuple[int, int]:
        """Copy .ipynb files for the competition's IDs and convert to Markdown.

        For each ID, builds source path: `<datasets_root>/0{first3}/{next3}/{ID}.ipynb`,
        where first3/next3 come from the first 6 digits of the ID.

        Copies to `<output_root>/task/<slug>/codes/` and writes a sibling `.md`.

        Returns a tuple: (num_copied, num_converted).
        """
        ids = self.get_notebook_ids_before_deadline(slug=slug)
        out_dir = self.output_root / "task" / slug / "codes"
        out_dir.mkdir(parents=True, exist_ok=True)

        num_copied = 0
        num_converted = 0
        root = Path(datasets_root)
        print(ids)

        for id_val in ids:
            id_str = str(int(id_val))
            first6 = id_str[:6]
            first3 = first6[:3]
            next3 = first6[3:6]
            src = root / f"0{first3}" / f"{next3}" / f"{id_str}.ipynb"
            dst = out_dir / f"{id_str}.ipynb"
            try:
                shutil.copy2(src, dst)
                num_copied += 1
                read_jupyter_notebook(dst)
                os.remove(dst)
                num_converted += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to copy/convert %s: %s", id_str, exc)

        return num_copied, num_converted


def _default_paths(root: Path) -> Tuple[Path, Path, Path, Path, Path]:
    return (
        root / "Competitions.parquet",
        root / "ForumTopics.parquet",
        root / "ForumMessages.parquet",
        root / "KernelVersionCompetitionSources.parquet",
        root / "Kernels.parquet",
    )


def _build_extractor_from_defaults(
    output_root: Path | str = ".",
    indent_spaces: int = 4,
    sort_by: str = "PostDate",
) -> MetadataExtractor:
    cwd = Path.cwd()
    competitions, topics, messages, kvcs, kernels = _default_paths(cwd)
    return MetadataExtractor(
        competitions_path=competitions,
        topics_path=topics,
        messages_path=messages,
        kernel_version_competition_sources_path=kvcs,
        kernels_path=kernels,
        output_root=output_root,
        indent_spaces=indent_spaces,
        sort_by=sort_by,
    )


# --------------------------- Utilities ---------------------------
def read_jupyter_notebook(file_path: Path) -> str:
    """Read code and markdown cells from a Jupyter notebook and write .md alongside."""
    assert file_path.suffix == ".ipynb", "File must be a Jupyter notebook."

    with open(file_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    result = ""
    last_cell_type = "markdown"
    for cell in notebook["cells"]:
        if last_cell_type != cell["cell_type"]:
            result += "\n```\n"
        else:
            result += "\n"
        if cell["cell_type"] == "code":
            result += "".join(cell.get("source", []))
        elif cell["cell_type"] == "markdown":
            result += "".join(cell.get("source", []))
        last_cell_type = cell["cell_type"]
    if last_cell_type == "code":
        result += "\n```\n"

    with open(file_path.with_suffix(".md"), "w", encoding="utf-8") as f:
        f.write(result)

    logger.info("Converted notebook %s to Markdown", file_path.name)
    return result


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(
        description="Export Kaggle forum topic discussions to Markdown files."
    )
    parser.add_argument("--slug", required=True, help="Competition slug (e.g., 'titanic')")
    parser.add_argument(
        "--output-root",
        default=".",
        help="Directory where the <slug>/ folder will be created (default: current dir)",
    )
    parser.add_argument(
        "--indent",
        dest="indent_spaces",
        type=int,
        default=4,
        help="Spaces per indentation level (default: 4)",
    )
    parser.add_argument(
        "--sort-by",
        choices=["PostDate", "Id"],
        default="PostDate",
        help="Ordering within reply groups (default: PostDate)",
    )
    parser.add_argument(
        "--competitions",
        default=None,
        help="Path to Competitions.parquet (default: <cwd>/Competitions.parquet)",
    )
    parser.add_argument(
        "--topics",
        default=None,
        help="Path to ForumTopics.parquet (default: <cwd>/ForumTopics.parquet)",
    )
    parser.add_argument(
        "--messages",
        default=None,
        help="Path to ForumMessages.parquet (default: <cwd>/ForumMessages.parquet)",
    )
    parser.add_argument(
        "--kernel-version-competition-sources",
        default=None,
        help="Path to KernelVersionCompetitionSources.parquet (default: <cwd>/KernelVersionCompetitionSources.parquet)",
    )
    parser.add_argument(
        "--kernels",
        default=None,
        help="Path to Kernels.parquet (default: <cwd>/Kernels.parquet)",
    )
    parser.add_argument(
        "--no-front-matter",
        action="store_true",
        help="Disable YAML front matter at the top of each file",
    )
    parser.add_argument(
        "--no-user",
        action="store_true",
        help="Omit user id from bracketed metadata tokens",
    )
    parser.add_argument(
        "--no-date",
        action="store_true",
        help="Omit ISO8601 date from bracketed metadata tokens",
    )

    args = parser.parse_args()

    if args.competitions and args.topics and args.messages and args.kernel_version_competition_sources and args.kernels:
        extractor = MetadataExtractor(
            competitions_path=args.competitions,
            topics_path=args.topics,
            messages_path=args.messages,
            kernel_version_competition_sources_path=args.kernel_version_competition_sources,
            kernels_path=args.kernels,
            output_root=args.output_root,
            indent_spaces=args.indent_spaces,
            sort_by=args.sort_by,
        )
    else:
        extractor = _build_extractor_from_defaults(
            output_root=args.output_root,
            indent_spaces=args.indent_spaces,
            sort_by=args.sort_by,
        )

    # Apply CLI toggles
    extractor.include_front_matter = not args.no_front_matter
    extractor.include_user = not args.no_user
    extractor.include_date = not args.no_date

    written = extractor.extract_for_competition(args.slug)
    logger.info("Exported %s topic file(s) to '%s'", written, Path(args.output_root) / args.slug)
