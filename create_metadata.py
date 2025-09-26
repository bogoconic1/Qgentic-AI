from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd
import polars as pl  # noqa: F401
import yaml
from dotenv import load_dotenv
from openai import OpenAI
from project_config import get_config
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
tqdm.pandas()

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n
        self.sizes = [1] * n

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u != root_v:
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
                self.sizes[root_u] += self.sizes[root_v]
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
                self.sizes[root_v] += self.sizes[root_u]
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1
                self.sizes[root_u] += self.sizes[root_v]

CONFIG = get_config()
CONFIG_LLM = CONFIG.get("llm", {}) if isinstance(CONFIG, dict) else {}

PROMPT = """You are a Kaggle Competitions Grandmaster tasked with extracting specific pipeline details from provided code. Begin with a concise checklist (3-7 bullets) of what you will do; keep items conceptual, not implementation-level. Extract the following components:

- overall_approach: One concise sentence summarizing the main idea.
- data_preprocessing: Chronological list of raw data cleaning/preparation steps.
- feature_engineering: Chronological list of feature creation steps after preprocessing.
- validation_strategy: One sentence describing the model validation strategy.
- modeling: One sentence summarizing modeling techniques and algorithms used (combine if several).
- post_processing: Ordered list of steps to refine or modify predictions before submission.
- technical_stack: List of libraries used in the notebook.

For missing components, use an empty array ([]) for lists or an empty string ("") for strings.

If information is unclear or more than one approach per component is present, summarize the essence concisely.

Given code:
{code}

# Output Format
Return the result in a JSON code block exactly as shown:

```json
{{
  "overall_approach": "string",
  "data_preprocessing": ["string", ...],
  "feature_engineering": ["string", ...],
  "validation_strategy": "string",
  "modeling": "string",
  "post_processing": ["string", ...],
  "technical_stack": ["string", ...]
}}
```

Maintain the chronological order as seen in the code for process lists. After extraction, validate that each required field is present and formatted correctly according to the schema; self-correct and reformat output if validation fails.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter Kaggle kernels and summarize approaches.")
    parser.add_argument(
        "--meta-kaggle-path",
        type=Path,
        default=Path("/workspace/meta-kaggle"),
        help="Path to the Meta Kaggle dataset directory.",
    )
    parser.add_argument(
        "--meta-kaggle-code-path",
        type=Path,
        default=Path("/workspace/meta-kaggle-code"),
        help="Path to the Meta Kaggle code archive.",
    )
    parser.add_argument(
        "--task-path",
        type=Path,
        default=Path("/workspace/gstar-project/task"),
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
    forum_id = int(row["ForumId"])
    start_ts = pd.to_datetime(row["EnabledDate"], errors="coerce", utc=True)
    cutoff_ts = pd.to_datetime(row["DeadlineDate"], errors="coerce", utc=True)
    if pd.notna(cutoff_ts):
        start_ts = start_ts.tz_convert("UTC")
        cutoff_ts = cutoff_ts.tz_convert("UTC")
    else:
        cutoff_ts = None
    return competition_id, forum_id, start_ts, cutoff_ts


def read_jupyter_notebook(file_path: Path) -> str:
    if file_path.suffix != ".ipynb":
        msg = "File must be a Jupyter notebook."
        raise AssertionError(msg)

    with open(file_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    result = ""
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            result += "".join(cell.get("source", []))
            result += "\n"

    with open(file_path.with_suffix(".md"), "w", encoding="utf-8") as f:
        f.write(result)

    return result


def summarize_code(client: OpenAI, code: str) -> dict:
    messages = [{'role': 'user', 'content': PROMPT.format(code=code)}]
    response = client.chat.completions.create(
        extra_body={},
        model=CONFIG_LLM.get("offline_model", "openai/gpt-5"),
        messages=messages,
    ).choices[0].message.content

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        try:
            pattern = r"```json\s*(.*?)\s*```"
            matches = re.findall(pattern, response, re.DOTALL)
            json_dict = "\n\n".join(matches).strip()
            return json.loads(json_dict)
        except json.JSONDecodeError:
            return "Failed to generate JSON"


def main() -> None:
    args = parse_args()

    load_dotenv()

    task_dir = args.task_path / args.competition_slug
    os.makedirs(task_dir, exist_ok=True)

    client = OpenAI(
        api_key=os.environ.get(CONFIG_LLM.get("api_key_env", "OPENROUTER_API_KEY")),
        base_url=CONFIG_LLM.get("base_url", "https://openrouter.ai/api/v1"),
    )

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

    kvcs = pd.read_csv(
        args.meta_kaggle_path / "KernelVersionCompetitionSources.csv",
        usecols=["Id", "KernelVersionId", "SourceCompetitionId"],
    )
    version_ids = kvcs.loc[kvcs["SourceCompetitionId"] == competition_id, "KernelVersionId"].astype("Int64")

    kernels = pd.read_csv(
        args.meta_kaggle_path / "Kernels.csv",
        usecols=["Id", "CurrentKernelVersionId", "CreationDate", "MadePublicDate", "Medal"],
    )
    kernels = kernels[kernels["CurrentKernelVersionId"].isin(version_ids)]
    kernels = kernels.loc[kernels["Medal"] == 1]
    kernels["CreationDate"] = pd.to_datetime(
        kernels["CreationDate"], format="%m/%d/%Y %H:%M:%S", errors="coerce", utc=True
    )
    kernels["MadePublicDate"] = pd.to_datetime(
        kernels["MadePublicDate"], format="%m/%d/%Y", errors="coerce", utc=True
    )
    kernels = kernels.loc[kernels["CreationDate"] >= start_ts]
    kernels = kernels.loc[
        kernels["MadePublicDate"].notna() & (kernels["MadePublicDate"] <= cutoff_ts)
    ]
    kernels["CurrentKernelVersionId"] = kernels["CurrentKernelVersionId"].astype("Int64")
    print(f"Extracted {len(kernels)} kernels")

    def extract_code(row):
        id_str = str(int(row["CurrentKernelVersionId"]))
        first6 = id_str[:-3]
        first3 = first6[:-3]
        next3 = first6[-3:]
        
        base_path = args.meta_kaggle_code_path / f"{first3}".zfill(4) / f"{next3}" / f"{id_str}"
        
        candidates = [base_path.with_suffix(".ipynb"), base_path.with_suffix(".py")]
        for src in candidates:
            if src.exists():
                dst = task_dir / "codes" / src.name
                os.makedirs(dst.parent, exist_ok=True)
                try:
                    shutil.copy2(src, dst)
                    if src.suffix == ".ipynb":
                        markdown = read_jupyter_notebook(dst)
                    elif src.suffix == ".py":
                        with open(dst, "r", encoding="utf-8") as f:
                            markdown = f.read()
                    os.remove(dst)
                    return markdown
                except FileNotFoundError:
                    return "Code file not found."
        
        return "Notebook or script not found."


    kernels["code"] = kernels.progress_apply(extract_code, axis=1)
    kernels = kernels[kernels["code"] != "Notebook not found"]
    print(f"Filtered {len(kernels)} kernels")

    kernels["code_summary"] = kernels["code"].progress_apply(lambda code: summarize_code(client, code))
    kernels = kernels[kernels["code_summary"] != "Failed to generate JSON"]

    all_kernels_metadata = defaultdict(list)
    for _, row in kernels.iterrows():
        for key, value in row["code_summary"].items():
            if isinstance(value, list):
                value = [v for v in value if v != ""]
                all_kernels_metadata[key].extend(value)
            elif value != "":
                all_kernels_metadata[key].append(value)

    st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")

    clusters = defaultdict(set)
    for key in all_kernels_metadata:
        uf = UnionFind(len(all_kernels_metadata[key]))
        embeddings = st_model.encode(
            all_kernels_metadata[key], convert_to_tensor=True, normalize_embeddings=True
        )
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = (embeddings[i] @ embeddings[j].T).item()
                if sim > 0.8:
                    uf.union(i, j)

        for i in range(len(all_kernels_metadata[key])):
            root = uf.find(i)
            label = f"{all_kernels_metadata[key][root]} ({uf.sizes[root]} recommendations)"
            clusters[key].add(label)

    insights = ""
    for key, values in clusters.items():
        insights += f"### {key.replace('_', ' ').title()}\n"
        for value in values:
            insights += f"- {value}\n"
        insights += "\n"

    insights_path = task_dir / "public_insights.md"
    with open(insights_path, "w", encoding="utf-8") as f:
        f.write(insights)


if __name__ == "__main__":
    main()
