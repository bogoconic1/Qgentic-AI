import logging
import os
import time

import openai
import weave

from project_config import get_config

def _build_directory_listing(root: str, num_files: int | None = None) -> str:
    cfg = get_config()
    runtime_cfg = cfg.get("runtime", {}) if isinstance(cfg, dict) else {}
    path_cfg = cfg.get("paths", {}) if isinstance(cfg, dict) else {}
    limit = num_files if num_files is not None else runtime_cfg.get("directory_listing_max_files", 10)
    if limit is None:
        limit = 10
    else:
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 10
    lines: list[str] = []
    ignored_dir = path_cfg.get("outputs_dirname", "outputs")
    for current_root, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d != ignored_dir)
        rel_root = os.path.relpath(current_root, root)
        depth = 0 if rel_root in (".", "") else rel_root.count(os.sep) + 1
        indent = "    " * depth
        folder_display = "." if rel_root in (".", "") else os.path.basename(rel_root)
        lines.append(f"{indent}{folder_display}/")
        for name in files[:limit]:  # to avoid stuffing context window
            lines.append(f"{indent}    {name}")
        if len(files) > limit:
            lines.append(f"{indent}    ... ({len(files) - limit} more files)")
    return "\n".join(lines)

@weave.op()
def call_llm_with_retry(llm_client, max_retries: int | None = None, **kwargs):
    """Call LLM with retry logic."""
    cfg = get_config()
    runtime_cfg = cfg.get("runtime", {}) if isinstance(cfg, dict) else {}
    retries = max_retries or runtime_cfg.get("llm_max_retries", 3)
    try:
        retries = int(retries)
    except (TypeError, ValueError):
        retries = 3
    retries = max(retries, 1)
    for attempt in range(retries):
        try:
            return llm_client.chat.completions.create(extra_body={}, **kwargs)
        except openai.InternalServerError as e:
            if "504" in str(e) and attempt < retries - 1:
                wait_time = 2**attempt
                print(f"Retry {attempt + 1}/{retries} in {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                continue
        except Exception as e:
            print(f"Error calling LLM: {e}")
            continue
    return ""
