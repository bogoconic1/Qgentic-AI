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
    outputs_dirname = path_cfg.get("outputs_dirname", "outputs")

    for current_root, dirs, files in os.walk(root):
        rel_root = os.path.relpath(current_root, root)

        # Determine traversal policy for the outputs tree: allow only external_data_*/ contents
        segments = [] if rel_root in (".", "") else rel_root.split(os.sep)
        files_to_show = files

        if segments and segments[0] == outputs_dirname:
            # At <root>/outputs
            if len(segments) == 1:
                # Only descend into iteration directories that are numeric; hide files at this level
                dirs[:] = sorted([d for d in dirs if d.isdigit()])
                files_to_show = []
            # At <root>/outputs/<iteration>
            elif len(segments) == 2:
                # Only descend into external_data_* subdirectories; hide files at this level
                dirs[:] = sorted([d for d in dirs if d.startswith("external_data_")])
                files_to_show = []
            else:
                # At or below <root>/outputs/<iteration>/*
                # Allow full traversal within external_data_*; block others entirely
                if len(segments) >= 3 and segments[2].startswith("external_data_"):
                    dirs[:] = sorted(dirs)
                    files_to_show = files
                else:
                    dirs[:] = []
                    files_to_show = []
        else:
            # Outside outputs subtree: normal traversal
            dirs[:] = sorted(dirs)
            files_to_show = files

        depth = 0 if rel_root in (".", "") else rel_root.count(os.sep) + 1
        indent = "    " * depth
        folder_display = "." if rel_root in (".", "") else os.path.basename(rel_root)
        lines.append(f"{indent}{folder_display}/")

        # Print files according to policy and limit
        for name in files_to_show[:limit]:  # to avoid stuffing context window
            lines.append(f"{indent}    {name}")
        if len(files_to_show) > limit:
            lines.append(f"{indent}    ... ({len(files_to_show) - limit} more files)")

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
