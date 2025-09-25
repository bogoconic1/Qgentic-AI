import logging
import time
import openai
import weave
import os

def _build_directory_listing(root: str, num_files: int = 10) -> str:
    lines: list[str] = []
    for current_root, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d != "outputs")
        rel_root = os.path.relpath(current_root, root)
        depth = 0 if rel_root in (".", "") else rel_root.count(os.sep) + 1
        indent = "    " * depth
        folder_display = "." if rel_root in (".", "") else os.path.basename(rel_root)
        lines.append(f"{indent}{folder_display}/")
        for name in files[:num_files]: # to avoid stuffing context window
            lines.append(f"{indent}    {name}")
        if len(files) > num_files:
            lines.append(f"{indent}    ... ({len(files) - num_files} more files)")
    return "\n".join(lines)

@weave.op()
def call_llm_with_retry(llm_client, max_retries=3, **kwargs):
    """Call LLM with retry logic."""
    for attempt in range(max_retries):
        try:
            return llm_client.chat.completions.create(extra_body={}, **kwargs)
        except openai.InternalServerError as e:
            if "504" in str(e) and attempt < max_retries - 1:
                wait_time = 2**attempt
                print(f"Retry {attempt + 1}/{max_retries} in {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                continue
        except Exception as e:
            print(f"Error calling LLM: {e}")
            continue
    return ""