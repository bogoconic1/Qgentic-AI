from __future__ import annotations


def get_hard_constraints(
    model_name: str | None = None,
) -> str:
    lines = ["**Hard Constraints:**"]

    if model_name:
        lines.append(
            f"- Use ONLY `{model_name}`. Do not swap the model; you may update preprocessing, postprocessing, or hyperparameters."
        )

    lines.append(
        """- **DO NOT** explicitly set `os.environ['CUDA_VISIBLE_DEVICES']`.
- Place `logging.basicConfig()` at the start of the script.
- Log validation results (per fold and overall), model loading, train/test set size, and any computed quantities that can go wrong (e.g. class weights, thresholds).
- Log final validation results, best epoch number, total training time, and submission prediction distribution.
- Do not use `try/except` to suppress errors.
- External datasets: may be appended **only** to training set.
- If asked to download external datasets, use kagglehub:
```
import kagglehub
path = kagglehub.dataset_download("<author>/<dataset_name>")
```"""
    )

    return "\n".join(lines)
