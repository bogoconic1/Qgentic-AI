# Qgentic-AI

Qgentic-AI is a research & development automation stack aimed at iterating on Kaggle
competitions with minimal human intervention. Two collaborating LLM-driven agents – a
**Researcher** and a **Developer** – take a competition bundle, explore the data, produce a
technical plan, generate code, run it locally, analyse the results, and keep refining the
solution. Guardrails and supporting tools keep the loop grounded, reproducible, and safe.

---

## Architecture at a Glance

- **Researcher Agent (`agents/researcher.py`)**
  - Uses tool-calling (EDA snippets, external dataset search) to understand the task.
  - Logs every step to `task/<slug>/outputs/<iteration>/researcher.txt`.
  - Persists the final plan in `plan.md` – the Developer consumes this verbatim.

- **Developer Agent (`agents/developer.py`)**
  - Generates a single Python training script per attempt using OpenRouter models.
  - Executes the script, captures logs, scores the output, and iterates up to the
    configured maximum tries.
  - Guardrails enforce logging order, DEBUG→FULL sequencing, NaN detection, and optional
    leakage checks.
  - Integrates with Weave & Weights & Biases for observability.

- **Guardrails (`guardrails/`), Tools (`tools/`) & Shared Config (`project_config.py`)**
  - `tools.developer` wraps code execution, stack-trace web search, and SOTA suggestion
    lookups.
  - `tools.researcher` exposes the EDA runtime and dataset downloader.
  - `config.yaml` overrides project defaults (model endpoints, runtime limits, etc.).

- **Task Bundles (`task/<slug>/`)**
  - Expected layout: Kaggle metadata, `description.md`, `plan.md`, `outputs/<iteration>/`
    (logs, generated code, submissions), and optional external-data caches.

---

## Getting Started

### 1. Prerequisites

- Python 3.10+ (the development environment uses 3.11).
- `git`, `patch`, and system build tools for native dependencies.
- Optional: CUDA-enabled GPU for training scripts that request GPU resources.

### 2. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure API Keys & Environment

Create a `.env` file in the project root (or export directly):

```
GOOGLE_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
EXA_API_KEY=...
OPENROUTER_API_KEY=...
E2B_API_KEY=...
```

These keys are loaded via `python-dotenv`. Adjust the environment variables listed in
`config.yaml` if you need custom names or endpoints.

### 4. Organise a Competition Bundle

Place each competition under `task/<slug>/`. A minimal bundle looks like:

```
task/
└─ statoil-iceberg-classifier-challenge/
   ├─ description.md
   ├─ plan.md               # (optional – generated if missing)
   ├─ sample_submission.csv # for baseline grading
   ├─ comp_metadata.yaml    # created by create_metadata.py
   ├─ outputs/
   │  └─ <iteration>/       # logs, generated code, submissions
   └─ external-data/        # optional downloaded datasets
```

Use `create_metadata.py` if you need to rebuild `comp_metadata.yaml` from Meta Kaggle.

### 5. Launch an Iteration

```bash
python launch_agent.py --slug statoil-iceberg-classifier-challenge --iteration 10 --tries 20
```

- The Researcher runs first (unless `plan.md` already exists for that iteration).
- The Developer then cycles through code generations, writing artefacts to
  `task/<slug>/outputs/<iteration>/`.
- `submission.csv` (or the configured `submission_{version}.csv`) is produced on success.

### 6. Monitoring & Artefacts

- `researcher.txt` / `developer.txt` capture detailed logs for each iteration.
- `code_{iteration}_v{version}.py` are the generated scripts; corresponding logs sit under
  `code_{iteration}_v{version}.txt`.
- Weights & Biases and Weave projects are initialised in `launch_agent.py`; set the
  respective API keys or disable instrumentation as needed.

---

## Configuration

Key settings live in `config.yaml` (merged with `project_config.py` defaults):

- **llm**: base URL, API key env var, model IDs for Researcher/Developer and guardrails.
- **runtime**: max steps/tries, retry counts, directory listing depth, patch mode switch.
- **paths**: root directories and naming templates for generated artefacts.
- **guardrails**: toggles for logging order checks, NaN guard, and leakage reviews.

> **Patch Mode (Experimental)** – The developer supports a token-efficient diff workflow.
> Toggle `runtime.patch_mode_enabled: true` to request unified diffs (with line numbers)
> from the model instead of full files. This feature is still being tuned; enable only if
> you are comfortable debugging occasional patch failures.

---

## Development & Testing

- `test_patch.py` – exercises patch application via `DeveloperAgent._apply_patch()`.
- `test_download_dataset.py` – validates the researcher’s external dataset download tool.
- `test_list_directory.py` – quick check of directory listings used in prompts.
- `test_sota_stack_trace.py`, `test_weave.py` – demonstrate tooling integrations.

Run any of the scripts directly with the virtualenv activated:

```bash
python test_patch.py
```

When modifying agents or tools, favour small, focused tests (either scripts like above or
bespoke notebooks) to verify changes end-to-end.

---

## Tips & Best Practices

- Keep `plan.md` and `outputs/<iteration>/` under version control to audit agent progress.
- Reset or archive `outputs` before re-running an iteration to avoid stale artefacts.
- If you introduce new guardrails or tools, wire them through `project_config.py` so they
  can be toggled without code changes.
- For manual experiments, you can reuse the generated scripts within the task directory –
  just remember to log results back into the outputs folder for traceability.

---

## License

This repository is provided as-is for experimentation with agentic Kaggle workflows.
Please review any third-party model or dataset licenses before usage.
