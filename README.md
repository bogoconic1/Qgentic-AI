# Qgentic-AI

Qgentic-AI is a research & development automation stack aimed at iterating on Kaggle
competitions with minimal human intervention. Two collaborating LLM-driven agents – a
**Researcher** and a **Developer** – take a competition bundle, explore the data, produce a
technical plan, generate code, run it locally, analyse the results, and keep refining the
solution. Guardrails and supporting tools keep the loop grounded, reproducible, and safe.

---
## News

**[2025/10/19]** Added initial study to recommend 5 potential models for experimenting -> executing them in parallel with a defined time limit

**[2025/10/17]** Updated evals for recent SOTA submissions to MLE-Bench repo. Chose competitions where the variation between the latest solutions are high.

**[2025/10/14]** Updated results of ongoing competition evaluation

## Preliminary Results

## Present Competitions

| Kaggle Competition | Public LB score | Notebook |
| --- | --- | --- |
| playground-series-s5e10 | TBC | [Here](https://www.kaggle.com/code/yeoyunsianggeremie/ps5e10-agentic-ai-solution) |

## Past Competitions

| Kaggle Competition | Difficulty | Type | Metric | FM Agent | InternAgent | Operand | R&D-Agent | Qgentic-AI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| us-patent-phrase-to-phrase-matching | Medium | Information Retrieval | PCC (higher) | [0.87466, 0.84211, 0.86829] | [0.86864, 0.8647, 0.87046] | [0.45894, 0.7729, 0.84507] | [0.75307, 0.80521, 0.84449] | TBC |
| learning-agency-lab-automated-essay-scoring-2 | Medium | Text | QWK (higher) | [0.8384, 0.84839, 0.84733] | [0.83995, 0.82771, 0.82221] | [0.83013] | [0.83751, 0.82051, 0.81547] | TBC |
| tabular-playground-series-dec-2021 | Easy | Tabular | Accuracy % (higher) | [0.96137, 0.95822, 0.96004] | [0.96216, 0.96302, 0.96286] | [0.96194, 0.96335, 0.96269] | [0.96295, 0.96312, 0.96276] | TBC |
| statoil-iceberg-classifier-challenge | Medium | Image Classification | Logloss (lower) | [0.71993, 0.6963, 2.35887] | [0.20956, 0.19655, 0.20297] | Failed | Failed | TBC |
| denoising-dirty-documents | Medium | Computer Vision | RMSE (lower) | [0.02779, 0.01785, 0.01311] | [0.0116, 0.01509, 0.0418] | [0.01884, 0.01081, 0.03939] | [0.01221, 0.01135, 0.01009] | TBC |
| whale-categorization-playground | Medium | Computer Vision | MAP@5 (higher) | [0.46374, 0.50367, 0.43165] | [0.31698, 0.19513, 0.03771] | [0.40842, 0.24289, 0.43053] | [0.25507, 0.25628, 0.27506] | TBC |
| google-quest-challenge | Medium | Text | Spearman Correlation (higher) | [0.3863, 0.38017, 0.41448] | [0.39183, 0.41797, 0.4164] | [0.4105, 0.39704, 0.38652] | [0.41739, 0.42004, 0.4072] | TBC |
--- 

## Architecture at a Glance

![Architecture diagram showing the Researcher and Developer flow](docs/assets/architecture_v2.png)

- **Starter Agent (`agents/starter.py`)**
  - Proposes 5 starter model ideas with short example code by referencing the competition description and `docs/state_of_competitions_2024.md`.
  - Persists `starter_suggestions.txt` and `starter_suggestions.json` in `task/<slug>/outputs/<iteration>/`.

- **Researcher Agent (`agents/researcher.py`)**
  - Uses tool-calling (EDA snippets, external dataset search) to understand the task.
  - Logs every step to `task/<slug>/outputs/<iteration>/researcher/`.
  - Persists the final plan in `plan.md` – consumed verbatim by downstream stages.

- **Developer Agent (`agents/developer.py`)**
  - Reads `task/<slug>/outputs/<iteration>/starter_suggestions.json` (five model ideas with example code).
  - Launches five baseline `DeveloperAgent` runs concurrently via `ProcessPoolExecutor` with iteration suffixes
    `"<iteration>_<1..5>"`, each constrained to its model name and example code.
  - Each baseline run returns `(best_score, best_code)`; results are merged into
    `baseline_results.json` under the keys `model_1`..`model_5` as `best_score` and `best_code`.
  - Baseline run time limit is configured in the orchestrator (default 3600s per baseline in this repo).

- **Ensembling Agent (`agents/ensembler.py`)**
  - work in progress

- **Guardrails (`guardrails/`), Tools (`tools/`) & Shared Config (`project_config.py`)**
  - `tools.developer` wraps code execution, stack-trace web search, and SOTA suggestions.
  - `tools.researcher` exposes the EDA runtime and dataset downloader.
  - `config.yaml` overrides project defaults (model endpoints, runtime limits, etc.).

- **Task Bundles (`task/<slug>/`)**
  - Expected layout: Kaggle metadata, `description.md`, `plan.md`, `outputs/<iteration>/`
    (logs, generated code, submissions), baseline artifacts (`baseline_results.json`),
    and per-baseline outputs under `outputs/<iteration>_<k>/`.

# Sample Logs

![Screenshot of wandb run metrics for the pipeline](docs/assets/wandb_sample_log.png)

## Getting Started

### 1. Prerequisites

- Python 3.10+ (the development environment uses 3.11).
- Optional: CUDA-enabled GPU for training scripts that request GPU resources.

```
conda create --name qgentic-ai python=3.12 -y
conda activate qgentic-ai

git clone https://github.com/bogoconic1/Qgentic-AI.git
cd Qgentic-AI
pip install uv
bash install.sh
```

Add your ```kaggle.json``` file in the Qgentic-AI directory

If you want to download MLE-Bench Data for another competition, modify ```install.sh``` ```TASK_NAME``` and only execute ```prepare_data``` and ```copy_task_data```

### 2. Install Dependencies

```bash
pip install vllm
```
This is an additional dependency not in requirements.txt, as running it together with others causes errors.

### 3. Configure API Keys & Environment

Create a `.env` file in the project root (or export directly):

```
GOOGLE_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
EXA_API_KEY=...
OPENROUTER_API_KEY=...
E2B_API_KEY=...
FIRECRAWL_API_KEY=...
```

These keys are loaded via `python-dotenv`. Adjust the environment variables listed in
`config.yaml` if you need custom names or endpoints.

### 4. Download Meta Kaggle and Meta Kaggle Code from Kaggle Datasets (Optional - currently not used)
```
sudo apt-get install unzip
curl -L -o /workspace/meta-kaggle.zip https://www.kaggle.com/api/v1/datasets/download/kaggle/meta-kaggle
curl -L -o /workspace/meta-kaggle-code.zip https://www.kaggle.com/api/v1/datasets/download/kaggle/meta-kaggle-code

unzip meta-kaggle.zip -d /workspace/meta-kaggle
unzip meta-kaggle-code.zip -d /workspace/meta-kaggle-code
```

Then run
```
python create_metadata.py --competition-slug "enter slug"
```

You will see something like this

```
task/
└─ "enter slug"/
   ├─ description.md
   ├─ public_insights.md
   ├─ sample_submission.csv
   ├─ comp_metadata.yaml   
   └─ train files/test files
```

### 5. Launch an Iteration

```bash
python launch_agent.py --slug "enter slug" --iteration 1 --time-seconds $((6*3600))
```

- The Researcher runs first (unless `plan.md` already exists for that iteration).
- Baseline Evaluator: five concurrent Developer runs using starter suggestions; results written to
  `baseline_results.json` and `outputs/<iteration>_<k>/`.
- The main Developer then cycles through code generations, writing artifacts to
  `task/<slug>/outputs/<iteration>/`.
- `submission.csv` (or the configured `submission_{version}.csv`) is produced on success.

### 6. Monitoring & Artefacts

- `researcher.txt` / `developer.txt` capture detailed logs for each iteration.
- `code_{iteration}_v{version}.py` are the generated scripts; corresponding logs sit under
  `code_{iteration}_v{version}.txt`.
- Weights & Biases and Weave projects are initialised in `launch_agent.py`; supply
  `--wandb-entity/--wandb-project`, export `WANDB_ENTITY/WANDB_PROJECT`, or define them
  in `config.yaml` under `tracking.wandb`.

---

## Configuration

Key settings live in `config.yaml` (merged with `project_config.py` defaults):

- **llm**: base URL, API key env var, model IDs for Researcher/Developer and guardrails.
- **runtime**: max steps/tries, retry counts, directory listing depth, patch mode switch.
- **paths**: root directories and naming templates for generated artefacts.
- **guardrails**: toggles for logging order checks, debug/NaN guard, and leakage reviews.

> **Patch Mode (Experimental)** – The developer supports a token-efficient diff workflow.
> Toggle `runtime.patch_mode_enabled: true` to request unified diffs (with line numbers)
> from the model instead of full files. This feature is still being tuned

---

## License

MIT
