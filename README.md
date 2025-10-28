# Qgentic-AI

Qgentic-AI is a research & development automation stack aimed at iterating on Kaggle
competitions with minimal human intervention. Two collaborating LLM-driven agents – a
**Researcher** and a **Developer** – take a competition bundle, explore the data, produce a
technical plan, generate code, run it locally, analyse the results, and keep refining the
solution. Guardrails and supporting tools keep the loop grounded, reproducible, and safe.

---
## News

**[2025/10/26]** Updated some evals on Qgentic-AI without ensembler agent

**[2025/10/25]** Added post-EDA agent to identify red flags in code/logs/submission and CPU/GPU (NVIDIA MIG) parallelism support

**[2025/10/22]** Added ModelRecommender agent - recommend candidate models, preprocessing/architecture, etc

**[2025/10/17]** Updated evals for recent SOTA submissions to MLE-Bench repo. Chose competitions where the variation between the latest solutions are high.

**[2025/10/14]** Updated results of ongoing competition evaluation

## Preliminary Results

## Present Competitions

| Kaggle Competition | Public LB score | Notebook |
| --- | --- | --- |
| playground-series-s5e10 | TBC | [Here](https://www.kaggle.com/code/yeoyunsianggeremie/ps5e10-agentic-ai-solution) |

## Past Competitions

| Kaggle Competition | Difficulty | Type | Metric | Qgentic-AI (no ensembler) | FM Agent | InternAgent | Operand | R&D-Agent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| us-patent-phrase-to-phrase-matching | Medium | Information Retrieval | PCC (higher) | 0.88072 | 0.86169 ± 0.01725 | 0.86793 ± 0.00294 | 0.69230 ± 0.20529 | 0.80092 ± 0.04586 |
| learning-agency-lab-automated-essay-scoring-2 | Medium | Text | QWK (higher) | 0.84303 ± 0.00719 | 0.84471 ± 0.00549 | 0.82996 ± 0.00908 | 0.83013 | 0.82450 ± 0.01155 |
| tabular-playground-series-dec-2021 | Easy | Tabular | Accuracy % (higher) | 0.96322 | 0.95988 ± 0.00158 | 0.96268 ± 0.00046 | 0.96266 ± 0.00071 | 0.96294 ± 0.00018 |
| statoil-iceberg-classifier-challenge | Medium | Image Classification | Logloss (lower) | 0.19081 | 1.25837 ± 0.95314 | 0.20303 ± 0.00651 | Failed | Failed |
| denoising-dirty-documents | Medium | Computer Vision | RMSE (lower) | TBC | 0.01958 ± 0.00749 | 0.02283 ± 0.01652 | 0.02301 ± 0.01474 | 0.01122 ± 0.00107 |
| whale-categorization-playground | Medium | Computer Vision | MAP@5 (higher) | 0.42885 ± 0.04164 | 0.46635 ± 0.03608 | 0.18327 ± 0.14001 | 0.36061 ± 0.10255 | 0.26214 ± 0.01121 |
| google-quest-challenge | Medium | Text | Spearman Correlation (higher) | TBC | 0.39365 ± 0.01830 | 0.40873 ± 0.01466 | 0.39802 ± 0.01202 | 0.41488 ± 0.00678 |
--- 

## Architecture at a Glance

![Architecture diagram showing the Researcher and Developer flow](docs/assets/architecture_v3.png)

- **Starter Agent (`agents/starter.py`)**
  - Proposes 5 starter model ideas with short example code by referencing the competition description and `docs/state_of_competitions_2024.md`.
  - Persists `starter_suggestions.txt` and `starter_suggestions.json` in `task/<slug>/outputs/<iteration>/`.
  - Uses `gpt-5-mini` for efficient initial exploration.

- **Researcher Agent (`agents/researcher.py`)**
  - Uses tool-calling (EDA snippets, external dataset search) to understand the task.
  - Logs every step to `task/<slug>/outputs/<iteration>/researcher/`.
  - Persists the final plan in `plan.md` – consumed verbatim by downstream stages.

- **Model Recommender Agent (`agents/model_recommender.py`)**
  - Recommends up to 6 suitable models with detailed strategies for preprocessing, architecture, loss functions, hyperparameters, and inference.
  - Splits recommendations into NOW (MUST_HAVE) and LATER (NICE_TO_HAVE) categories for iterative development.
  - Supports fold split strategy recommendations and web search for SOTA techniques.
  - Uses `gpt-5-mini` for cost-effective model recommendations.

- **Developer Agent (`agents/developer.py`)**
  - Implements a two-stage approach for each iteration:
    1. **Stage 1 (Red Flags)**: Uses `search_red_flags()` with EDA tool-calling to identify issues in code/logs/submissions.
    2. **Stage 2 (SOTA Suggestions)**: Uses `search_sota_suggestions()` based on red flags to generate improvements.
  - Tracks both blacklisted ideas (failed strategies) and successful ideas (working strategies) for knowledge accumulation.
  - Supports dynamic resource allocation with CPU affinity and NVIDIA MIG GPU isolation for parallel execution.
  - Each baseline run returns `(best_score, best_code_file, blacklisted_ideas, successful_ideas)`.
  - Results are merged into `baseline_results.json` with full metadata including recommendations and strategy outcomes.

- **Parallel Baseline Execution (`agents/orchestrator.py`)**
  - Launches multiple baseline `DeveloperAgent` runs concurrently using `ThreadPoolExecutor`.
  - **Dynamic Resource Allocation**: Uses Queue-based pools for CPU cores and MIG instances.
    - Processes grab resources when available and return them upon completion.
    - Prevents cyclic pre-assignment issues and resource contention.
  - **CPU Affinity**: Pins each process to specific CPU cores using `psutil` to prevent interference.
  - **NVIDIA MIG Support**: Auto-detects MIG instances and isolates each baseline to a dedicated GPU partition.
  - Configuration in `config.yaml`:
    - `enable_cpu_affinity`: Enable CPU core pinning
    - `enable_mig`: Enable MIG GPU isolation (auto-detects worker count from available MIG instances)
    - `baseline_max_parallel_workers`: Fallback worker count when MIG is disabled

- **Ensembling Agent (`agents/ensembler.py`)**
  - Work in progress

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

- Python 3.12.
- CUDA-enabled GPU.

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

### 4. Download Meta Kaggle from Kaggle Datasets
```
sudo apt-get install unzip
curl -L -o /workspace/meta-kaggle.zip https://www.kaggle.com/api/v1/datasets/download/kaggle/meta-kaggle

unzip meta-kaggle.zip -d /workspace/meta-kaggle
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

- **llm**: Model configurations for different components:
  - `developer_model`: Main Developer agent code generation (`gpt-5`)
  - `developer_tool_model`: Developer tools (red flags, SOTA suggestions, debug) (`gpt-5`)
  - `starter_model`: Starter agent for initial exploration (`gpt-5-mini`)
  - `model_recommender_model`: Model recommendation agent (`gpt-5-mini`)
  - `researcher_model`: Main Researcher agent planning (`gpt-5`)
  - `researcher_tool_offline_model`: EDA tool execution (`gpt-5`)
  - `researcher_tool_online_model`: External dataset search (`gpt-5`)
  - `leakage_review_model` / `leakage_followup_model`: Guardrails (`gpt-5-mini`)

- **runtime**: Execution parameters:
  - `ask_eda_max_attempts`: Max retry attempts for EDA tool (default: 3)
  - `researcher_max_steps`: Max steps for researcher exploration (default: 512)
  - `llm_max_retries`: Max retries for LLM calls (default: 3)
  - `baseline_max_parallel_workers`: Max parallel baseline workers when MIG disabled (default: 3)
  - `enable_mig`: Enable NVIDIA MIG GPU isolation (auto-detects worker count)
  - `enable_cpu_affinity`: Enable CPU core pinning for parallel processes
  - `patch_mode_enabled`: Experimental diff-based workflow (default: false)

- **paths**: Root directories and naming templates for generated artifacts.

- **guardrails**: Toggles for logging order checks, debug/NaN guard, and leakage reviews.

- **model_recommender**: Model recommendation settings:
  - `default_models`: Fallback model list (default: `["deberta-v3-large"]`)
  - `enable_web_search`: Enable web search for SOTA strategies (default: true)

> **Patch Mode (Experimental)** – The developer supports a token-efficient diff workflow.
> Toggle `runtime.patch_mode_enabled: true` to request unified diffs (with line numbers)
> from the model instead of full files. This feature is still being tuned.

> **Parallel Execution** – Configure CPU affinity and MIG GPU isolation for running multiple
> baseline models concurrently. When `enable_mig: true`, the system auto-detects available
> MIG instances and sets worker count accordingly. Otherwise, uses `baseline_max_parallel_workers`.

---

## License

MIT
