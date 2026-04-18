# Qgentic-AI

Qgentic-AI is an automated ML engineering stack. LLM-driven agents take a problem description, produce a technical plan, generate code, run it locally, analyse the results, and keep refining the solution. It supports two modes:

- **Competition mode** — a Researcher and Developer agent pair iterate on Kaggle-style competitions with CV metrics.
- **Goal mode** — a standalone generate-execute-review loop iterates toward a free-form goal described in `GOAL.md` (no metric required).

## Problem Statement

> "If you can solve your own problem, it's much more likely you're solving the problem for others." - The engineers of Claude Code

I'm 2 golds away from Kaggle Competitions Grandmaster. A gold medal means finishing in the top ~1% against thousands of competitors -- many of whom are full-time ML engineers and PhD researchers dedicating weeks to a single competition. But working 6 days a week makes it extremely difficult to put together a top-notch solution. Gold-Medal performance usually requires **200+ hours of investment**.

Most of that time goes to repetitive "maintenance" tasks with diminishing educational value:
- Checking intermediate training results.
- Debugging crashed runs.
- Iterating endlessly on the same model family.
- Running endless evaluations.

A UCI study showed it takes ~23 minutes to regain focus after an interruption. Constantly context-switching between my job and my models was silently destroying my productivity.

I won a solo gold during a period where I was unemployed and could dedicate full weeks to a single competition. That experience made one thing clear: the iteration work is automatable, but having the free time to do it manually is not sustainable. Qgentic-AI was born so that the human effort for a top solution drops from 200+ hours to 20. The agent iterates on weekdays; I chime in on the weekends.

## Results

[Kaggle Writeup](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/writeups/25th-post-training-qwen2-5-32b-and-72b-with-gemi)

| Kaggle Competition | LB Score | Ranking |
| --- | --- | --- |
| deep-past-initiative-machine-translation | **38.6113** | **Silver Medal Top 1% (24/2673)** |
| csiro-biomass | **0.63772** | **Silver Medal Top 1% (32/3802)** |

---

## Getting Started

### 1. Prerequisites

- Python 3.12
- CUDA-enabled GPU

```
conda create --name qgentic-ai python=3.12 -y
conda activate qgentic-ai

git clone https://github.com/bogoconic1/Qgentic-AI.git
cd Qgentic-AI
pip install uv
uv pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
FIRECRAWL_API_KEY=...
HF_TOKEN=...
GOOGLE_CLOUD_PROJECT=...
GOOGLE_CLOUD_LOCATION=global
GOOGLE_GENAI_USE_VERTEXAI=True
KAGGLE_USERNAME=
KAGGLE_KEY=
```

### 3. Download Meta Kaggle

```
sudo apt-get install unzip
curl -L -o /workspace/meta-kaggle.zip https://www.kaggle.com/api/v1/datasets/download/kaggle/meta-kaggle
unzip meta-kaggle.zip -d /workspace/meta-kaggle
```

Then run:
```
python create_metadata.py --competition-slug "enter slug"
```

---

## Competition Mode

The original Kaggle pipeline: Researcher + Developer agents iterate on a competition with a CV metric.

### Create Required Files

Before running, create these files in `task/<slug>/`:

- **`description.md`**: Competition description and evaluation criteria
- **`cv_splits.json`**: Cross-validation fold indices
- **`metric.py`**: Competition-specific evaluation metric

Optionally, add Human-In-The-Loop (HITL) instructions in `INSTRUCTIONS.md`:
- `# Researcher Instructions`: Guide research direction — fed into every Deep Research sub-agent invocation.
- `# Developer Instructions`: Guide code implementation — fed into every Developer iteration.

### Launch

```bash
python launch_agent.py --slug "enter slug" --iteration 1
```

### Monitoring

- `researcher.txt` / `developer.txt` capture detailed logs.
- Weights & Biases tracking is configured via `config.yaml` under `tracking.wandb`.

---

## Goal Mode

A standalone generate-execute-review loop for problems that don't have a CV metric — data generation, inference speed optimization, or any task you can describe in natural language.

### How it works

1. You write a `GOAL.md` in the repo root describing what you want and any hard constraints.
2. The agent generates a `train.py` script, executes it with the same guardrails and log monitoring as competition mode, then a reviewer LLM judges the result and produces a structured `GoalReview` (score, done, violations, next_step).
3. If not done, the reviewer's `next_step` feeds into the next iteration. The conversation thread grows across iterations so the codegen LLM sees the full history.
4. Artifacts land under `goal_runs/<run_id>/v<N>/` with `train.py`, `train.txt` (execution log), and `review.json` per version, plus a summary `history.json` at the run root.

### Launch

```bash
python goal_mode.py
python goal_mode.py --goal-file GOAL.md --run-id my_run --max-versions 30
python goal_mode.py --wandb-entity myteam --wandb-project myproject
```

### Options

| Flag | Default | Description |
| --- | --- | --- |
| `--goal-file` | `GOAL.md` | Path to the goal description |
| `--run-id` | timestamp | Identifier for this run |
| `--max-versions` | 500 | Maximum iterations |
| `--max-time-seconds` | 432 hours | Wall-clock budget |
| `--wandb-entity` | from config | W&B entity |
| `--wandb-project` | from config | W&B project |
| `--wandb-run-name` | `goal-<run_id>` | W&B run name |

---

## License

MIT
