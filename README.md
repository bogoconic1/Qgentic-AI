# Qgentic-AI

Qgentic-AI is an automated ML engineering stack. LLM-driven agents take a problem description, produce a technical plan, generate code, run it locally, analyse the results, and keep refining the solution. Targeted at Kaggle-style competitions today; the stack is extensible to non-competition goals.

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

The repo ships three empty templates — `GOAL.example.md`, `DEVELOPER_INSTRUCTIONS.example.md`, `RESEARCHER_INSTRUCTIONS.example.md`. Copy each one to the matching `*.md` filename and fill it in for your task; the `*.md` working copies are gitignored so per-task content never lands in the repo:

```bash
cp GOAL.example.md GOAL.md
cp DEVELOPER_INSTRUCTIONS.example.md DEVELOPER_INSTRUCTIONS.md
cp RESEARCHER_INSTRUCTIONS.example.md RESEARCHER_INSTRUCTIONS.md
```

Then edit each `*.md` at the repo root:

- **`GOAL.md`**: Session-wide objective, threaded into every agent's system prompt.
- **`DEVELOPER_INSTRUCTIONS.md`**: Task-specific guidance inlined into the Developer subagent's system prompt.
- **`RESEARCHER_INSTRUCTIONS.md`**: Task-specific guidance inlined into the Researcher subagent's system prompt.

Each launch copies all three into `task/<slug>/`, overwriting any stale copies. Root is source of truth; all three are required (the launcher errors at startup if any are missing).

A fourth file, **`task/<slug>/description.md`**, holds the competition description / evaluation criteria. It is **not** copied from root — `launch_agent.py` populates it from `kagglehub` automatically on the first launch with that slug.

The Main Agent bootstraps everything else — it writes ideas, research reports, and per-iteration developer outputs under `task/<slug>/<run_id>/` itself.

### Launch

```bash
python launch_agent.py --slug "enter slug"
python launch_agent.py --slug "enter slug" --run-id my_run --goal-file path/to/GOAL.md
```

`--slug` triggers a `kagglehub` download into `task/<slug>/`. Run id defaults to a timestamp. Main Agent runs indefinitely — SIGINT/SIGKILL when satisfied.

### Monitoring

- `task/<slug>/<run_id>/main_agent_chat.jsonl` — append-only audit log of every MainAgent step (assistant turn + tool result).
- `task/<slug>/<run_id>/developer_v{N}/` — per-iteration developer artifacts (`SOLUTION.py`, `SOLUTION.txt`, `SOLUTION.json`, `submission.csv`, …) plus `developer_chat.jsonl`.
- `task/<slug>/<run_id>/research_<N>/` — per-call researcher artifacts (`RESEARCH.md` + `web_research/`/`web_fetch/` audit records) plus `researcher_chat.jsonl`.
- `task/<slug>/<run_id>/ideas/` — idea pool (memdir-style `INDEX.md` + one file per idea).
- Weights & Biases / Weave tracking is configured via `config.yaml` under `tracking.wandb`.

#### Web viewer

A local Flask app reads the three `*_chat.jsonl` files and renders the full transcript with collapsible tool calls/results, links to companion artifacts (`MAIN.md`, `SOLUTION.{py,md,json,txt}`, `RESEARCH.md`, `web_research/`, `web_fetch/`), and a `?live=1` mode that meta-refreshes every 3 s.

```bash
python -m scripts.viewer --port 8765
# → http://127.0.0.1:8765/
```

Defaults to `127.0.0.1` only — never binds publicly. To reach a remote VM, SSH local-forward the port (`ssh -L 8765:127.0.0.1:8765 …`) rather than passing `--host 0.0.0.0`. Bash tool calls render as `$ <command>` in a shell-style block instead of escaped JSON.

---

## License

MIT
