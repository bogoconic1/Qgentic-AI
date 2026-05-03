# Qgentic-AI

Qgentic-AI is an automated ML engineering stack. LLM-driven agents take a problem description, produce a technical plan, generate code, run it locally, analyse the results, and keep refining the solution. Targeted at Kaggle-style competitions today; the stack is extensible to non-competition goals.

▶ **[See an example run](https://htmlpreview.github.io/?https://gist.githubusercontent.com/bogoconic1/93fea3777d9baf5a9d67626223a83db6/raw/7dd88a442f484ce2fa8a740ae5d4636455bca54f/trace.html)** — full transcript of MainAgent + Researcher iterating on a competition, exported as a single self-contained HTML file.

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

The original Kaggle pipeline: a MainAgent (which authors and runs SOLUTION.py directly) plus a Researcher subagent for web-grounded research, iterating on a competition with a CV metric.

### Architecture

```mermaid
flowchart TB
    USER(["User"])
    GOAL["GOAL.md<br/>RESEARCHER_INSTRUCTIONS.md"]
    KH[("kagglehub")]
    GEM[("Gemini API")]
    WEB[("Exa + Firecrawl")]

    USER -->|"launch_agent.py --slug X"| LA["launch_agent.py<br/>copies inputs · downloads competition · creates run dir"]
    GOAL --> LA
    KH -. competition .-> LA

    LA --> MA["MainAgent · Gemini loop<br/><br/>Tools<br/>start_dev_session · run_solution<br/>research · web_search_stack_trace<br/>add_idea / update_idea / remove_idea<br/>bash · read_file · write_file · edit_file<br/>list_dir · grep_code · glob_files"]

    MA -. each bash call .-> JUDGE[/"LLM bash safety judge"/]
    MA -. run_solution .-> MON[/"LLM training monitor<br/>watches stdout / stderr live"/]
    MA -. LLM .-> GEM

    MA -->|"research(instruction)"| RA["Researcher subagent · Gemini loop<br/><br/>Tools<br/>web_research · web_fetch<br/>bash · read_file · write_file · edit_file<br/>list_dir · grep_code · glob_files"]
    RA -. LLM .-> GEM
    RA -. search / fetch .-> WEB

    MA --> RD
    RA --> RD
    subgraph RD["Run dir · task/&lt;slug&gt;/&lt;run_id&gt;/"]
      direction LR
      MAIN["MAIN.md"]
      IDEAS["ideas/INDEX.md<br/>ideas/&lt;id&gt;.md"]
      DEV["developer_vN/<br/>SOLUTION.py · .txt · .json · .md<br/>submission.csv"]
      RES["research_N/<br/>RESEARCH.md · web_research/ · web_fetch/"]
      LOG1["main_agent_chat.jsonl"]
      LOG2["research_N/researcher_chat.jsonl"]
    end

    RD --> VS["scripts.viewer<br/>Flask · localhost:8765"]
    RD --> VE["scripts.viewer.export<br/>self-contained trace.html"]
```

Reading the diagram: `launch_agent.py` boots a single in-process **MainAgent** (Gemini-driven) which owns SOLUTION.py authoring + execution directly — no separate developer agent. Each `bash` call is gated by a safety-judge LLM; each `run_solution` streams stdout/stderr through a training-monitor LLM that can kill the process on NaN loss / OOM / deadlock. The only subagent is **Researcher**, invoked on demand via `research(instruction)`; it does its own Gemini loop with `web_research` (Exa neural search) + `web_fetch` (Firecrawl) plus the same filesystem + bash palette MainAgent has, so it can maintain `RESEARCH.md` and probe data as it goes. Every step appends to the run dir, which is what `scripts.viewer` (live UI) and `scripts.viewer.export` (shareable single-file HTML) read.

### Create Required Files

The repo ships two empty templates — `GOAL.example.md`, `RESEARCHER_INSTRUCTIONS.example.md`. Copy each one to the matching `*.md` filename and fill it in for your task; the `*.md` working copies are gitignored so per-task content never lands in the repo:

```bash
cp GOAL.example.md GOAL.md
cp RESEARCHER_INSTRUCTIONS.example.md RESEARCHER_INSTRUCTIONS.md
```

Then edit each `*.md` at the repo root:

- **`GOAL.md`**: Session-wide objective, threaded into every agent's system prompt.
- **`RESEARCHER_INSTRUCTIONS.md`**: Task-specific guidance inlined into the Researcher subagent's system prompt.

Each launch copies both into `task/<slug>/`, overwriting any stale copies. Root is source of truth; both are required (the launcher errors at startup if any are missing).

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
- `task/<slug>/<run_id>/developer_v{N}/` — per-attempt artifacts MainAgent writes via `start_dev_session` (`SOLUTION.py`, `SOLUTION.txt`, `SOLUTION.json`, `submission.csv`, …).
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

##### Sharing a run as a gist

Export one run as a self-contained HTML file (inline CSS, MainAgent + every Researcher subagent inlined as collapsible sections, no broken links), then upload it as a secret gist and view it through the htmlpreview proxy.

[**Example trace**](https://htmlpreview.github.io/?https://gist.githubusercontent.com/bogoconic1/93fea3777d9baf5a9d67626223a83db6/raw/7dd88a442f484ce2fa8a740ae5d4636455bca54f/trace.html) — what one exported run looks like.

```bash
python -m scripts.viewer.export <slug> <run_id> -o trace.html
gh gist create trace.html                                  # secret gist; URL-only access. Add --public to list it on your profile.
GIST_ID=<id-from-the-printed-URL>
RAW=$(gh api gists/$GIST_ID --jq '.files["trace.html"].raw_url')
echo "https://htmlpreview.github.io/?$RAW"                 # open this — gists serve HTML as text/plain otherwise
```

Re-uploading (`gh gist edit $GIST_ID trace.html`) changes the raw URL hash, so re-fetch `RAW` and the wrapped preview link after every edit. Secret ≠ private — anyone with the URL can view; if the run touched secrets, host it somewhere with auth instead.

---

## License

MIT
