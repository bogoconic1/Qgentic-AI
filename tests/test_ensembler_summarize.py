import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prompts.ensembler_agent import (
    ensembler_code_summarize_system_prompt,
    ensembler_code_summarize_user_prompt
)
from tools.helpers import call_llm_with_retry

from dotenv import load_dotenv
import logging
import wandb
import weave

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

wandb.init(entity='520f8592abc', project='qgentic-ai', name='test_ensembler_summarize')
weave.init(project_name='520f8592abc/qgentic-ai')

# Test with DeBERTa model (code_2_1_v3.py)
ensemble_dir = Path("/home/geremieyeo/Qgentic-AI/task/learning-agency-lab-automated-essay-scoring-2/outputs/2/ensemble")
description_path = Path("/home/geremieyeo/Qgentic-AI/task/learning-agency-lab-automated-essay-scoring-2/description.md")

# Read inputs
with open(description_path, "r") as f:
    description = f.read()

model_name = "DeBERTa-v3-large (chunked 512 with overlap + ordinal head)"

with open(ensemble_dir / "code_2_1_v3.py", "r") as f:
    code = f.read()

with open(ensemble_dir / "code_2_1_v3.txt", "r") as f:
    logs = f.read()

# Generate prompts
system_prompt = ensembler_code_summarize_system_prompt()
user_prompt = ensembler_code_summarize_user_prompt(description, model_name, code, logs)

print("=" * 80)
print("SYSTEM PROMPT")
print("=" * 80)
print(system_prompt)
print("\n" + "=" * 80)
print("USER PROMPT (first 500 chars)")
print("=" * 80)
print(user_prompt[:500])
print("...\n")

# Call LLM
print("=" * 80)
print("CALLING LLM...")
print("=" * 80)

response = call_llm_with_retry(
    model="gpt-5",
    instructions=system_prompt,
    tools=[],  # No tools needed for summarization
    messages=[{"role": "user", "content": user_prompt}]
)

summary = response.output_text

print("\n" + "=" * 80)
print("SUMMARY OUTPUT")
print("=" * 80)
print(summary)

# Save to file
output_path = ensemble_dir / "summaries" / "summary_model_1.md"
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    f.write(summary)

print("\n" + "=" * 80)
print(f"Saved summary to: {output_path}")
print("=" * 80)

# Clean shutdown
try:
    weave.finish()
except Exception:
    pass
try:
    wandb.finish()
except Exception:
    pass
