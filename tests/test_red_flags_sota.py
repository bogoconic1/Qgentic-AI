import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.developer import search_red_flags, search_sota_suggestions
import re

from dotenv import load_dotenv
import logging
import wandb
import weave
load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logging.getLogger("tools.developer").setLevel(logging.DEBUG)

wandb.init(entity='520f8592abc', project='qgentic-ai', name=f'test_red_flags_sota')
weave.init(project_name='520f8592abc/qgentic-ai')

# Test files
slug = "whale-categorization-playground"
base_dir = Path(__file__).resolve().parents[1] / "task" / slug
code_file = base_dir / "outputs/2_1/code_2_1_v5.py"
log_file = base_dir / "outputs/2_1/code_2_1_v5.txt"
submission_file = base_dir / "outputs/2_1/submission_5.csv"
description_file = base_dir / "description.md"

# Read files
print("=" * 80)
print("Reading test files...")
print("=" * 80)

with open(description_file, "r") as f:
    description = f.read()

with open(code_file, "r") as f:
    code = f.read()

with open(log_file, "r") as f:
    logs = f.read()

# Build context (similar to what Developer Agent does)
context = f"<code>\n{code}\n</code>\n"
context += f"<validation_log>\n{logs[-30000:]}\n</validation_log>\n"  # Last 30k chars

print(f"Description length: {len(description)} chars")
print(f"Code length: {len(code)} chars")
print(f"Logs length: {len(logs)} chars")
print(f"Submission file: {submission_file}")

# STAGE 1: Identify red flags
print("\n" + "=" * 80)
print("STAGE 1: Identifying red flags with EDA tool-calling...")
print("=" * 80)

red_flags_response = search_red_flags(
    description=description,
    context=context,
    data_path=str(base_dir),
    submission_path=str(submission_file),
    max_steps=10,
)

print(f"\nRed flags response length: {len(red_flags_response)} chars")
print("\n" + "-" * 80)
print("Red flags response:")
print("-" * 80)
print(red_flags_response)

# Extract Final Summary
print("\n" + "=" * 80)
print("Extracting Final Summary...")
print("=" * 80)

match = re.search(r'### Final Summary\s*\n(.+)', red_flags_response, re.DOTALL)
if match:
    final_summary = match.group(1).strip()
    print(f"\nFinal Summary ({len(final_summary)} chars):")
    print("-" * 80)
    print(final_summary)
else:
    print("WARNING: Could not extract Final Summary, using full response")
    final_summary = red_flags_response

# STAGE 2: Generate SOTA suggestions
print("\n" + "=" * 80)
print("STAGE 2: Generating SOTA suggestions based on red flags...")
print("=" * 80)

sota_suggestions = search_sota_suggestions(
    description=description,
    context=context,
    red_flags=final_summary,
    executed_suggestion="Initial model implementation",
    failed_to_improve_score=True,
    failed_ideas=[],
    executed_code=None,
    later_recommendations=None,
)

print(f"\nSOTA suggestions length: {len(sota_suggestions)} chars")
print("\n" + "-" * 80)
print("SOTA suggestions response:")
print("-" * 80)
print(sota_suggestions)

print("\n" + "=" * 80)
print("Test completed successfully!")
print("=" * 80)
