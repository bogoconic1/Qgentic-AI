import logging
from dotenv import load_dotenv
from tools.developer import search_sota_suggestions, web_search_stack_trace
import weave
import wandb

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logging.getLogger("tools.developer").setLevel(logging.DEBUG)

wandb.init(project="gstar-project-tests")
weave.init('test_tools_developer')


with open("task/us-patent-phrase-to-phrase-matching/description.md", "r") as f:
    description = f.read()

with open("/workspace/gstar-project/task/us-patent-phrase-to-phrase-matching/outputs/3/code_3_v1.py", "r") as f:
    code = f.read()

with open("/workspace/gstar-project/task/us-patent-phrase-to-phrase-matching/outputs/3/code_3_v1.txt", "r") as f:
    logs = f.read()

suggestions = search_sota_suggestions(
    description,
    code + "\n\n" + logs,
    executed_suggestion=None,
    failed_to_improve_score=True,
    failed_ideas=[],
    executed_code=None,
)
print("SOTA Suggestions:\n", suggestions)

stack_trace = """Traceback (most recent call last):
  File "/workspace/gstar-project/test.py", line 4, in <module>
    from transformers import AdamW
ImportError: cannot import name 'AdamW' from 'transformers' (/opt/conda/lib/python3.11/site-packages/transformers/__init__.py)
"""

stack_trace = """
Give me code to run Qwen/Qwen3-Next-80B-A3B-Instruct model.
"""
fix = web_search_stack_trace(stack_trace)
print("Fix:\n", fix)
