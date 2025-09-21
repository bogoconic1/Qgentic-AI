import logging

from dotenv import load_dotenv

from tools.developer import search_sota_suggestions


load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logging.getLogger("tools.developer").setLevel(logging.DEBUG)


with open("task/us-patent-phrase-to-phrase-matching/description.md", "r") as f:
    description = f.read()

with open("/workspace/gstar-project/task/us-patent-phrase-to-phrase-matching/outputs/3/code_3_v1.py", "r") as f:
    code = f.read()

with open("/workspace/gstar-project/task/us-patent-phrase-to-phrase-matching/outputs/3/code_3_v1.txt", "r") as f:
    logs = f.read()

suggestions = search_sota_suggestions(description, code + "\n\n" + logs)
print("SOTA Suggestions:\n", suggestions)
