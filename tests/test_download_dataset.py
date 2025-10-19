import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import researcher

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

wandb.init(entity='520f8592abc', project='qgentic-ai', name=f'test_download_dataset')
weave.init(project_name='520f8592abc/qgentic-ai')

slug = "us-patent-phrase-to-phrase-matching"
query = f"Give me datasets which are shared in the {slug} competition and is widely used by the community. Make sure the datasets are published before the competition end date."
print(researcher.download_external_datasets(query=query, slug=slug))
