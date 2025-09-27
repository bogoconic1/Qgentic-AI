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

wandb.init(entity='bogoconic1', project='gstar-wandb', name=f'test_download_dataset')
weave.init(project_name='bogoconic1/gstar-wandb')

slug = "learning-agency-lab-automated-essay-scoring-2"
query = (
    "Download 'Feedback Prize - Assessing Quality' dataset (train.csv with text and labels) into external/feedback_prize_aq, "
    "and 'Persuade 2.0' argumentative essays corpus if available into external/persuade_2_corpus."
)

print(researcher.download_external_datasets(query=query, slug=slug))
