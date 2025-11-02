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

slug = "playground-series-s5e11"
query = "nabihazahid/loan-prediction-dataset-2025"
query2 = "nabihazahid loan-prediction-dataset-2025" 
query3 = "Loan Prediction dataset by nabihazahid"
print(researcher.download_external_datasets(question_1=query, question_2=query2, question_3=query3, slug=slug))
