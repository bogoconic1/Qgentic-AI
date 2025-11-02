import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.ensembler import recommend_ensemble_strategies

from dotenv import load_dotenv
import logging
import wandb
import weave
load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logging.getLogger("utils.ensembler").setLevel(logging.DEBUG)

wandb.init(entity='520f8592abc', project='qgentic-ai', name='test_generate_ensemble_strategies')
weave.init(project_name='520f8592abc/qgentic-ai')

slug = "playground-series-s5e10"
iteration = 4

print("\n" + "="*80)
print(f"Testing ensemble strategy recommendation for {slug} iteration {iteration}")
print("="*80 + "\n")

strategies = recommend_ensemble_strategies(slug=slug, iteration=iteration)

print("\n" + "="*80)
print(f"Generated {len(strategies)} strategies:")
print("="*80)
for i, strategy_obj in enumerate(strategies, 1):
    if isinstance(strategy_obj, dict):
        strategy_text = strategy_obj.get("strategy", "")
        models_needed = strategy_obj.get("models_needed", [])
        print(f"\n{i}. {strategy_text}")
        if models_needed:
            print(f"   Models needed: {', '.join(models_needed)}")
    else:
        print(f"\n{i}. {strategy_obj}")
print("\n" + "="*80)
