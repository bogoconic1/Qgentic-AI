"""
Demo script to test ONLY ensemble strategy recommendation (skip summary generation).

This script:
1. Calls recommend_ensemble_strategies() using existing summaries
2. Displays the results
"""

from pathlib import Path
import sys
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.ensembler import recommend_ensemble_strategies


def main():
    """Main execution function."""
    # Configuration
    slug = "google-quest-challenge"
    iteration = 4

    print("\n" + "=" * 80)
    print("ENSEMBLE STRATEGY GENERATION (SKIP SUMMARIES)")
    print("=" * 80)
    print(f"\nCompetition: {slug}")
    print(f"Iteration: {iteration}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    start_time = datetime.now()

    print("Calling recommend_ensemble_strategies()...")
    print("(Using existing summaries; this may take 3-5 minutes with web search)\n")

    try:
        strategies = recommend_ensemble_strategies(slug=slug, iteration=iteration)

        print("\n" + "=" * 80)
        print("RESULTS: GENERATED STRATEGIES")
        print("=" * 80)
        print(f"\nTotal strategies generated: {len(strategies)}\n")

        for i, strategy in enumerate(strategies, 1):
            print(f"\n{'─' * 80}")
            print(f"STRATEGY {i}")
            print(f"{'─' * 80}")

            # Display strategy details
            print(f"\n📋 Strategy Description:")
            strategy_text = strategy.get('strategy', 'N/A')
            for line in strategy_text.split('\n'):
                if line.strip():
                    print(f"   {line}")

            print(f"\n🔧 Models Needed:")
            models = strategy.get('models_needed', [])
            if models:
                # Highlight NEW models
                for model in models:
                    if model.startswith('NEW:'):
                        print(f"   🆕 {model}")
                    else:
                        print(f"   - {model}")
            else:
                print("   - None specified")

            print(f"\n📊 Ensemble Method: {strategy.get('ensemble_method', 'N/A')}")

        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()

        print(f"\n\n{'=' * 80}")
        print("✅ DEMO COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"\nStrategies generated: {len(strategies)}")
        print(f"Execution time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        print(f"Finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Count NEW models
        new_count = sum(1 for s in strategies for m in s.get('models_needed', []) if m.startswith('NEW:'))
        if new_count > 0:
            print(f"🆕 Total NEW models suggested: {new_count}\n")

    except Exception as e:
        print(f"\n❌ ERROR during strategy generation:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
