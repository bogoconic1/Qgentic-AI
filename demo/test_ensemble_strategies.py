"""
Demo script to test ensemble summary generation and strategy recommendation.

This script:
1. Calls generate_ensemble_summaries() to create technical summaries
2. Calls recommend_ensemble_strategies() to generate ensemble strategies
3. Displays the results
"""

from pathlib import Path
import json
import sys
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.ensembler import generate_ensemble_summaries, recommend_ensemble_strategies


def run_generate_summaries(slug: str, iteration: int):
    """Generate ensemble summaries and display results."""
    print("=" * 80)
    print("STEP 1: GENERATING ENSEMBLE SUMMARIES")
    print("=" * 80)
    print(f"\nCompetition: {slug}")
    print(f"Iteration: {iteration}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    try:
        print("Calling generate_ensemble_summaries()...")
        print("(This may take several minutes as it calls LLM for each model)\n")

        summaries = generate_ensemble_summaries(slug=slug, iteration=iteration)

        print("\n" + "=" * 80)
        print("RESULTS: GENERATED SUMMARIES")
        print("=" * 80)
        print(f"\nTotal summaries generated: {len(summaries)}\n")

        for model_name, summary_content in summaries.items():
            print(f"\n{'─' * 80}")
            print(f"MODEL: {model_name}")
            print(f"{'─' * 80}")
            print(f"Summary length: {len(summary_content)} characters")

            # Display first few lines
            lines = summary_content.split('\n')[:8]
            print("\nPreview:")
            for line in lines:
                if line.strip():
                    print(f"  {line[:100]}")
            print()

        return summaries

    except Exception as e:
        print(f"\n❌ ERROR during summary generation:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return {}


def run_generate_strategies(slug: str, iteration: int):
    """Generate ensemble strategies and display results."""
    print("\n" + "=" * 80)
    print("STEP 2: GENERATING ENSEMBLE STRATEGIES")
    print("=" * 80)
    print(f"\nCompetition: {slug}")
    print(f"Iteration: {iteration}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    try:
        print("Calling recommend_ensemble_strategies()...")
        print("(This may take several minutes as it uses LLM with web search)\n")

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
            # Print strategy with proper wrapping
            for line in strategy_text.split('\n'):
                if line.strip():
                    print(f"   {line}")

            print(f"\n🔧 Models Needed:")
            models = strategy.get('models_needed', [])
            if models:
                for model in models:
                    print(f"   - {model}")
            else:
                print("   - None specified")

            print(f"\n📊 Ensemble Method: {strategy.get('ensemble_method', 'N/A')}")

        return strategies

    except Exception as e:
        print(f"\n❌ ERROR during strategy generation:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return []


def main():
    """Main execution function."""
    # Configuration
    slug = "google-quest-challenge"
    iteration = 4

    print("\n" + "=" * 80)
    print("ENSEMBLE PIPELINE DEMO")
    print("=" * 80)
    print(f"\nThis demo will run the full ensemble preparation pipeline:")
    print(f"  1. Generate technical summaries for baseline models")
    print(f"  2. Generate ensemble strategy recommendations")
    print(f"\nNote: Total execution may take 10+ minutes\n")

    start_time = datetime.now()

    # Step 1: Generate summaries
    summaries = run_generate_summaries(slug, iteration)

    if not summaries:
        print("\n⚠️  No summaries generated. Exiting.")
        return

    print(f"\n✅ Step 1 completed: {len(summaries)} summaries generated")

    # Step 2: Generate strategies
    strategies = run_generate_strategies(slug, iteration)

    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()

    # Final summary
    print(f"\n\n{'=' * 80}")
    if strategies:
        print("✅ DEMO COMPLETED SUCCESSFULLY")
    else:
        print("⚠️  DEMO COMPLETED WITH WARNINGS")
    print("=" * 80)
    print(f"\nSummaries generated: {len(summaries)}")
    print(f"Strategies generated: {len(strategies)}")
    print(f"Total execution time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"Finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    main()
