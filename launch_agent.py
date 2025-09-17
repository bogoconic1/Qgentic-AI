import argparse
import os

from agents.orchestrator import Orchestrator


def main():
    parser = argparse.ArgumentParser(description="Run Researcher+Developer pipeline")
    parser.add_argument("slug", type=str, help="Competition slug under task/<slug>")
    parser.add_argument("iteration", type=int, help="Iteration number (e.g., 1)")
    parser.add_argument("--tries", type=int, default=3, help="Max code attempts per iteration")
    args = parser.parse_args()

    os.environ["TASK_SLUG"] = args.slug

    orchestrator = Orchestrator(args.slug, args.iteration)
    success, plan = orchestrator.run(max_code_tries=args.tries)

    print("Researcher Strategic Plan (Markdown):\n")
    print(plan)

    outputs_dir = os.path.join("task", args.slug, "outputs", str(args.iteration))
    submission = os.path.join(outputs_dir, "submission.csv")
    if success:
        print(f"\nSuccess. Submission generated at {submission}")
    else:
        print(f"\nFailed to generate submission at {submission}")


if __name__ == "__main__":
    main()


