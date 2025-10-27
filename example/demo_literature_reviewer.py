"""Quick demo showing how to use the LiteratureReviewer tool.

Usage:
    export S2_API_KEY="your-semantic-scholar-api-key"
    python example/demo_literature_reviewer.py --query "vision-language agents"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from tools.literature_reviewer import LiteratureReviewer
from prompts.literature_reviewer import literature_reviewer_prompt

LOGGER = logging.getLogger(__name__)


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo: LiteratureReviewer usage.")
    parser.add_argument(
        "--query",
        required=True,
        help="Semantic Scholar search query (e.g., 'multimodal web agents').",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Maximum number of papers to fetch.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory where PDFs will be downloaded.",
    )
    parser.add_argument(
        "--recommendations",
        action="store_true",
        help="Include Semantic Scholar recommendations for each seed paper.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    configure_logging(args.verbose)

    api_key = os.getenv("S2_API_KEY")
    if not api_key:
        LOGGER.warning(
            "Environment variable S2_API_KEY is not set. Requests may be rate-limited."
        )

    reviewer = LiteratureReviewer(
        data_dir=Path(args.data_dir),
        max_workers=4,
    )

    reviews = reviewer.review(
        query=args.query,
        limit=args.limit,
        include_recommendations=args.recommendations,
    )

    system_prompt = literature_reviewer_prompt()

    for review in reviews:
        print("=" * 120)
        print(f"Title: {review.metadata.title}")
        print(f"Paper ID: {review.metadata.paper_id}")
        print(f"URL: {review.metadata.url or 'N/A'}")
        print(f"PDF: {review.pdf_path or 'Not downloaded'}")
        if review.error:
            print(f"Error: {review.error}")
            continue
        print("\nSample prompt (truncated) for downstream LLM analysis:")
        prompt_payload = {
            "paper_title": review.metadata.title,
            "paper_abstract": review.metadata.abstract or "unknown",
            "paper_highlights": [],
            "paper_sections": [],
        }
        preview = json.dumps(prompt_payload, indent=2)[:500]
        print(system_prompt.split("## Output Expectations")[0].strip())
        print("\nPayload stub:")
        print(preview)
        if review.insights:
            print("\nExtracted insights (heuristic snippet capture):")
            for key, values in review.insights.items():
                print(f"- {key}:")
                for snippet in values:
                    print(f"    • {snippet}")


if __name__ == "__main__":
    main()
