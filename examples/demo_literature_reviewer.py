"""Quick demo showing how to use the LiteratureReviewer tool.

Usage:
    set S2_API_KEY="your-semantic-scholar-api-key" in .env file
    python examples/demo_literature_reviewer.py --query "vision-language agents"

The underlying Semantic Scholar client now runs a multi-query search that expands
the seed topic with related phrases and influential authors while filtering for
recent (2022+) publications. This script prints the resulting paper metadata and
high-level analyzer output.

The downloader now retries transient failures (rate limits, timeouts) with
exponential backoff and falls back to CorpusID-hosted PDFs when available.
CLI flags expose the retry configuration so you can tweak behavior when running
large review batches.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from tools.literature_reviewer import LiteratureReviewer, PaperDownloader
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
    parser.add_argument(
        "--max-download-attempts",
        type=int,
        default=3,
        help="Number of retry attempts per PDF candidate URL.",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=1.5,
        help="Base backoff multiplier for PDF retry attempts.",
    )
    parser.add_argument(
        "--max-retry-wait",
        type=float,
        default=30.0,
        help="Maximum wait time (seconds) between retry attempts.",
    )
    args = parser.parse_args()

    configure_logging(args.verbose)

    api_key = os.getenv("S2_API_KEY")
    if not api_key:
        LOGGER.warning(
            "Environment variable S2_API_KEY is not set. Requests may be rate-limited."
        )

    downloader = PaperDownloader(
        target_dir=Path(args.data_dir),
        max_attempts_per_url=args.max_download_attempts,
        backoff_factor=args.retry_backoff,
        max_retry_wait=args.max_retry_wait,
    )
    reviewer = LiteratureReviewer(
        data_dir=Path(args.data_dir),
        downloader=downloader,
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
        print(f"Published: {review.metadata.year or 'Unknown'}")
        primary_authors = ", ".join(review.metadata.authors[:5]) or "Unknown"
        print(f"Authors: {primary_authors}")
        print(f"Open Access: {'Yes' if review.metadata.is_open_access else 'No'}")
        print(f"Corpus ID: {review.metadata.corpus_id or 'Unknown'}")
        print(f"Open Access Corpus ID: {review.metadata.open_access_corpus_id or 'Unknown'}")
        print(f"PDF: {review.pdf_path or 'Not downloaded'}")
        if review.error:
            print(f"Error: {review.error}")
            if not review.pdf_path:
                print("Hint: try adjusting retry settings or inspecting the corpus IDs above.")
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
