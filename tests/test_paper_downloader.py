import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.literature_reviewer import LiteratureReviewer, PaperDownloader


@pytest.mark.skipif(
    not os.getenv("S2_API_KEY"),
    reason="S2_API_KEY environment variable must be set to download papers.",
)
def test_paper_download_and_analysis(tmp_path):
    reviewer = LiteratureReviewer(data_dir=tmp_path, max_workers=2)
    results = reviewer.review("multimodal agents", limit=2)

    assert results, "Expected at least one paper result from Semantic Scholar."

    downloadable = [item for item in results if item.pdf_path]
    if not downloadable:
        pytest.skip("No open access PDFs were returned for the query.")

    for review in downloadable:
        assert review.pdf_path is not None
        assert review.pdf_path.exists()

        expected_name = PaperDownloader._sanitize_title(review.metadata.title or "")
        if expected_name:
            assert review.pdf_path.stem.startswith(expected_name[:60])

        if review.insights:
            assert all(isinstance(v, list) for v in review.insights.values())
