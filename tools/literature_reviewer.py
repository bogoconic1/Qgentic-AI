"""Semantic Scholar powered literature review helper.

This module provides a modular, concurrency-friendly workflow that:
    1. Searches Semantic Scholar for papers related to a query.
    2. Downloads any available open-access PDFs into ``data/``.
    3. Extracts lightweight implementation insights (model, architecture, dataset)
       from the PDF text using PyMuPDF.

Example:
    >>> reviewer = LiteratureReviewer()
    >>> summaries = reviewer.review("vision transformers for medical imaging", limit=3)
    >>> for item in summaries:
    ...     print(item.metadata.title, item.pdf_path)
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import requests
from dotenv import load_dotenv

import fitz


load_dotenv()

LOGGER = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1"
DEFAULT_FIELDS = (
    "paperId,title,url,abstract,isOpenAccess,openAccessPdf,authors"
)
DEFAULT_LIMIT = 5
DEFAULT_MAX_WORKERS = 4
CHUNK_SIZE = 8192


class PaperDownloadError(Exception):
    """Raised when a paper fails to download."""


@dataclass(frozen=True)
class PaperMetadata:
    """Minimal metadata about a Semantic Scholar paper."""

    paper_id: str
    title: str
    url: Optional[str]
    abstract: Optional[str]
    is_open_access: bool
    open_access_pdf_url: Optional[str]
    authors: tuple[str, ...]


@dataclass
class PaperReview:
    """Container for a downloaded paper and extracted insights."""

    metadata: PaperMetadata
    pdf_path: Optional[Path]
    insights: Dict[str, List[str]]
    error: Optional[str] = None


class SemanticScholarClient:
    """Simple client for Semantic Scholar's Graph API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = SEMANTIC_SCHOLAR_API_URL,
        timeout: int = 30,
    ) -> None:
        self._api_key = api_key or os.getenv("S2_API_KEY")
        if not self._api_key:
            LOGGER.warning(
                "S2_API_KEY not found. API calls may be rate limited or rejected."
            )
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self._api_key:
            headers["X-API-KEY"] = self._api_key
        return headers

    def search(
        self,
        query: str,
        limit: int = DEFAULT_LIMIT,
        fields: str = DEFAULT_FIELDS,
    ) -> List[PaperMetadata]:
        if not query:
            raise ValueError("Query string cannot be empty.")

        params = {"query": query, "limit": limit, "fields": fields}
        url = f"{self._base_url}/paper/search"
        LOGGER.debug("Searching Semantic Scholar: %s", params)
        response = requests.get(
            url,
            headers=self._headers(),
            params=params,
            timeout=self._timeout,
        )
        response.raise_for_status()
        payload = response.json()
        papers = payload.get("data", [])
        return [self._parse_paper(entry) for entry in papers]

    def recommendations(
        self,
        paper_id: str,
        limit: int = DEFAULT_LIMIT,
        fields: str = DEFAULT_FIELDS,
    ) -> List[PaperMetadata]:
        url = (
            f"{self._base_url}/recommendations/v1/papers/forpaper/{paper_id}"
        )
        params = {"limit": limit, "fields": fields}
        LOGGER.debug("Fetching recommendations for %s", paper_id)
        response = requests.get(
            url,
            headers=self._headers(),
            params=params,
            timeout=self._timeout,
        )
        response.raise_for_status()
        payload = response.json()
        papers = payload.get("recommendedPapers", [])
        return [self._parse_paper(entry) for entry in papers]

    @staticmethod
    def _parse_paper(entry: Dict) -> PaperMetadata:
        authors = tuple(author.get("name", "").strip()
                        for author in entry.get("authors", []))
        open_access_pdf = entry.get("openAccessPdf") or {}
        return PaperMetadata(
            paper_id=entry.get("paperId", ""),
            title=entry.get("title", "").strip(),
            url=entry.get("url"),
            abstract=(entry.get("abstract") or "").strip() or None,
            is_open_access=bool(entry.get("isOpenAccess")),
            open_access_pdf_url=open_access_pdf.get("url"),
            authors=authors,
        )


class PaperDownloader:
    """Handles concurrent download of open-access PDFs."""

    def __init__(
        self,
        target_dir: Path | str = "data",
        user_agent: str = "literature-reviewer/1.0",
    ) -> None:
        self._target_dir = Path(target_dir)
        self._target_dir.mkdir(parents=True, exist_ok=True)
        self._user_agent = user_agent

    def download(self, paper: PaperMetadata) -> Optional[Path]:
        if not paper.is_open_access or not paper.open_access_pdf_url:
            LOGGER.info(
                "Paper '%s' is not open access; skipping download.", paper.title)
            return None

        dest_path = self._target_dir / f"{paper.paper_id}.pdf"
        if dest_path.exists():
            LOGGER.debug("PDF already present for %s at %s",
                         paper.paper_id, dest_path)
            return dest_path

        headers = {"user-agent": self._user_agent}
        LOGGER.info("Downloading PDF for '%s' -> %s", paper.title, dest_path)
        with requests.Session() as session:
            with session.get(
                paper.open_access_pdf_url,
                headers=headers,
                stream=True,
                timeout=120,
                verify=False,  # noqa: S501 (semanticscholar hosts many certs)
            ) as response:
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                if "pdf" not in content_type.lower():
                    raise PaperDownloadError(
                        f"Expected a PDF, got '{content_type}' for {paper.paper_id}"
                    )
                with dest_path.open("wb") as fh:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            fh.write(chunk)
        return dest_path


class PaperAnalyzer:
    """Extracts coarse-grained implementation insights from PDFs."""

    def __init__(self, min_sentence_length: int = 25) -> None:
        self._min_sentence_length = min_sentence_length

    def analyze(self, pdf_path: Path) -> Dict[str, List[str]]:
        text = self._extract_text(pdf_path)
        return self._extract_insights(text)

    def _extract_text(self, pdf_path: Path) -> str:
        LOGGER.debug("Extracting text from %s", pdf_path)
        with fitz.open(pdf_path) as document:
            pages = [page.get_text("text") for page in document]
        return "\n".join(pages)

    def _extract_insights(self, raw_text: str) -> Dict[str, List[str]]:
        sentences = self._split_sentences(raw_text)
        buckets: Dict[str, List[str]] = {
            "model": [], "architecture": [], "dataset": []}

        for sentence in sentences:
            lower_sentence = sentence.lower()
            for key in buckets:
                if key in lower_sentence:
                    buckets[key].append(sentence.strip())
        # Keep sentences unique and reasonably short for downstream consumption
        for key, values in buckets.items():
            deduped = []
            seen = set()
            for value in values:
                normalized = value.strip()
                if normalized and normalized not in seen:
                    deduped.append(normalized)
                    seen.add(normalized)
            buckets[key] = deduped[:5]
        return buckets

    def _split_sentences(self, text: str) -> List[str]:
        import re

        text = text.replace("\r", " ")
        candidate_sentences = re.split(r"(?<=[.!?])\s+", text)
        return [
            sentence.strip()
            for sentence in candidate_sentences
            if len(sentence.strip()) >= self._min_sentence_length
        ]


class LiteratureReviewer:
    """Coordinates search, download, and analysis with optional concurrency."""

    def __init__(
        self,
        data_dir: Path | str = "data",
        client: Optional[SemanticScholarClient] = None,
        downloader: Optional[PaperDownloader] = None,
        analyzer: Optional[PaperAnalyzer] = None,
        max_workers: Optional[int] = DEFAULT_MAX_WORKERS,
    ) -> None:
        self.client = client or SemanticScholarClient()
        self.downloader = downloader or PaperDownloader(data_dir)
        self.analyzer = analyzer or PaperAnalyzer()
        self.max_workers = max_workers or DEFAULT_MAX_WORKERS

    def review(
        self,
        query: str,
        limit: int = DEFAULT_LIMIT,
        include_recommendations: bool = False,
        recommendations_per_paper: int = 3,
    ) -> List[PaperReview]:
        papers = self.client.search(query, limit=limit)
        if include_recommendations:
            recs = self._fetch_recommendations(
                papers, recommendations_per_paper)
            papers.extend(recs)

        unique_papers = self._deduplicate(papers)
        LOGGER.info("Processing %s papers for query '%s'",
                    len(unique_papers), query)

        results: List[PaperReview] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(self.max_workers, max(1, len(unique_papers)))
        ) as executor:
            future_map = {
                executor.submit(self._process_paper, paper): paper for paper in unique_papers
            }
            for future in concurrent.futures.as_completed(future_map):
                paper = future_map[future]
                try:
                    review = future.result()
                except Exception as exc:  # pragma: no cover - surfaced for visibility
                    LOGGER.exception(
                        "Failed to process paper %s: %s", paper.paper_id, exc)
                    results.append(
                        PaperReview(
                            metadata=paper,
                            pdf_path=None,
                            insights={},
                            error=str(exc),
                        )
                    )
                else:
                    results.append(review)
        return results

    def _process_paper(self, paper: PaperMetadata) -> PaperReview:
        pdf_path: Optional[Path] = None
        insights: Dict[str, List[str]] = {}
        error: Optional[str] = None

        try:
            pdf_path = self.downloader.download(paper)
            if pdf_path:
                insights = self.analyzer.analyze(pdf_path)
            else:
                LOGGER.debug(
                    "Skipping analysis for %s (no PDF downloaded).", paper.paper_id)
        except Exception as exc:
            error = str(exc)
            LOGGER.warning("Issue while processing %s: %s",
                           paper.paper_id, exc)

        return PaperReview(metadata=paper, pdf_path=pdf_path, insights=insights, error=error)

    def _fetch_recommendations(
        self,
        papers: Sequence[PaperMetadata],
        recommendations_per_paper: int,
    ) -> List[PaperMetadata]:
        recommendations: List[PaperMetadata] = []
        for paper in papers:
            try:
                recommendations.extend(
                    self.client.recommendations(
                        paper.paper_id, limit=recommendations_per_paper)
                )
            except Exception as exc:
                LOGGER.warning(
                    "Failed to load recommendations for %s: %s", paper.paper_id, exc)
        return recommendations

    @staticmethod
    def _deduplicate(papers: Iterable[PaperMetadata]) -> List[PaperMetadata]:
        seen: set[str] = set()
        unique: List[PaperMetadata] = []
        for paper in papers:
            if paper.paper_id and paper.paper_id not in seen:
                unique.append(paper)
                seen.add(paper.paper_id)
        return unique


def review_literature(
    query: str,
    limit: int = DEFAULT_LIMIT,
    include_recommendations: bool = False,
    recommendations_per_paper: int = 3,
    data_dir: Path | str = "data",
    max_workers: Optional[int] = DEFAULT_MAX_WORKERS,
) -> List[PaperReview]:
    """Convenience function for one-off usage."""
    reviewer = LiteratureReviewer(
        data_dir=data_dir,
        max_workers=max_workers,
    )
    return reviewer.review(
        query=query,
        limit=limit,
        include_recommendations=include_recommendations,
        recommendations_per_paper=recommendations_per_paper,
    )


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def _format_summary(review: PaperReview) -> str:
    lines = [
        f"Title: {review.metadata.title}",
        f"Paper ID: {review.metadata.paper_id}",
        f"URL: {review.metadata.url or 'N/A'}",
        f"PDF: {review.pdf_path or 'Not downloaded'}",
    ]
    if review.error:
        lines.append(f"Error: {review.error}")
    for key, snippets in review.insights.items():
        if not snippets:
            continue
        lines.append(f"{key.capitalize()} insights:")
        for snippet in snippets:
            lines.append(f"  - {snippet}")
    return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Semantic Scholar literature reviewer.")
    parser.add_argument("query", help="Search query for Semantic Scholar.")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                        help="Number of search results to process.")
    parser.add_argument(
        "--include-recommendations",
        action="store_true",
        help="Also fetch recommendations for each seed paper.",
    )
    parser.add_argument(
        "--recommendations-per-paper",
        type=int,
        default=3,
        help="Number of recommendations per paper when enabled.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory where PDFs will be stored.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Maximum concurrent worker threads.",
    )
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging.")

    args = parser.parse_args()
    _configure_logging(args.verbose)

    reviews = review_literature(
        query=args.query,
        limit=args.limit,
        include_recommendations=args.include_recommendations,
        recommendations_per_paper=args.recommendations_per_paper,
        data_dir=args.data_dir,
        max_workers=args.max_workers,
    )

    for review in reviews:
        print("=" * 80)
        print(_format_summary(review))


if __name__ == "__main__":
    main()
