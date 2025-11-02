"""Semantic Scholar powered literature review helper.

This module provides a modular, concurrency-friendly workflow that:
    1. Searches Semantic Scholar for papers related to a query.
    2. Downloads PDFs into ``data/`` using open-access links, detailed lookups,
       and external identifiers when available.
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
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Sequence

import requests
from dotenv import load_dotenv

try:
    import pymupdf
    _PYMUPDF = pymupdf
except ImportError:
    try:
        import fitz

        _PYMUPDF = fitz
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is required for PDF analysis. Install via `pip install pymupdf`."
        ) from exc


load_dotenv()

LOGGER = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1"
DEFAULT_FIELDS = (
    "paperId,title,url,abstract,isOpenAccess,openAccessPdf,authors,externalIds"
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
    external_ids: Dict[str, str] = field(default_factory=dict)


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

    def get_paper(
        self,
        paper_id: str,
        fields: str = DEFAULT_FIELDS,
    ) -> Optional[PaperMetadata]:
        if not paper_id:
            raise ValueError("paper_id cannot be empty.")
        url = f"{self._base_url}/paper/{paper_id}"
        params = {"fields": fields}
        LOGGER.debug("Fetching paper details for %s", paper_id)
        response = requests.get(
            url,
            headers=self._headers(),
            params=params,
            timeout=self._timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload:
            return None
        return self._parse_paper(payload)

    @staticmethod
    def _parse_paper(entry: Dict) -> PaperMetadata:
        authors = tuple(
            author.get("name", "").strip()
            for author in entry.get("authors", [])
        )
        open_access_pdf = entry.get("openAccessPdf") or {}
        raw_external_ids = entry.get("externalIds") or {}
        external_ids: Dict[str, str] = {}
        for key, value in raw_external_ids.items():
            if not value:
                continue
            if isinstance(value, str):
                external_ids[key] = value
            elif isinstance(value, (list, tuple)):
                external_ids[key] = str(value[0])
            else:
                external_ids[key] = str(value)
        return PaperMetadata(
            paper_id=entry.get("paperId", ""),
            title=entry.get("title", "").strip(),
            url=entry.get("url"),
            abstract=(entry.get("abstract") or "").strip() or None,
            is_open_access=bool(entry.get("isOpenAccess")),
            open_access_pdf_url=open_access_pdf.get("url"),
            authors=authors,
            external_ids=external_ids,
        )


class PaperDownloader:
    """Handles concurrent download of PDFs with multiple fallbacks."""

    def __init__(
        self,
        target_dir: Path | str = "data",
        user_agent: str = "literature-reviewer/1.0",
    ) -> None:
        self._target_dir = Path(target_dir)
        self._target_dir.mkdir(parents=True, exist_ok=True)
        self._user_agent = user_agent

    def download(self, paper: PaperMetadata) -> Optional[Path]:
        dest_path = self._build_dest_path(paper)
        if dest_path.exists():
            LOGGER.debug(
                "PDF already present for %s at %s",
                paper.paper_id,
                dest_path,
            )
            return dest_path

        candidates = self._candidate_urls(paper)
        if not candidates:
            LOGGER.info(
                "No PDF candidates available for '%s'; skipping download.",
                paper.title,
            )
            return None

        for url in candidates:
            try:
                LOGGER.info(
                    "Attempting download for '%s' from %s",
                    paper.title,
                    url,
                )
                self._download_from_url(url, dest_path)
                return dest_path
            except PaperDownloadError as exc:
                LOGGER.debug(
                    "Candidate rejected for %s (%s): %s",
                    paper.paper_id,
                    url,
                    exc,
                )
            except requests.RequestException as exc:
                LOGGER.debug(
                    "Request error for %s (%s): %s",
                    paper.paper_id,
                    url,
                    exc,
                )

        LOGGER.info(
            "Failed to download a PDF for '%s' after trying %s candidates.",
            paper.title,
            len(candidates),
        )
        return None

    def _build_dest_path(self, paper: PaperMetadata) -> Path:
        base_name = self._sanitize_title(paper.title or "")
        if not base_name:
            fallback = paper.paper_id or "paper"
            base_name = self._sanitize_title(fallback) or "paper"

        candidate = self._target_dir / f"{base_name}.pdf"
        index = 1
        while candidate.exists():
            candidate = self._target_dir / f"{base_name} ({index}).pdf"
            index += 1
        return candidate

    @staticmethod
    def _sanitize_title(value: str) -> str:
        normalized = value.encode("ascii", "ignore").decode("ascii")
        cleaned = re.sub(r"[\\/:*?\"<>|]", "", normalized)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned[:180]

    def _candidate_urls(self, paper: PaperMetadata) -> List[str]:
        urls: List[str] = []

        def add(url: Optional[str]) -> None:
            if url and url not in urls:
                urls.append(url)

        add(paper.open_access_pdf_url)

        external_ids = paper.external_ids or {}
        arxiv_id = external_ids.get("ArXiv") or external_ids.get("ARXIV")
        if arxiv_id:
            arxiv_clean = arxiv_id.replace("arXiv:", "")
            add(f"https://arxiv.org/pdf/{arxiv_clean}.pdf")

        acl_id = external_ids.get("ACL")
        if acl_id:
            lower_acl = acl_id.lower()
            add(f"https://aclanthology.org/{lower_acl}.pdf")

        pmc_id = (
            external_ids.get("PubMedCentral")
            or external_ids.get("PMCID")
            or external_ids.get("PUBMEDCENTRAL")
        )
        if pmc_id:
            add(f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/pdf")

        doi = external_ids.get("DOI")
        if doi:
            add(f"https://doi.org/{doi}")

        add(paper.url)
        return urls

    def _download_from_url(self, url: str, dest_path: Path) -> None:
        headers = {"user-agent": self._user_agent}
        with requests.Session() as session:
            with session.get(
                url,
                headers=headers,
                stream=True,
                timeout=120,
                allow_redirects=True,
                verify=False,  # noqa: S501 (semanticscholar hosts many certs)
            ) as response:
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                first_chunk: Optional[bytes] = None
                file_handle = None
                try:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if not chunk:
                            continue
                        if first_chunk is None:
                            first_chunk = chunk
                            if not self._looks_like_pdf(content_type, first_chunk):
                                raise PaperDownloadError(
                                    f"URL did not return PDF content (content-type='{content_type}')"
                                )
                            file_handle = dest_path.open("wb")
                            file_handle.write(first_chunk)
                        else:
                            if file_handle is None:
                                file_handle = dest_path.open("wb")
                            file_handle.write(chunk)
                    if file_handle is None:
                        raise PaperDownloadError(
                            "Empty response when downloading PDF."
                        )
                except Exception:
                    if file_handle is not None:
                        file_handle.close()
                    if dest_path.exists():
                        dest_path.unlink(missing_ok=True)
                    raise
                else:
                    if file_handle is not None:
                        file_handle.close()

    @staticmethod
    def _looks_like_pdf(content_type: str, first_chunk: bytes) -> bool:
        if content_type and "pdf" in content_type.lower():
            return True
        stripped = first_chunk.lstrip()
        return stripped.startswith(b"%PDF")


class PaperAnalyzer:
    """Extracts coarse-grained implementation insights from PDFs."""

    def __init__(self, min_sentence_length: int = 25) -> None:
        self._min_sentence_length = min_sentence_length

    def analyze(self, pdf_path: Path) -> Dict[str, List[str]]:
        text = self._extract_text(pdf_path)
        return self._extract_insights(text)

    def _extract_text(self, pdf_path: Path) -> str:
        LOGGER.debug("Extracting text from %s", pdf_path)
        with _PYMUPDF.open(str(pdf_path)) as document:
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
        metadata = paper

        try:
            if not metadata.open_access_pdf_url or not metadata.is_open_access:
                try:
                    detailed = self.client.get_paper(metadata.paper_id)
                    if detailed:
                        metadata = detailed
                except Exception as enrich_exc:
                    LOGGER.debug(
                        "Unable to enrich metadata for %s: %s",
                        metadata.paper_id,
                        enrich_exc,
                    )

            pdf_path = self.downloader.download(metadata)
            if pdf_path:
                insights = self.analyzer.analyze(pdf_path)
            else:
                LOGGER.debug(
                    "Skipping analysis for %s (no PDF downloaded).",
                    metadata.paper_id,
                )
                error = error or "PDF not available via open-access or fallback sources."
        except Exception as exc:
            error = str(exc)
            LOGGER.warning("Issue while processing %s: %s",
                           paper.paper_id, exc)

        return PaperReview(metadata=metadata, pdf_path=pdf_path, insights=insights, error=error)

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
