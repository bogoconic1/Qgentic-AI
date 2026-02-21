import logging
import os
import re

from dotenv import load_dotenv
from firecrawl import Firecrawl
import weave
from tools.generate_paper_summary import PaperSummaryClient

load_dotenv()


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
for _noisy in ("httpcore", "httpx", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


@weave.op()
def read_research_paper(arxiv_link: str) -> str:
    """Read and summarize a research paper from arxiv.

    Args:
        arxiv_link: ArXiv paper link (e.g., "https://arxiv.org/pdf/2510.22916" or just "2510.22916")

    Returns:
        Structured markdown summary with sections: Abstract, Introduction, Related Work,
        Method/Architecture, Experiments/Results, and Conclusion.
    """
    arxiv_id_match = re.search(r"(\d{4}\.\d{4,5})", arxiv_link)
    if arxiv_id_match:
        arxiv_id = arxiv_id_match.group(1)
    else:
        # Assume it's already just an ID
        arxiv_id = arxiv_link

    logger.info("Reading research paper with arxiv ID: %s", arxiv_id)

    client = PaperSummaryClient(is_model=False)
    summary = client.generate_summary(model_name=arxiv_id)
    logger.info("Successfully generated paper summary (length: %d chars)", len(summary))
    return summary


@weave.op()
def scrape_web_page(url: str) -> str:
    """Scrape a web page and return LLM-ready markdown content.

    Useful for reading blog posts, documentation, competition forums, winner solutions,
    technical tutorials, and other domain-specific web content that complements arxiv papers.

    Args:
        url: The webpage URL to scrape (e.g., blog posts, Kaggle discussions, documentation)

    Returns:
        Markdown content from the page with title and metadata
    """
    logger.info("Scraping web page: %s", url)

    api_key = os.environ["FIRECRAWL_API_KEY"]
    app = Firecrawl(api_key=api_key)
    doc = app.scrape(url, formats=["markdown"])

    title = doc.metadata.title if doc.metadata and doc.metadata.title else url
    logger.info(
        "Successfully scraped page (length: %d chars, title: %s)",
        len(doc.markdown),
        title,
    )
    return f"# {title}\n\nSource: {url}\n\n{doc.markdown}"
