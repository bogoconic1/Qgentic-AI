import logging
import os
import re
from pathlib import Path

from dotenv import load_dotenv
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
    arxiv_id_match = re.search(r'(\d{4}\.\d{4,5})', arxiv_link)
    if arxiv_id_match:
        arxiv_id = arxiv_id_match.group(1)
    else:
        # Assume it's already just an ID
        arxiv_id = arxiv_link

    logger.info("Reading research paper with arxiv ID: %s", arxiv_id)

    try:
        client = PaperSummaryClient(is_model=False)
        summary = client.generate_summary(model_name=arxiv_id)
        logger.info("Successfully generated paper summary (length: %d chars)", len(summary))
        return summary
    except Exception as e:
        logger.exception("Failed to read research paper: %s", arxiv_id)
        return f"Error reading paper: {str(e)}"


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

    try:
        from firecrawl import Firecrawl

        api_key = os.environ.get("FIRECRAWL_API_KEY")
        if not api_key:
            logger.error("FIRECRAWL_API_KEY not set in environment")
            return "Error: FIRECRAWL_API_KEY environment variable is not set. Cannot scrape web pages."

        app = Firecrawl(api_key=api_key)
        doc = app.scrape(url, formats=["markdown"])

        if doc.markdown:
            title = doc.metadata.title if doc.metadata and doc.metadata.title else url
            content = doc.markdown[:30000]  # Limit to first 30000 characters
            truncated = len(doc.markdown) > 30000
            logger.info("Successfully scraped page (length: %d chars, truncated: %s, title: %s)",
                       len(doc.markdown), truncated, title)

            result = f"# {title}\n\nSource: {url}\n\n{content}"
            if truncated:
                result += f"\n\n... (content truncated at 30000 characters, original length: {len(doc.markdown)} chars)"
            return result
        else:
            logger.warning("No markdown content returned from Firecrawl for URL: %s", url)
            return f"Error: Failed to scrape page at {url}. The page may be inaccessible or contain no readable content."

    except ImportError:
        logger.error("Firecrawl package not installed")
        return "Error: Firecrawl package is not installed. Install with: pip install firecrawl-py"
    except Exception as e:
        logger.exception("Failed to scrape web page: %s", url)
        return f"Error scraping page: {str(e)}"
