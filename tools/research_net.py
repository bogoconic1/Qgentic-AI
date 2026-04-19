"""Web-research tools for MainAgent.

Exa ``web_research`` (URL discovery + full page text) and Firecrawl ``web_fetch``
(single-URL scrape → markdown). Plain helpers — no sub-agent wrapper. Callers
are responsible for persisting audit records (MainAgent does this in its
dispatcher).
"""

from __future__ import annotations

import json
import logging
import os

from dotenv import load_dotenv
from exa_py import Exa
from firecrawl import Firecrawl


load_dotenv()


logger = logging.getLogger(__name__)


def tool_web_research(query: str, num_results: int | None = None) -> str:
    """Exa neural search. Returns full text per result — no truncation.

    JSON-encoded dict: ``{"results": [{url, title, text, published_date}, …]}``
    or ``{"error": "..."}`` on failure.
    """
    logger.info("web_research query=%r num_results=%s", query, num_results)

    exa_client = Exa(api_key=os.environ["EXA_API_KEY"])
    search_kwargs = {"type": "auto", "text": True}
    if num_results is not None:
        search_kwargs["num_results"] = num_results

    try:
        search_response = exa_client.search_and_contents(query, **search_kwargs)
    except Exception as exc:
        logger.exception("Exa search_and_contents failed")
        return json.dumps({"error": f"exa search failed: {exc}"})

    results = [
        {
            "url": r.url,
            "title": r.title,
            "text": r.text or "",
            "published_date": r.published_date,
        }
        for r in search_response.results
    ]
    if not results:
        return json.dumps({"error": "no results — try reformulating the query"})

    return json.dumps({"results": results})


def tool_web_fetch(url: str) -> str:
    """Firecrawl scrape → markdown. Full content, no truncation.

    JSON-encoded dict: ``{"url": ..., "title": ..., "markdown": ...}``
    or ``{"error": "..."}`` on failure.
    """
    logger.info("web_fetch url=%s", url)

    firecrawl_client = Firecrawl(api_key=os.environ["FIRECRAWL_API_KEY"])

    try:
        doc = firecrawl_client.scrape(url, only_main_content=True, formats=["markdown"])
    except Exception as exc:
        logger.exception("Firecrawl scrape failed for %s", url)
        return json.dumps({"error": f"firecrawl scrape failed: {exc}"})

    title = doc.metadata.title if doc.metadata is not None else None
    markdown = doc.markdown or ""

    return json.dumps({"url": url, "title": title or url, "markdown": markdown})


def render_tool_record_markdown(
    tool_name: str, seq: int, args: dict, result_json: str
) -> str:
    """Render one tool call (args + result) as a self-contained markdown audit note.

    Supports ``web_research`` and ``web_fetch``. Used by MainAgent's dispatcher
    to persist full (untruncated) results to disk before truncating for the LLM.
    """
    result = json.loads(result_json)
    header = f"# {tool_name} #{seq}\n\n"

    if "error" in result:
        return header + f"**ERROR:** {result['error']}\n"

    if tool_name == "web_research":
        lines = [header, f"**Query:** {args['query']}\n"]
        if args.get("num_results") is not None:
            lines.append(f"**Requested num_results:** {args['num_results']}\n")
        lines.append(f"**Num results returned:** {len(result['results'])}\n\n---\n\n")
        for idx, item in enumerate(result["results"], start=1):
            lines.append(f"## Result {idx}: {item['title'] or '(no title)'}\n\n")
            lines.append(f"- **URL:** {item['url']}\n")
            if item.get("published_date"):
                lines.append(f"- **Published:** {item['published_date']}\n")
            lines.append(f"\n{item['text']}\n\n---\n\n")
        return "".join(lines)

    if tool_name == "web_fetch":
        return (
            header
            + f"**URL:** {result['url']}\n"
            + f"**Title:** {result['title']}\n\n"
            + "---\n\n"
            + f"{result['markdown']}\n"
        )

    raise ValueError(f"Unknown tool_name for record rendering: {tool_name}")
