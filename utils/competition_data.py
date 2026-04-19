"""Populate ``task/<slug>/`` from Kaggle: data + description.md."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from firecrawl import Firecrawl
import kagglehub

logger = logging.getLogger(__name__)


def download_competition_data(slug: str, target_dir: Path) -> None:
    """Download ``slug`` from Kaggle and mirror it into ``target_dir``.

    Wraps ``kagglehub.competition_download`` — which transparently caches under
    ``~/.cache/kagglehub/competitions/<slug>/`` and re-downloads when Kaggle
    publishes a new dataset version — and copies the cache contents to
    ``target_dir``. Fails loudly on missing credentials / network errors.
    """

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    load_dotenv()

    cache_path = Path(kagglehub.competition_download(slug))
    for entry in cache_path.iterdir():
        dest = target_dir / entry.name
        if entry.is_dir():
            shutil.copytree(entry, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(entry, dest)

    logger.info("Synced competition data for slug=%s from %s to %s",
                slug, cache_path, target_dir)


def generate_description_md(slug: str, target_dir: Path) -> None:
    """Scrape Kaggle's ``/overview`` + ``/data`` pages into ``description.md``.

    No-op if ``description.md`` already exists under ``target_dir``. Uses
    Firecrawl (same client the researcher subagent already depends on).
    Fails loudly on missing ``FIRECRAWL_API_KEY`` / network errors.
    """

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    description_path = target_dir / "description.md"
    if description_path.exists():
        return

    load_dotenv()
    client = Firecrawl(api_key=os.environ["FIRECRAWL_API_KEY"])

    overview_url = f"https://www.kaggle.com/competitions/{slug}/overview"
    data_url = f"https://www.kaggle.com/competitions/{slug}/data"

    overview = client.scrape(overview_url, only_main_content=True, formats=["markdown"])
    data = client.scrape(data_url, only_main_content=True, formats=["markdown"])

    content = (
        f"# {slug} — Competition description\n\n"
        f"## Overview ({overview_url})\n\n"
        f"{overview.markdown or ''}\n\n"
        f"---\n\n"
        f"## Data ({data_url})\n\n"
        f"{data.markdown or ''}\n"
    )
    description_path.write_text(content)
    logger.info(
        "Wrote %s for slug=%s (%d chars)", description_path, slug, len(content)
    )
