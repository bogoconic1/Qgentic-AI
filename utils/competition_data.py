"""Download raw Kaggle competition data into ``task/<slug>/``."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from dotenv import load_dotenv
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
