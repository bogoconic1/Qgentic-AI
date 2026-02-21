"""Project-wide configuration loader."""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache
from typing import Any, Mapping

import yaml


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    """Return project configuration loaded from config.yaml."""
    config_path = Path(__file__).resolve().parent / "config.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_config_value(*keys: str, default: Any | None = None) -> Any:
    """Convenience accessor for nested configuration keys."""
    cfg: Any = get_config()
    for key in keys:
        if isinstance(cfg, Mapping) and key in cfg:
            cfg = cfg[key]
        else:
            return default
    return cfg


_INSTRUCTIONS_HEADINGS = [
    "# Researcher Instructions",
    "# Developer Instructions",
    "# Models",
]


def _extract_section(text: str, heading: str, next_heading: str | None) -> list[str]:
    """Extract non-empty lines between ``heading`` and ``next_heading``."""
    start = text.find(heading)
    if start == -1:
        return []
    start += len(heading)
    if next_heading is not None:
        end = text.find(next_heading, start)
        chunk = text[start:end] if end != -1 else text[start:]
    else:
        chunk = text[start:]
    return [line.strip() for line in chunk.splitlines() if line.strip()]


@lru_cache(maxsize=1)
def get_instructions() -> dict[str, list[str]]:
    """Parse INSTRUCTIONS.md and return content under each known H1 heading.

    Headings (in order): # Researcher Instructions, # Developer Instructions, # Models.
    Every non-empty line between one heading and the next belongs to that section.
    """
    instructions_path = Path(__file__).resolve().parent / "INSTRUCTIONS.md"
    if not instructions_path.exists():
        return {h: [] for h in _INSTRUCTIONS_HEADINGS}
    try:
        text = instructions_path.read_text(encoding="utf-8")
    except Exception:
        return {h: [] for h in _INSTRUCTIONS_HEADINGS}

    sections: dict[str, list[str]] = {}
    for i, heading in enumerate(_INSTRUCTIONS_HEADINGS):
        next_heading = _INSTRUCTIONS_HEADINGS[i + 1] if i + 1 < len(_INSTRUCTIONS_HEADINGS) else None
        sections[heading] = _extract_section(text, heading, next_heading)
    return sections
