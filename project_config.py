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


