"""Project-wide configuration loader."""

from __future__ import annotations

import copy
from pathlib import Path
from functools import lru_cache
from typing import Any, Mapping

import yaml


_DEFAULT_CONFIG: dict[str, Any] = {
    "llm": {
        "developer_model": "gpt-5",
        "developer_tool_model": "gpt-5",
        "researcher_model": "gpt-5",
        "model_selector_model": "gpt-5",
        "model_recommender_model": "gpt-5",
    },
    "runtime": {
        "researcher_max_steps": 512,
        "llm_max_retries": 3,
        "max_developer_input_tokens": 250000,
        "directory_listing_max_files": 10,
        "patch_mode_enabled": False,
        "use_validation_score": False,
    },
    "paths": {
        "task_root": "task",
        "outputs_dirname": "outputs",
        "external_data_dirname": "external-data",
    },
    "guardrails": {
        "logging_basicconfig_order": True,
        "leakage_review": True,
        "enable_code_safety": True,
    },
    "tracking": {
        "wandb": {
            "entity": None,
            "project": None,
        }
    },
    "model_recommender": {
        "enable_web_search": True,
    },
    "developer": {
        "hitl_sota": False,
    },
}


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, Mapping)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    """Return project configuration (merging config.yaml with defaults)."""
    config_path = Path(__file__).resolve().parent / "config.yaml"
    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as f:
                user_config = yaml.safe_load(f) or {}
            if isinstance(user_config, Mapping):
                return _deep_merge(_DEFAULT_CONFIG, user_config)
        except Exception:
            pass
    return copy.deepcopy(_DEFAULT_CONFIG)


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
