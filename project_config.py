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
        "researcher_tool_offline_model": "gpt-5",
        "researcher_tool_online_model": "gpt-5",
        "model_selector_model": "gpt-5",
        "model_recommender_model": "gpt-5",
    },
    "runtime": {
        "ask_eda_max_attempts": 3,
        "download_datasets_max_attempts": 1,
        "researcher_max_steps": 512,
        "llm_max_retries": 3,
        "directory_listing_max_files": 10,
        "researcher_parallel_runs": 1,
        "patch_mode_enabled": False,
    },
    "paths": {
        "task_root": "task",
        "outputs_dirname": "outputs",
        "external_data_dirname": "external-data",
    },
    "guardrails": {
        "logging_basicconfig_order": True,
    },
    "tracking": {
        "wandb": {
            "entity": None,
            "project": None,
        }
    },
    "model_recommender": {
        "hitl_models": [],
        "enable_web_search": True,
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
