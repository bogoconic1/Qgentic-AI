"""Prompts for EnsemblerAgent."""

from __future__ import annotations
import json


def build_system(ensemble_folder: str) -> str:
    return f"""
"""


def build_initial_user(description: str, metadata: dict) -> str:
    return f"""<competition description>
{description}
</competition description>

<ensemble metadata>
{json.dumps(metadata, indent=2)}
</ensemble metadata>
"""
