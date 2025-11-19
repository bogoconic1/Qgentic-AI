import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from project_config import get_config
from tools.helpers import call_llm_with_retry, call_llm_with_retry_anthropic, call_llm_with_retry_google
from utils.llm_utils import detect_provider, append_message
from prompts.starter_agent import (
    build_system as prompt_build_system,
    build_user as prompt_build_user,
)
from constants import VALID_TASK_TYPES, normalize_task_type
from schemas.starter import StarterSuggestions
import weave


logger = logging.getLogger(__name__)


_CONFIG = get_config()
_LLM_CFG = _CONFIG.get("llm")
_PATH_CFG = _CONFIG.get("paths")

_STARTER_MODEL = _LLM_CFG.get("starter_model")

_TASK_ROOT = Path(_PATH_CFG.get("task_root"))
_OUTPUTS_DIRNAME = _PATH_CFG.get("outputs_dirname")


class StarterAgent:
    def __init__(self, slug: str, iteration: int):
        load_dotenv()
        self.slug = slug
        self.iteration = iteration

        self.task_root = _TASK_ROOT
        self.outputs_dirname = _OUTPUTS_DIRNAME
        self.base_dir = self.task_root / slug
        self.outputs_dir = self.base_dir / self.outputs_dirname / str(iteration)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        self.text_path = self.outputs_dir / "starter_suggestions.txt"
        self.json_path = self.outputs_dir / "starter_suggestions.json"

        # Configure logger (avoid duplicate handlers to same file)
        self._configure_logger()

    def _configure_logger(self) -> None:
        existing_paths = {getattr(h, "baseFilename", None) for h in logger.handlers}
        log_file = self.outputs_dir / "starter.log"
        if str(log_file) not in existing_paths:
            try:
                fh = logging.FileHandler(log_file)
                fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
                logger.addHandler(fh)
            except Exception:
                # Fallback: no-op if file handler cannot be created
                pass
        logger.setLevel(logging.DEBUG)

    @weave.op()
    def run(self):
        """Run the starter prompt and persist outputs; return parsed suggestions."""
        with open(self.base_dir / "description.md", "r") as f:
            description = f.read()
        system_prompt = prompt_build_system()
        user_prompt = prompt_build_user(description=description)

        # Detect provider from model name
        provider = detect_provider(_STARTER_MODEL)

        # Create messages in provider-specific format
        messages = [append_message(provider, "user", user_prompt)]
        logger.info("Dispatching StarterAgent request for slug=%s iteration=%s using provider=%s",
                   self.slug, self.iteration, provider)

        # Call appropriate LLM helper based on provider
        if provider == "openai":
            response = call_llm_with_retry(
                model=_STARTER_MODEL,
                instructions=system_prompt,
                tools=[],
                messages=messages,
                web_search_enabled=True,
                text_format=StarterSuggestions,
            )
        elif provider == "anthropic":
            response = call_llm_with_retry_anthropic(
                model=_STARTER_MODEL,
                instructions=system_prompt,
                tools=[],
                messages=messages,
                web_search_enabled=True,
                text_format=StarterSuggestions,
            )
        elif provider == "google":
            response = call_llm_with_retry_google(
                model=_STARTER_MODEL,
                system_instruction=system_prompt,
                messages=messages,
                text_format=StarterSuggestions,
                enable_google_search=True,
            )
        else:
            raise ValueError(f"Unsupported provider for StarterAgent: {provider}")

        # Use structured output parsing
        suggestions: Dict[str, Any] = {}
        try:
            if not response:
                raise ValueError("No structured output received from model")

            # Response is already parsed Pydantic object
            parsed = response

            # Validate and normalize task_types (list)
            if not parsed.task_types:
                raise ValueError("task_types list cannot be empty")

            # Validate and normalize each task type
            normalized_task_types = []
            for raw_task_type in parsed.task_types:
                normalized_task_type = normalize_task_type(raw_task_type)

                if raw_task_type.lower() != normalized_task_type:
                    logger.info(f"Normalized task_type from '{raw_task_type}' to '{normalized_task_type}'")

                if normalized_task_type not in VALID_TASK_TYPES:
                    raise ValueError(
                        f"Invalid task_type '{raw_task_type}' (normalized: '{normalized_task_type}'). "
                        f"Must be one of {VALID_TASK_TYPES}"
                    )

                normalized_task_types.append(normalized_task_type)

            suggestions = {
                "task_types": normalized_task_types,
                "task_summary": parsed.task_summary
            }

        except ValueError as e:
            logger.error(f"Validation error in starter response: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error processing starter response: {e}")
            raise

        # Persist JSON
        try:
            self.json_path.write_text(json.dumps(suggestions, indent=2))
        except Exception:
            logger.debug("Failed to persist starter_suggestions.json")

        logger.info("StarterAgent completed with %s keys", len(suggestions) if isinstance(suggestions, dict) else 0)



