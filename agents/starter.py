import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from project_config import get_config
from tools.helpers import call_llm_with_retry
from prompts.starter_agent import (
    build_system as prompt_build_system,
    build_user as prompt_build_user,
)
import weave


logger = logging.getLogger(__name__)


_CONFIG = get_config()
_LLM_CFG = _CONFIG.get("llm")
_PATH_CFG = _CONFIG.get("paths")

_STARTER_MODEL = _LLM_CFG.get("developer_tool_model")

_TASK_ROOT = Path(_PATH_CFG.get("task_root"))
_OUTPUTS_DIRNAME = _PATH_CFG.get("outputs_dirname")


class StarterAgent:
    """Propose 5 starter model ideas with code snippets and JSON summary.

    Inputs: competition description and the full 2024 report (handled in prompt).
    Outputs: starter_suggestions.txt (raw), starter_suggestions.json (parsed JSON).
    """

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

    @staticmethod
    def _extract_json_block(text: str) -> Optional[str]:
        if not text:
            return None
        # Prefer fenced JSON block
        import re
        try:
            m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        except Exception:
            pass
        # Fallback: find outermost braces
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            return text[start:end]
        except Exception:
            return None

    @weave.op()
    def run(self):
        """Run the starter prompt and persist outputs; return parsed suggestions."""
        description = open(self.base_dir / "description.md", "r").read()
        system_prompt = prompt_build_system()
        user_prompt = prompt_build_user(description=description)

        messages = [{"role": "user", "content": user_prompt}]

        logger.info("Dispatching StarterAgent request for slug=%s iteration=%s", self.slug, self.iteration)
        response = call_llm_with_retry(
            model=_STARTER_MODEL,
            instructions=system_prompt,
            tools=[],
            messages=messages,
            web_search_enabled=True
        )
        content = response.output_text or ""

        # Persist raw content
        try:
            self.text_path.write_text(content)
        except Exception:
            logger.debug("Failed to persist starter_suggestions.txt")

        # Extract JSON
        suggestions: Dict[str, Any] = {}
        try:
            json_text = self._extract_json_block(content)
            if json_text:
                suggestions = json.loads(json_text)
        except Exception:
            logger.debug("Failed to parse JSON from starter response")

        # Persist JSON
        try:
            self.json_path.write_text(json.dumps(suggestions, indent=2))
        except Exception:
            logger.debug("Failed to persist starter_suggestions.json")

        logger.info("StarterAgent completed with %s keys", len(suggestions) if isinstance(suggestions, dict) else 0)



