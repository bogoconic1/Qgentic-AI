import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from project_config import get_config
from tools.helpers import call_llm
from utils.llm_utils import append_message
from prompts.starter_agent import (
    build_system as prompt_build_system,
    build_user as prompt_build_user,
)
from schemas.starter import StarterSuggestions
import weave


logger = logging.getLogger(__name__)


_CONFIG = get_config()
_LLM_CFG = _CONFIG["llm"]
_PATH_CFG = _CONFIG["paths"]

_STARTER_MODEL = _LLM_CFG["starter_model"]

_TASK_ROOT = Path(_PATH_CFG["task_root"])
_OUTPUTS_DIRNAME = _PATH_CFG["outputs_dirname"]


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
            fh = logging.FileHandler(log_file)
            fh.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
            )
            logger.addHandler(fh)
        logger.setLevel(logging.DEBUG)

    @weave.op()
    def run(self):
        """Run the starter prompt and persist outputs; return parsed suggestions."""
        with open(self.base_dir / "description.md", "r") as f:
            description = f.read()
        system_prompt = prompt_build_system()
        user_prompt = prompt_build_user(description=description)

        messages = [append_message("user", user_prompt)]
        logger.info(
            "Dispatching StarterAgent request for slug=%s iteration=%s",
            self.slug,
            self.iteration,
        )

        response = call_llm(
            model=_STARTER_MODEL,
            system_instruction=system_prompt,
            messages=messages,
            text_format=StarterSuggestions,
            enable_google_search=True,
        )

        suggestions = {
            "task_types": response.task_types,
            "task_summary": response.task_summary,
        }

        self.json_path.write_text(json.dumps(suggestions, indent=2))
        logger.info("StarterAgent completed: task_types=%s", response.task_types)
