import json
import logging
import os
import base64
import mimetypes
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from openai import OpenAI
from project_config import get_config

from tools.researcher import ask_eda, download_external_datasets, get_tools
from prompts.researcher_agent import (
    build_system as prompt_build_system,
    initial_user_for_build_plan as prompt_initial_user,
)
from tools.helpers import call_llm_with_retry
import weave
import wandb


def _safe_read(path: str) -> str:
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception:
        return ""

logger = logging.getLogger(__name__)

_CONFIG = get_config()
_LLM_CFG = _CONFIG.get("llm", {}) if isinstance(_CONFIG, dict) else {}
_PATH_CFG = _CONFIG.get("paths", {}) if isinstance(_CONFIG, dict) else {}
_RUNTIME_CFG = _CONFIG.get("runtime", {}) if isinstance(_CONFIG, dict) else {}

_BASE_URL = _LLM_CFG.get("base_url", "https://openrouter.ai/api/v1")
_API_KEY_ENV = _LLM_CFG.get("api_key_env", "OPENROUTER_API_KEY")
_RESEARCHER_AGENT_MODEL = _LLM_CFG.get("researcher_model", "google/gemini-2.5-pro")

_TASK_ROOT = Path(_PATH_CFG.get("task_root", "task"))
_OUTPUTS_DIRNAME = _PATH_CFG.get("outputs_dirname", "outputs")

_DEFAULT_MAX_STEPS = _RUNTIME_CFG.get("researcher_max_steps", 512)

# Media ingestion limits/types
SUPPORTED_IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
MAX_IMAGES_PER_STEP = 6
MAX_TOTAL_BYTES = 5 * 1024 * 1024


class ResearcherAgent:
    """Lead Research Strategist with tool-calling.

    Uses OpenAI tool calling to decide when/what to ask via ask_domain_expert.
    Continues executing tool calls until the model returns a final plan
    (no further tool calls), or the tool-call budget is exhausted.
    """

    def __init__(self, slug: str, iteration: int, run_id: int = 1):
        load_dotenv()
        self.slug = slug
        self.iteration = iteration
        self.run_id = run_id
        os.environ["TASK_SLUG"] = slug
        self.client = OpenAI(api_key=os.environ.get(_API_KEY_ENV), base_url=_BASE_URL)
        self.messages: List[dict] = []
        self.base_dir = _TASK_ROOT / self.slug
        self.outputs_dir = self.base_dir / _OUTPUTS_DIRNAME / str(self.iteration)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        # Configure per-run output dirs for media/external data under outputs/<iter>
        assert self.run_id is not None, "run_id is required"
        self.media_dir = self.outputs_dir / f"media_{self.run_id}"
        self.external_dir = self.outputs_dir / f"external_data_{self.run_id}"
        try:
            self.media_dir.mkdir(parents=True, exist_ok=True)
            self.external_dir.mkdir(parents=True, exist_ok=True)
            os.environ["MEDIA_DIR"] = str(self.media_dir)
            os.environ["EXTERNAL_DATA_DIR"] = str(self.external_dir)
        except Exception:
            pass
        per_run_log_dir = self.outputs_dir / "researcher" / f"run_{self.run_id}"
        per_run_log_dir.mkdir(parents=True, exist_ok=True)
        self.researcher_log_path = per_run_log_dir / f"researcher_{self.run_id}.txt"
        self._configure_logger()

    def _configure_logger(self) -> None:
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.DEBUG,
            )

        # Prevent duplicate handlers
        existing_paths = {
            getattr(handler, "baseFilename", None)
            for handler in logger.handlers
        }
        if str(self.researcher_log_path) not in existing_paths:
            file_handler = logging.FileHandler(self.researcher_log_path)
            logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)
        logger.info(
            "ResearcherAgent initialized for slug=%s iteration=%s", self.slug, self.iteration
        )
        
    def _list_media_files(self) -> set[Path]:
        media_dir = self.media_dir
        if not media_dir.exists():
            return set()
        files: set[Path] = set()
        for p in media_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_TYPES:
                files.add(p)
        return files

    def _encode_image_to_data_url(self, path: Path) -> str:
        mime, _ = mimetypes.guess_type(path.name)
        if mime is None:
            suffix = path.suffix.lower()
            if suffix in {".jpg", ".jpeg"}:
                mime = "image/jpeg"
            elif suffix == ".png":
                mime = "image/png"
            elif suffix == ".webp":
                mime = "image/webp"
            elif suffix == ".gif":
                mime = "image/gif"
            else:
                return ""
        try:
            data = base64.b64encode(path.read_bytes()).decode("utf-8")
        except Exception:
            return ""
        return f"data:{mime};base64,{data}"

    def _ingest_new_media(self, before_set: set[Path]) -> None:
        after_set = self._list_media_files()
        new_files = list(after_set - before_set)
        if not new_files:
            return
        new_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        selected: list[Path] = []
        total = 0
        for p in new_files:
            try:
                size = p.stat().st_size
            except Exception:
                continue
            if size <= 0:
                continue
            if len(selected) >= MAX_IMAGES_PER_STEP:
                break
            if total + size > MAX_TOTAL_BYTES:
                continue
            selected.append(p)
            total += size
        if not selected:
            return
        content: list[dict] = [{"type": "text", "text": f"Attaching {len(selected)} new EDA chart(s) from media/."}]
        for p in selected:
            data_url = self._encode_image_to_data_url(p)
            if not data_url:
                continue
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        if len(content) > 1:
            self.messages.append({"role": "user", "content": content})

    def _compose_system(self) -> str:
        # Description is read here for reuse in initial user message
        self.description = _safe_read(str(self.base_dir / "description.md"))
        return prompt_build_system(str(self.base_dir))

    def _read_starter_summary(self) -> str:
        # Prefer raw starter_suggestions.txt; fallback to JSON; else 'None'
        try:
            txt_path = self.outputs_dir / "starter_suggestions.txt"
            if txt_path.exists():
                return _safe_read(str(txt_path))
        except Exception:
            pass
        try:
            json_path = self.outputs_dir / "starter_suggestions.json"
            if json_path.exists():
                return _safe_read(str(json_path))
        except Exception:
            pass
        return "None"

    @weave.op()
    def build_plan(self, max_steps: int | None = None) -> str:
        system_prompt = self._compose_system()
        starter_summary = self._read_starter_summary()
        self.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_initial_user(self.description or "", starter_summary)},
        ]

        max_steps = max_steps or _DEFAULT_MAX_STEPS
        logger.info("Researcher run started with max_steps=%s", max_steps)
        tools = get_tools()

        for step in range(max_steps):
            # logger.debug("Current conversation tail: %s", self.messages[-2:])
            logger.info("[Researcher] Step %s/%s", step + 1, max_steps)
            if step == max_steps - 1:
                self.messages.append({"role": "user", "content": "This is your FINAL step. Output the final plan now!"})
                logger.info("Reached final step; forcing plan output prompt")

            llm_params = {
                "model": _RESEARCHER_AGENT_MODEL,
                "messages": self.messages,
                "tools": tools,
                "tool_choice": "auto",
            }

            msg_content = "<tool_call>"

            while "<tool_call>" in msg_content or msg_content == "" or """{"name":""" in msg_content:
                completion = call_llm_with_retry(self.client, **llm_params)
                try:
                    msg = completion.choices[0].message
                    msg_content = msg.content
                except Exception:
                    msg = ""
                    msg_content = ""
                logger.debug("Model response content length: %s", len(msg_content or ""))
                if msg.tool_calls:
                    break
                
            self.messages.append(msg.model_dump())

            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    function_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                    logger.info(
                        "Tool call issued: %s with arguments=%s", function_name, arguments
                    )
                    if function_name == "ask_eda":
                        question = arguments.get("question", "")
                        logger.info(f"{question}")
                        if len(question) == 0:
                            tool_output = "Your question cannot be answered based on the competition discussion threads."
                        else:
                            before_media = self._list_media_files()
                            tool_output = ask_eda(question, self.description, data_path=str(self.base_dir))

                        logger.info("```tool")
                        logger.info(f"{tool_output}")
                        logger.info("```")

                        self.messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": tool_output or "Your question cannot be answered based on the competition discussion threads.",
                            }
                        )
                        try:
                            self._ingest_new_media(before_media)
                        except Exception:
                            logger.debug("No media ingested for this step.")
                    elif function_name == "download_external_datasets":
                        query = arguments.get("query", "")
                        if not query:
                            tool_output = "Search query missing; please provide a specific data need."
                        else:
                            tool_output = download_external_datasets(query, self.slug)

                        logger.info("External search response length=%s", len(tool_output or ""))

                        self.messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": tool_output or "No relevant datasets found.",
                            }
                        )
                continue

            # No tool calls -> final plan
            final_content = msg.content or ""
            logger.info("Final plan received at step %s with length=%s", step + 1, len(final_content))
            if len(final_content) == 0:
                logger.error("LLM returned empty final plan at step %s", step + 1)
                raise RuntimeError("No plan produced. Please review the docs and craft a plan.")
            logger.info("Researcher run completed successfully")
            return final_content
