import json
import logging
import os
import base64
from pathlib import Path

from dotenv import load_dotenv
from project_config import get_config, get_instructions

from tools.researcher import read_research_paper, scrape_web_page
from tools.developer import execute_code, _build_resource_header
from utils.llm_utils import get_tools, append_message, encode_image_to_data_url
from prompts.researcher_agent import (
    build_system as prompt_build_system,
    initial_user_for_build_plan as prompt_initial_user,
)
from tools.helpers import call_llm
from google.genai import types
import weave

logger = logging.getLogger(__name__)

_CONFIG = get_config()
_LLM_CFG = _CONFIG["llm"]
_PATH_CFG = _CONFIG["paths"]
_RUNTIME_CFG = _CONFIG["runtime"]
_RESEARCHER_AGENT_MODEL = _LLM_CFG["researcher_model"]

_TASK_ROOT = Path(_PATH_CFG["task_root"])
_OUTPUTS_DIRNAME = _PATH_CFG["outputs_dirname"]

_DEFAULT_MAX_STEPS = _RUNTIME_CFG["researcher_max_steps"]
_HITL_INSTRUCTIONS = get_instructions()["# Researcher Instructions"]

# Media ingestion limits/types
SUPPORTED_IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
MAX_IMAGES_PER_STEP = 6
MAX_TOTAL_BYTES = 5 * 1024 * 1024


class ResearcherAgent:
    """Lead Research Strategist with tool-calling.

    Uses Gemini tool calling to decide when/what to ask via ask_domain_expert.
    Continues executing tool calls until the model returns a final plan
    (no further tool calls), or the tool-call budget is exhausted.
    """

    def __init__(
        self,
        slug: str,
        iteration: int,
        cpu_core_pool: list[list[int]] | None = None,
        gpu_pool: list[str] | None = None,
    ):
        load_dotenv()
        self.slug = slug
        self.iteration = iteration

        self.cpu_core_pool = cpu_core_pool or []
        self.gpu_pool = gpu_pool or []

        self.base_dir = _TASK_ROOT / self.slug
        self.outputs_dir = self.base_dir / _OUTPUTS_DIRNAME / str(self.iteration)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        with open(self.base_dir / "description.md", "r") as f:
            self.description = f.read()

        self.media_dir = self.outputs_dir / "media"
        self.external_dir = self.outputs_dir / "external_data"
        self.media_dir.mkdir(parents=True, exist_ok=True)
        self.external_dir.mkdir(parents=True, exist_ok=True)

        os.environ["TASK_SLUG"] = slug
        os.environ["MEDIA_DIR"] = str(self.media_dir)
        os.environ["EXTERNAL_DATA_DIR"] = str(self.external_dir)

        researcher_log_dir = self.outputs_dir / "researcher"
        researcher_log_dir.mkdir(parents=True, exist_ok=True)
        self.researcher_log_path = researcher_log_dir / "researcher.txt"

        self._configure_logger()

    def _configure_logger(self) -> None:
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.DEBUG,
            )
            for _noisy in ("httpcore", "httpx", "urllib3"):
                logging.getLogger(_noisy).setLevel(logging.WARNING)

        # Prevent duplicate handlers
        existing_paths = {
            getattr(handler, "baseFilename", None) for handler in logger.handlers
        }
        if str(self.researcher_log_path) not in existing_paths:
            file_handler = logging.FileHandler(self.researcher_log_path)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
            )
            logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)
        logger.info(
            "ResearcherAgent initialized for slug=%s iteration=%s",
            self.slug,
            self.iteration,
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
        """Wrapper around utils.llm_utils.encode_image_to_data_url."""
        return encode_image_to_data_url(path)

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
        content: list = []
        for p in selected:
            data_url = self._encode_image_to_data_url(p)
            if ";base64," in data_url:
                mime_part, b64_data = data_url.split(";base64,", 1)
                mime_type = mime_part.replace("data:", "")
                content.append(
                    types.Part(
                        inline_data=types.Blob(
                            mime_type=mime_type, data=base64.b64decode(b64_data)
                        )
                    )
                )

        return [{"role": "user", "parts": content}]

    def _compose_system(self) -> str:
        json_path = self.outputs_dir / "starter_suggestions.json"

        with open(json_path, "r") as f:
            starter_data = json.load(f)

        task_types = starter_data["task_types"]
        return prompt_build_system(
            str(self.base_dir),
            task_types=task_types,
            hitl_instructions=_HITL_INSTRUCTIONS,
        )

    def _read_starter_suggestions(self) -> str:
        json_path = self.outputs_dir / "starter_suggestions.json"
        with open(json_path, "r") as f:
            starter_suggestions = json.load(f)
        task_types = ", ".join(starter_suggestions["task_types"])
        task_summary = starter_suggestions["task_summary"]
        return f"""<task_types>
{task_types}
</task_types>
<task_summary>
{task_summary}
</task_summary>
"""

    def _execute_tool_call(self, item, input_list: list = None, step: int = 0):
        """Execute a single tool call and return result dict.

        Args:
            item: Tool call item from Gemini response (a part with function_call attribute)
            input_list: Message list (unused for Gemini, kept for interface consistency)
            step: Current tool loop step (for unique filenames)

        Returns:
            dict (tool_result format)
        """
        tool_name = item.function_call.name
        tool_args = dict(item.function_call.args)
        tool_id = item.function_call.id if hasattr(item.function_call, "id") else None

        # Execute the tool based on tool_name
        if tool_name == "execute_python":
            code = tool_args.get("code", "")
            if not code:
                tool_output = "Error: code parameter is required."
                return {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": json.dumps({"output": tool_output}),
                    "is_error": True,
                }
            else:
                before_media = self._list_media_files()

                script_file = self.outputs_dir / f"execute_python_{step}.py"
                cpu_range = self.cpu_core_pool[0] if self.cpu_core_pool else None
                gpu_id = self.gpu_pool[0] if self.gpu_pool else None
                resource_header = _build_resource_header(cpu_range, gpu_id)
                script_file.write_text(resource_header + code)

                logger.info("execute_python (code_len=%d, step=%d)", len(code), step)
                job = execute_code(str(script_file), timeout_seconds=300)
                tool_output = job.result()

                result = {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": json.dumps({"output": tool_output}),
                    "is_error": False,
                }

                media_prompt = None
                try:
                    media_prompt = self._ingest_new_media(before_media)
                except Exception as e:
                    logger.error("Failed to ingest new media: %s", e)

                self._pending_media = media_prompt
                return result

        elif tool_name == "read_research_paper":
            arxiv_link = tool_args.get("arxiv_link", "")

            if not arxiv_link:
                tool_output = "Error: arxiv_link is required."
            else:
                logger.info("Reading research paper: %s", arxiv_link)
                tool_output = read_research_paper(arxiv_link)

            logger.info("Paper summary response length=%s", len(tool_output or ""))

            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": json.dumps({"summary": tool_output}),
                "is_error": False,
            }

        elif tool_name == "scrape_web_page":
            url = tool_args.get("url", "")

            if not url:
                tool_output = "Error: url is required."
            else:
                logger.info("Scraping web page: %s", url)
                tool_output = scrape_web_page(url)

            logger.info("Scraped content length=%s", len(tool_output or ""))

            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": json.dumps({"content": tool_output}),
                "is_error": False,
            }

    @weave.op()
    def build_plan(self, max_steps: int | None = None) -> str:
        system_prompt = self._compose_system()
        starter_suggestions = self._read_starter_suggestions()

        max_steps = max_steps or _DEFAULT_MAX_STEPS
        logger.info("Researcher run started with max_steps=%s", max_steps)

        tools = get_tools()
        logger.info("Using model: %s", _RESEARCHER_AGENT_MODEL)

        initial_prompt = prompt_initial_user(self.description, starter_suggestions)
        input_list = [append_message("user", initial_prompt)]

        for step in range(max_steps):
            logger.info("[Researcher] Step %s/%s", step + 1, max_steps)

            if step == max_steps - 1:
                input_list.append(
                    append_message(
                        "user", "This is your FINAL step. Output the final plan now!"
                    )
                )
                logger.info("Reached final step; forcing plan output prompt")

            response = call_llm(
                model=_RESEARCHER_AGENT_MODEL,
                system_instruction=system_prompt,
                messages=input_list,
                function_declarations=tools if tools else [],
                enable_google_search=True,
            )

            has_tool_calls = False
            function_call_parts = []
            if hasattr(response, "candidates") and response.candidates:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        has_tool_calls = True
                        function_call_parts.append(part)

            if has_tool_calls:
                function_response_parts = []
                self._pending_media = None  # Reset pending media

                for part in function_call_parts:
                    result_dict = self._execute_tool_call(part, step=step)

                    function_response_parts.append(
                        types.Part.from_function_response(
                            name=part.function_call.name, response=result_dict
                        )
                    )

                input_list.append(response.candidates[0].content)

                input_list.append(
                    types.Content(role="function", parts=function_response_parts)
                )

                if self._pending_media:
                    media_parts = self._pending_media[0]["parts"]
                    input_list.append(types.Content(role="user", parts=media_parts))
                    logger.info(
                        "Added %d media items to conversation", len(media_parts)
                    )
                    self._pending_media = None

                continue

            final_content = response.text if hasattr(response, "text") else ""

            if step < 5:
                input_list.append(append_message("assistant", final_content))
                input_list.append(append_message("user", "Go ahead."))
                continue

            logger.info(
                "Final plan received at step %s with length=%s",
                step + 1,
                len(final_content),
            )
            logger.info("Researcher run completed successfully")
            return final_content
