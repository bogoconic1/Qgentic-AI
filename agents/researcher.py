import json
import logging
import os
import base64
import mimetypes
from pathlib import Path

from dotenv import load_dotenv
from project_config import get_config

from tools.researcher import ask_eda, download_external_datasets, get_tools
from prompts.researcher_agent import (
    build_system as prompt_build_system,
    initial_user_for_build_plan as prompt_initial_user,
)
from tools.helpers import call_llm_with_retry
import weave

logger = logging.getLogger(__name__)

_CONFIG = get_config()
_LLM_CFG = _CONFIG.get("llm")
_PATH_CFG = _CONFIG.get("paths")
_RUNTIME_CFG = _CONFIG.get("runtime")
_RESEARCHER_AGENT_MODEL = _LLM_CFG.get("researcher_model")

_TASK_ROOT = Path(_PATH_CFG.get("task_root"))
_OUTPUTS_DIRNAME = _PATH_CFG.get("outputs_dirname")

_DEFAULT_MAX_STEPS = _RUNTIME_CFG.get("researcher_max_steps")

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

        self.base_dir = _TASK_ROOT / self.slug
        self.outputs_dir = self.base_dir / _OUTPUTS_DIRNAME / str(self.iteration)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        with open(self.base_dir / "description.md", "r") as f:
            self.description = f.read()
        # Configure per-run output dirs for media/external data under outputs/<iter>
        assert self.run_id is not None, "run_id is required"

        self.media_dir = self.outputs_dir / f"media_{self.run_id}"
        self.external_dir = self.outputs_dir / f"external_data_{self.run_id}"
        self.media_dir.mkdir(parents=True, exist_ok=True)
        self.external_dir.mkdir(parents=True, exist_ok=True)

        os.environ["TASK_SLUG"] = slug
        os.environ["MEDIA_DIR"] = str(self.media_dir)
        os.environ["EXTERNAL_DATA_DIR"] = str(self.external_dir)

        per_run_log_dir = self.outputs_dir / "researcher" / f"run_{self.run_id}"
        per_run_log_dir.mkdir(parents=True, exist_ok=True)
        self.researcher_log_path = per_run_log_dir / f"researcher_{self.run_id}.txt"

        # Track AB test history: list of dicts with 'question' and 'code'
        self.ab_test_history: list[dict] = []

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
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
            )
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
        content: list[dict] = []
        for p in selected:
            data_url = self._encode_image_to_data_url(p)
            if not data_url:
                continue
            content.append({"type": "input_image", "image_url": data_url})

        return [{"role": "user", "content": content}]

    def _compose_system(self) -> str:
        # Load task_types from starter_suggestions.json
        json_path = self.outputs_dir / "starter_suggestions.json"

        if not json_path.exists():
            raise FileNotFoundError(
                f"starter_suggestions.json not found at {json_path}. "
                "Run starter agent first."
            )

        with open(json_path, "r") as f:
            starter_data = json.load(f)

        if "task_types" not in starter_data:
            raise ValueError(
                f"starter_suggestions.json missing required field 'task_types'. "
                f"Found keys: {list(starter_data.keys())}"
            )

        task_types = starter_data["task_types"]

        if not isinstance(task_types, list) or len(task_types) == 0:
            raise ValueError(
                f"task_types must be a non-empty list, got: {task_types}"
            )

        return prompt_build_system(str(self.base_dir), task_type=task_types)

    def _read_starter_suggestions(self) -> str:
        # Prefer raw starter_suggestions.txt; fallback to JSON; else 'None'
        json_path = self.outputs_dir / "starter_suggestions.json"
        with open(json_path, "r") as f:
            starter_suggestions = json.load(f)
        res = ""
        for key in starter_suggestions.keys():
            res += f"<{key}>\n{starter_suggestions[key]}\n</{key}>\n"
        return res

    def _execute_tool_call(self, item, input_list: list) -> None:
        """Execute a single tool call and append results to input_list.

        Args:
            item: Tool call item from LLM response
            input_list: Message list to append tool results to (modified in-place)
        """
        if item.name == "ask_eda" or item.name == "run_ab_test":
            try:
                question = json.loads(item.arguments)["question"]
            except Exception as e:
                logger.error("Failed to parse %s arguments: %s", item.name, e)
                question = ""
            logger.info(f"{question}")
            if len(question) == 0:
                tool_output = "An error occurred. Please retry."
                input_list.append({
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": json.dumps({
                        "insights": tool_output,
                    })
                })
            else:
                before_media = self._list_media_files()

                # Determine if this is an AB test or EDA question
                is_ab_test = (item.name == "run_ab_test")

                # Get last 6 AB tests for AB test questions, empty list for EDA
                previous_ab_tests = self.ab_test_history[-6:] if is_ab_test else []

                tool_output = ask_eda(
                    question,
                    self.description,
                    data_path=str(self.base_dir),
                    previous_ab_tests=previous_ab_tests
                )

                # For AB tests, extract and store the code in history
                if is_ab_test:
                    # Extract code from eda_temp.py that was just executed
                    code_file = self.base_dir / "eda_temp.py"
                    if code_file.exists():
                        with open(code_file, "r") as f:
                            executed_code = f.read()

                        # Strip OpenBLAS prefix if present (first 3 lines)
                        if executed_code.startswith('import os\nos.environ["OPENBLAS_NUM_THREADS"]'):
                            lines = executed_code.split('\n')
                            executed_code = '\n'.join(lines[3:])

                        self.ab_test_history.append({
                            'question': question,
                            'code': executed_code
                        })
                        logger.info("Stored AB test #%d in history", len(self.ab_test_history))

                input_list.append({
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": json.dumps({
                        "insights": tool_output,
                    })
                })
                try:
                    # I don't care if this silently fails - its not that critical
                    media_prompt = self._ingest_new_media(before_media)
                    if media_prompt: input_list += media_prompt
                except Exception as e:
                    logger.error("Failed to ingest new media: %s", e)
                    logger.info("No media ingested for this step.")

        elif item.name == "download_external_datasets":
            args = json.loads(item.arguments)
            question_1 = args.get("question_1", "")
            question_2 = args.get("question_2", "")
            question_3 = args.get("question_3", "")

            if not question_1 or not question_2 or not question_3:
                tool_output = "All 3 question phrasings are required. Please provide question_1, question_2, and question_3."
            else:
                tool_output = download_external_datasets(question_1, question_2, question_3, self.slug)

            logger.info("External search response length=%s", len(tool_output or ""))

            input_list.append({
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": json.dumps({
                    "results": tool_output,
                })
            })

    @weave.op()
    def build_plan(self, max_steps: int | None = None) -> str:
        system_prompt = self._compose_system()
        starter_suggestions = self._read_starter_suggestions()
        input_list = [
            {"role": "user", "content": prompt_initial_user(self.description, starter_suggestions)},
        ]

        max_steps = max_steps or _DEFAULT_MAX_STEPS
        logger.info("Researcher run started with max_steps=%s", max_steps)
        tools = get_tools()

        for step in range(max_steps):
            # logger.debug("Current conversation tail: %s", self.messages[-2:])
            logger.info("[Researcher] Step %s/%s", step + 1, max_steps)
            if step == max_steps - 1:
                input_list.append({"role": "user", "content": "This is your FINAL step. Output the final plan now!"})
                logger.info("Reached final step; forcing plan output prompt")

            response = call_llm_with_retry(
                model=_RESEARCHER_AGENT_MODEL,
                instructions=system_prompt,
                tools=tools,
                messages=input_list,
                web_search_enabled=True,
            )

            input_list += response.output
            tool_calls = False

            for item in response.output:
                if item.type == "function_call":
                    tool_calls = True
                    self._execute_tool_call(item, input_list)

            if tool_calls: continue

            # No tool calls -> final plan
            final_content = response.output_text or ""
            logger.info("Final plan received at step %s with length=%s", step + 1, len(final_content))
            if len(final_content) == 0:
                logger.error("LLM returned empty final plan at step %s", step + 1)
                raise RuntimeError("No plan produced. Please review the docs and craft a plan.")
            logger.info("Researcher run completed successfully")
            return final_content
