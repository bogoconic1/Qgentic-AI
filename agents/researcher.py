import json
import logging
import os
import base64
import mimetypes
import threading
from pathlib import Path
from concurrent.futures import as_completed

from dotenv import load_dotenv
from project_config import get_config
from weave.trace.util import ThreadPoolExecutor

from tools.researcher import ask_eda, download_external_datasets, read_research_paper, scrape_web_page
from utils.llm_utils import get_tools_for_provider, detect_provider, append_message
from prompts.researcher_agent import (
    build_system as prompt_build_system,
    initial_user_for_build_plan as prompt_initial_user,
)
from tools.helpers import call_llm_with_retry, call_llm_with_retry_anthropic, call_llm_with_retry_google
import weave

logger = logging.getLogger(__name__)

_CONFIG = get_config()
_LLM_CFG = _CONFIG.get("llm")
_PATH_CFG = _CONFIG.get("paths")
_RUNTIME_CFG = _CONFIG.get("runtime")
_RESEARCHER_CFG = _CONFIG.get("researcher", {})
_RESEARCHER_AGENT_MODEL = _LLM_CFG.get("researcher_model")

_TASK_ROOT = Path(_PATH_CFG.get("task_root"))
_OUTPUTS_DIRNAME = _PATH_CFG.get("outputs_dirname")

_DEFAULT_MAX_STEPS = _RUNTIME_CFG.get("researcher_max_steps")
_HITL_INSTRUCTIONS = _RESEARCHER_CFG.get("hitl_instructions", [])

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

    def __init__(self, slug: str, iteration: int, run_id: int = 1, max_parallel_workers: int = 1, cpu_core_pool: list[list[int]] | None = None, gpu_pool: list[str] | None = None):
        load_dotenv()
        self.slug = slug
        self.iteration = iteration
        self.run_id = run_id
        self.max_parallel_workers = max_parallel_workers

        # Resource pools for parallel AB test execution (CPU affinity and GPU isolation)
        self.cpu_core_pool = cpu_core_pool or []  # List of CPU core ranges
        self.gpu_pool = gpu_pool or []  # List of GPU identifiers (MIG UUIDs or GPU IDs)

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

        # Track AB test history: only stores baseline (first) AB test with 'question' and 'code'
        self.ab_test_history: list[dict] = []
        self.ab_test_lock = threading.Lock()  # Thread-safe access to ab_test_history

        # Track EDA history: stores last 6 EDA calls with 'question' and 'code'
        self.eda_history: list[dict] = []
        self.eda_lock = threading.Lock()  # Thread-safe access to eda_history

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

    def _encode_image_to_data_url(self, path: Path, resize_for_anthropic: bool = False) -> str:
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
            image_bytes = path.read_bytes()

            # Resize image if needed for Anthropic (max 2000px per dimension for multi-image requests)
            if resize_for_anthropic:
                try:
                    from PIL import Image
                    import io

                    img = Image.open(io.BytesIO(image_bytes))
                    width, height = img.size

                    # Check if resizing is needed
                    if width > 2000 or height > 2000:
                        # Calculate new dimensions maintaining aspect ratio
                        if width > height:
                            new_width = 2000
                            new_height = int(height * (2000 / width))
                        else:
                            new_height = 2000
                            new_width = int(width * (2000 / height))

                        # Resize the image
                        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                        # Save to bytes
                        output = io.BytesIO()
                        # Preserve format, default to PNG if format is unknown
                        img_format = img.format if img.format else 'PNG'
                        img.save(output, format=img_format)
                        image_bytes = output.getvalue()

                        logger.info(f"Resized image {path.name} from {width}x{height} to {new_width}x{new_height} for Anthropic API")
                except ImportError:
                    logger.warning("PIL not available for image resizing. Image may fail if dimensions exceed 2000px.")
                except Exception as e:
                    logger.warning(f"Failed to resize image {path.name}: {e}. Using original.")

            data = base64.b64encode(image_bytes).decode("utf-8")
        except Exception:
            return ""
        return f"data:{mime};base64,{data}"

    def _ingest_new_media(self, before_set: set[Path], provider: str = "openai") -> None:
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
            # Resize images for Anthropic to respect 2000px limit for multi-image requests
            resize_for_anthropic = (provider == "anthropic")
            data_url = self._encode_image_to_data_url(p, resize_for_anthropic=resize_for_anthropic)
            if not data_url:
                continue

            # Format image based on provider
            if provider == "anthropic":
                # Extract base64 data and mime type from data URL
                # data_url format: "data:image/jpeg;base64,/9j/4AAQ..."
                if ";base64," in data_url:
                    mime_part, b64_data = data_url.split(";base64,", 1)
                    mime_type = mime_part.replace("data:", "")
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": b64_data
                        }
                    })
            elif provider == "google":
                # Gemini format: Part with inline_data
                from google.genai import types
                if ";base64," in data_url:
                    mime_part, b64_data = data_url.split(";base64,", 1)
                    mime_type = mime_part.replace("data:", "")
                    content.append(
                        types.Part(
                            inline_data=types.Blob(
                                mime_type=mime_type,
                                data=base64.b64decode(b64_data)
                            )
                        )
                    )
            else:  # OpenAI format
                content.append({"type": "input_image", "image_url": data_url})

        # Return in provider-specific format
        if provider == "google":
            return [{"role": "user", "parts": content}]
        else:
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

        return prompt_build_system(str(self.base_dir), task_type=task_types, max_parallel_workers=self.max_parallel_workers, hitl_instructions=_HITL_INSTRUCTIONS)

    def _read_starter_suggestions(self) -> str:
        # Prefer raw starter_suggestions.txt; fallback to JSON; else 'None'
        json_path = self.outputs_dir / "starter_suggestions.json"
        with open(json_path, "r") as f:
            starter_suggestions = json.load(f)
        res = ""
        for key in starter_suggestions.keys():
            value = starter_suggestions[key]
            # Format lists as comma-separated strings instead of Python list representation
            if isinstance(value, list):
                value = ", ".join(str(item) for item in value)
            res += f"<{key}>\n{value}\n</{key}>\n"
        return res

    def _execute_single_ab_test(self, question: str, index: int) -> tuple[str, str]:
        """Execute a single AB test question with resource isolation.

        Args:
            question: AB test question to execute
            index: Index for unique file naming and resource assignment (0-based)

        Returns:
            Tuple of (question, result)
        """
        logger.info(f"Executing AB test #{index+1}: {question}")

        file_suffix = f"_{index+1}"

        # Assign CPU and GPU resources based on index
        cpu_core_range = self.cpu_core_pool[index] if index < len(self.cpu_core_pool) else None
        gpu_identifier = self.gpu_pool[index] if index < len(self.gpu_pool) else None

        if cpu_core_range:
            logger.info(f"AB test #{index+1} assigned CPU cores: {len(cpu_core_range)} cores")
        if gpu_identifier:
            logger.info(f"AB test #{index+1} assigned GPU: {gpu_identifier}")

        # Get baseline (first) AB test for context (thread-safe)
        with self.ab_test_lock:
            previous_ab_tests = self.ab_test_history[:1].copy()

        tool_output = ask_eda(
            question,
            self.description,
            data_path=str(self.base_dir),
            previous_ab_tests=previous_ab_tests,
            file_suffix=file_suffix,
            cpu_core_range=cpu_core_range,
            gpu_identifier=gpu_identifier
        )

        # Extract and store the code in history
        code_file = self.base_dir / f"eda_temp{file_suffix}.py"
        if code_file.exists():
            with open(code_file, "r") as f:
                executed_code = f.read()

            # Strip resource allocation header (CPU affinity, CUDA, OpenBLAS)
            # New format with CPU affinity or old format with just OpenBLAS
            lines = executed_code.split('\n')

            # Find the first line that's not part of the header
            start_idx = 0
            for i, line in enumerate(lines):
                stripped = line.strip()
                # Skip imports, psutil calls, os.environ assignments, comments, and empty lines
                if (stripped.startswith('import os') or
                    stripped.startswith('import psutil') or
                    stripped.startswith('psutil.Process(') or
                    stripped.startswith('os.environ["CUDA_VISIBLE_DEVICES"]') or
                    stripped.startswith('os.environ["OPENBLAS_NUM_THREADS"]') or
                    stripped.startswith('# CPU affinity') or
                    stripped == ''):
                    continue
                else:
                    start_idx = i
                    break

            if start_idx > 0:
                executed_code = '\n'.join(lines[start_idx:])

            # Only store the baseline (first) AB test in memory
            with self.ab_test_lock:
                if len(self.ab_test_history) == 0:
                    self.ab_test_history.append({
                        'question': question,
                        'code': executed_code
                    })
                    logger.info("Stored baseline AB test in history")
                else:
                    logger.info("Skipping storage of AB test #%d (only baseline stored)", index + 1)

        return question, tool_output

    def _format_tool_result(self, tool_id: str, output_data: dict, provider: str, is_error: bool = False):
        """Format tool result for the appropriate provider.

        Args:
            tool_id: Tool call ID (call_id for OpenAI, id for Anthropic)
            output_data: Dict of output data (e.g., {"insights": "..."})
            provider: "openai" or "anthropic"
            is_error: Whether this is an error response

        Returns:
            Formatted dict for the provider
        """
        if provider == "openai":
            return {
                "type": "function_call_output",
                "call_id": tool_id,
                "output": json.dumps(output_data)
            }
        else:  # anthropic or google
            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": json.dumps(output_data),
                "is_error": is_error
            }

    def _execute_tool_call(self, item, input_list: list = None, provider: str = "openai"):
        """Execute a single tool call and append results to input_list or return dict.

        Args:
            item: Tool call item from LLM response
                  - OpenAI: item with .name, .arguments, .call_id
                  - Anthropic: item with .name, .input, .id
                  - Gemini: part with .function_call (which has .name, .args, .id)
            input_list: Message list to append tool results to (for OpenAI, modified in-place)
                       If None, returns dict (for Anthropic/Gemini)
            provider: "openai", "anthropic", or "google"

        Returns:
            None for OpenAI (modifies input_list in-place)
            dict for Anthropic/Gemini (tool_result format)
        """
        # Parse tool name and arguments based on provider
        if provider == "openai":
            tool_name = item.name
            try:
                tool_args = json.loads(item.arguments)
            except Exception as e:
                logger.error("Failed to parse arguments: %s", e)
                tool_args = {}
            tool_id = item.call_id
        elif provider == "anthropic":
            tool_name = item.name
            tool_args = item.input  # Already a dict
            tool_id = item.id
        elif provider == "google":
            # For Gemini, item is a part with function_call attribute
            tool_name = item.function_call.name
            tool_args = dict(item.function_call.args)
            tool_id = item.function_call.id if hasattr(item.function_call, 'id') else None
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        # Execute the tool based on tool_name
        if tool_name == "ask_eda":
            question = tool_args.get("question", "")
            logger.info(f"EDA: {question}")
            if len(question) == 0:
                tool_output = "An error occurred. Please retry."
                result = self._format_tool_result(tool_id, {"insights": tool_output}, provider, is_error=True)
                if provider == "openai":
                    input_list.append(result)
                else:  # anthropic or google
                    return result
            else:
                before_media = self._list_media_files()

                # Get last 6 EDA calls for context (thread-safe)
                with self.eda_lock:
                    previous_edas = self.eda_history[-6:].copy()

                tool_output = ask_eda(
                    question,
                    self.description,
                    data_path=str(self.base_dir),
                    previous_ab_tests=previous_edas  # Pass EDA history (parameter name kept for compatibility)
                )

                # Extract and store the executed code in EDA history
                code_file = self.base_dir / "eda_temp.py"
                if code_file.exists():
                    try:
                        with open(code_file, "r") as f:
                            executed_code = f.read()

                        # Strip resource allocation header (CPU affinity, CUDA, OpenBLAS)
                        # New format with CPU affinity or old format with just OpenBLAS
                        lines = executed_code.split('\n')

                        # Find the first line that's not part of the header
                        start_idx = 0
                        for i, line in enumerate(lines):
                            stripped = line.strip()
                            # Skip imports, psutil calls, os.environ assignments, comments, and empty lines
                            if (stripped.startswith('import os') or
                                stripped.startswith('import psutil') or
                                stripped.startswith('psutil.Process(') or
                                stripped.startswith('os.environ["CUDA_VISIBLE_DEVICES"]') or
                                stripped.startswith('os.environ["OPENBLAS_NUM_THREADS"]') or
                                stripped.startswith('# CPU affinity') or
                                stripped == ''):
                                continue
                            else:
                                start_idx = i
                                break

                        if start_idx > 0:
                            executed_code = '\n'.join(lines[start_idx:])

                        # Store in EDA history (keep last 6)
                        with self.eda_lock:
                            self.eda_history.append({
                                'question': question,
                                'code': executed_code
                            })
                            # Keep only last 6
                            if len(self.eda_history) > 6:
                                self.eda_history = self.eda_history[-6:]
                            logger.info("Stored EDA call in history (%d total)", len(self.eda_history))
                    except Exception as e:
                        logger.error("Failed to store EDA history: %s", e)

                result = self._format_tool_result(tool_id, {"insights": tool_output}, provider)

                # Ingest media for both providers
                media_prompt = None
                try:
                    media_prompt = self._ingest_new_media(before_media, provider)
                except Exception as e:
                    logger.error("Failed to ingest new media: %s", e)
                    logger.info("No media ingested for this step.")

                if provider == "openai":
                    input_list.append(result)
                    if media_prompt:
                        input_list += media_prompt
                else:  # anthropic or google
                    # For Anthropic/Gemini, return both tool result and media (to be combined in main loop)
                    self._pending_media = media_prompt  # Store for later use
                    return result

        elif tool_name == "run_ab_test":
            questions = tool_args.get("questions", [])
            if not isinstance(questions, list):
                questions = [questions]

            if len(questions) == 0:
                tool_output = "An error occurred. Please retry with valid questions."
                result = self._format_tool_result(tool_id, {"insights": tool_output}, provider, is_error=True)
                if provider == "openai":
                    input_list.append(result)
                else:  # anthropic or google
                    return result
            else:
                # Truncate to max_parallel_workers if too many questions provided
                truncation_warning = ""
                if len(questions) > self.max_parallel_workers:
                    logger.warning(f"Received {len(questions)} questions but max_parallel_workers is {self.max_parallel_workers}. Truncating to first {self.max_parallel_workers} questions.")
                    skipped_questions = questions[self.max_parallel_workers:]
                    logger.info(f"Skipped questions: {skipped_questions}")
                    truncation_warning = f"\n\nWARNING: You provided {len(questions)} questions but the maximum allowed is {self.max_parallel_workers}. Only the first {self.max_parallel_workers} were executed. The following questions were skipped:\n" + "\n".join([f"- {q}" for q in skipped_questions])
                    questions = questions[:self.max_parallel_workers]

                logger.info(f"Running {len(questions)} AB tests in parallel")
                before_media = self._list_media_files()

                # Execute AB tests in parallel with resource isolation
                results = []
                with ThreadPoolExecutor(max_workers=self.max_parallel_workers) as executor:
                    future_to_question = {
                        executor.submit(self._execute_single_ab_test, q, i): q
                        for i, q in enumerate(questions)
                    }

                    for future in as_completed(future_to_question):
                        question = future_to_question[future]
                        try:
                            q, output = future.result()
                            results.append(f"Question: {q}\nResult: {output}")
                        except Exception as e:
                            logger.error(f"AB test failed for question '{question}': {e}")
                            results.append(f"Question: {question}\nResult: Error - {str(e)}")

                # Combine all results
                combined_output = "\n\n---\n\n".join(results) + truncation_warning

                result = self._format_tool_result(tool_id, {"insights": combined_output}, provider)

                # Ingest media for both providers
                media_prompt = None
                try:
                    media_prompt = self._ingest_new_media(before_media, provider)
                except Exception as e:
                    logger.error("Failed to ingest new media: %s", e)
                    logger.info("No media ingested for this step.")

                if provider == "openai":
                    input_list.append(result)
                    if media_prompt:
                        input_list += media_prompt
                else:  # anthropic or google
                    # For Anthropic/Gemini, return both tool result and media (to be combined in main loop)
                    self._pending_media = media_prompt  # Store for later use
                    return result

        elif tool_name == "download_external_datasets":
            question_1 = tool_args.get("question_1", "")
            question_2 = tool_args.get("question_2", "")
            question_3 = tool_args.get("question_3", "")

            if not question_1 or not question_2 or not question_3:
                tool_output = "All 3 question phrasings are required. Please provide question_1, question_2, and question_3."
            else:
                tool_output = download_external_datasets(question_1, question_2, question_3, self.slug)

            logger.info("External search response length=%s", len(tool_output or ""))

            result = self._format_tool_result(tool_id, {"results": tool_output}, provider)
            if provider == "openai":
                input_list.append(result)
            else:
                return result

        elif tool_name == "read_research_paper":
            arxiv_link = tool_args.get("arxiv_link", "")

            if not arxiv_link:
                tool_output = "Error: arxiv_link is required."
            else:
                logger.info("Reading research paper: %s", arxiv_link)
                tool_output = read_research_paper(arxiv_link)

            logger.info("Paper summary response length=%s", len(tool_output or ""))

            result = self._format_tool_result(tool_id, {"summary": tool_output}, provider)
            if provider == "openai":
                input_list.append(result)
            else:
                return result

        elif tool_name == "scrape_web_page":
            url = tool_args.get("url", "")

            if not url:
                tool_output = "Error: url is required."
            else:
                logger.info("Scraping web page: %s", url)
                tool_output = scrape_web_page(url)

            logger.info("Scraped content length=%s", len(tool_output or ""))

            result = self._format_tool_result(tool_id, {"content": tool_output}, provider)
            if provider == "openai":
                input_list.append(result)
            else:
                return result

    @weave.op()
    def build_plan(self, max_steps: int | None = None) -> str:
        system_prompt = self._compose_system()
        starter_suggestions = self._read_starter_suggestions()

        max_steps = max_steps or _DEFAULT_MAX_STEPS
        logger.info("Researcher run started with max_steps=%s, max_parallel_workers=%s", max_steps, self.max_parallel_workers)

        # Detect provider and get appropriate tools
        provider = detect_provider(_RESEARCHER_AGENT_MODEL)
        tools = get_tools_for_provider(provider, max_parallel_workers=self.max_parallel_workers)
        logger.info("Using provider: %s with model: %s", provider, _RESEARCHER_AGENT_MODEL)

        # Initialize conversation in provider-specific format using append_message
        initial_prompt = prompt_initial_user(self.description, starter_suggestions)
        input_list = [append_message(provider, "user", initial_prompt)]

        for step in range(max_steps):
            logger.info("[Researcher] Step %s/%s", step + 1, max_steps)

            if step == max_steps - 1:
                input_list.append(append_message(provider, "user", "This is your FINAL step. Output the final plan now!"))
                logger.info("Reached final step; forcing plan output prompt")

            if provider == "openai":
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
                        self._execute_tool_call(item, input_list, provider="openai")

                if tool_calls:
                    continue

                # No tool calls -> final plan
                final_content = response.output_text or ""

            elif provider == "anthropic":
                response = call_llm_with_retry_anthropic(
                    model=_RESEARCHER_AGENT_MODEL,
                    instructions=system_prompt,
                    tools=tools,
                    messages=input_list,
                    web_search_enabled=True,
                )

                # Check stop_reason
                if response.stop_reason == "tool_use":
                    # Extract tool_use blocks from content
                    tool_uses = [
                        block for block in response.content
                        if hasattr(block, 'type') and block.type == 'tool_use'
                    ]

                    # Execute all tools and collect results
                    tool_results = []
                    self._pending_media = None  # Reset pending media
                    for tool_use in tool_uses:
                        result = self._execute_tool_call(tool_use, provider="anthropic")
                        tool_results.append(result)

                    # Add to messages in Anthropic format
                    # IMPORTANT: Keep ALL content blocks including web_search_result
                    # (needed for citations to work, even though they increase token usage)
                    input_list.append({
                        "role": "assistant",
                        "content": response.content
                    })

                    # Combine tool results with media if available
                    user_content = tool_results
                    if hasattr(self, '_pending_media') and self._pending_media:
                        # Add media images after tool results
                        user_content = tool_results + self._pending_media[0]["content"]
                        self._pending_media = None  # Clear after use

                    input_list.append({
                        "role": "user",
                        "content": user_content
                    })

                    continue

                # No tool use -> final plan
                final_content = ''.join([
                    block.text for block in response.content
                    if hasattr(block, 'text')
                ])

            elif provider == "google":
                response = call_llm_with_retry_google(
                    model=_RESEARCHER_AGENT_MODEL,
                    system_instruction=system_prompt,
                    messages=input_list,  # Pass full conversation history
                    function_declarations=tools if tools else [],
                    enable_google_search=True,
                )

                # Check if response has function calls
                has_tool_calls = False
                function_call_parts = []
                if hasattr(response, 'candidates') and response.candidates:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            has_tool_calls = True
                            function_call_parts.append(part)

                if has_tool_calls:
                    # Execute all tool calls and collect results
                    from google.genai import types

                    function_response_parts = []
                    self._pending_media = None  # Reset pending media

                    for part in function_call_parts:
                        # Execute tool using existing method
                        result_dict = self._execute_tool_call(part, provider="google")

                        # Format as Gemini function response
                        function_response_parts.append(
                            types.Part.from_function_response(
                                name=part.function_call.name,
                                response=result_dict
                            )
                        )

                    # Add model's response with tool calls to conversation
                    input_list.append(response.candidates[0].content)

                    # Add function responses
                    input_list.append(
                        types.Content(role="function", parts=function_response_parts)
                    )

                    # Add media if available
                    if hasattr(self, '_pending_media') and self._pending_media:
                        # _pending_media format: [{"role": "user", "parts": [Part, Part, ...]}]
                        media_parts = self._pending_media[0]["parts"]
                        input_list.append(
                            types.Content(role="user", parts=media_parts)
                        )
                        logger.info("Added %d media items to conversation", len(media_parts))
                        self._pending_media = None

                    continue

                # No tool calls -> final plan
                final_content = response.text if hasattr(response, 'text') else ""

            else:
                raise ValueError(f"Unsupported provider: {provider}")

            if step < 5:
                input_list.append(append_message(provider, "assistant", final_content))
                input_list.append(append_message(provider, "user", "Go ahead."))
                continue

            # Common final plan handling
            logger.info("Final plan received at step %s with length=%s", step + 1, len(final_content))
            if len(final_content) == 0:
                logger.error("LLM returned empty final plan at step %s", step + 1)
                raise RuntimeError("No plan produced. Please review the docs and craft a plan.")
            logger.info("Researcher run completed successfully")
            return final_content
