"""Utility helpers for executing generated code with rich logging."""

import base64
import json
import logging
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

from dotenv import load_dotenv
from project_config import get_config
from google.genai import types
from tools.helpers import call_llm
from utils.llm_utils import (
    extract_text_from_response,
    append_message,
    encode_image_to_data_url,
    get_tools,
    get_monitor_tools,
)
from prompts.tools_developer import (
    build_stack_trace_prompt as prompt_stack_trace,
    build_stack_trace_pseudo_prompt as prompt_stack_trace_pseudo,
    red_flags_system as prompt_red_flags_system,
    red_flags_user as prompt_red_flags_user,
    sota_system as prompt_sota_system,
    sota_user as prompt_sota_user,
    log_monitor_system as prompt_log_monitor_system,
    log_monitor_user as prompt_log_monitor_user,
)
from schemas.developer import StackTraceSolution, SOTAResponse, LogMonitorVerdict
import weave

load_dotenv()

# Configure logging once at import. Downstream callers can override if needed.
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
for _noisy in ("httpcore", "httpx", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


_CONFIG = get_config()
_LLM_CFG = _CONFIG["llm"]
_DEVELOPER_TOOL_MODEL = _LLM_CFG["developer_tool_model"]
_FINETUNED_CODE_API_MODEL = _LLM_CFG["finetuned_code_api_model"]
_RUNTIME_CFG = _CONFIG["runtime"]
_BASELINE_TIME_LIMIT = _RUNTIME_CFG["baseline_time_limit"]
_BASELINE_CODE_TIMEOUT = _RUNTIME_CFG["baseline_code_timeout"]
_LOG_MONITOR_INTERVAL = _RUNTIME_CFG["log_monitor_interval"]
_PATH_CFG = _CONFIG["paths"]
_OUTPUTS_DIRNAME = _PATH_CFG["outputs_dirname"]


def _build_resource_header(
    cpu_core_range: list[int] | None, gpu_identifier: str | None
) -> str:
    """Build a Python header that sets CPU affinity and GPU assignment."""
    lines = ["import os\n"]
    if cpu_core_range:
        lines.append("import psutil")
        lines.append(f"psutil.Process().cpu_affinity({cpu_core_range})\n")
    if gpu_identifier is not None:
        lines.append(f'os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu_identifier}"')
    lines.append('os.environ["OPENBLAS_NUM_THREADS"] = "32"\n')
    return "\n".join(lines) + "\n"


@weave.op()
def web_search_stack_trace(query: str) -> str:
    """Research how to fix a bug based on the stack trace and error message."""
    logger.info("Dispatching web search for stack trace remediation guidance.")
    trace_index = query.find("Traceback (most recent call last)")
    if trace_index != -1:
        query = query[trace_index:]
    logger.debug("Stack trace query: %s", query)

    logger.info("Attempting fine-tuned model endpoint first...")

    ft_system_prompt = prompt_stack_trace_pseudo()
    ft_user_prompt = "<query>\n" + query + "\n</query>"

    ft_response = call_llm(
        model=_FINETUNED_CODE_API_MODEL,
        system_instruction=ft_system_prompt,
        messages=ft_user_prompt,
        text_format=StackTraceSolution,
        temperature=1.0,
        max_retries=3,
        enable_google_search=False,
        top_p=1.0,
        thinking_budget=None,
    )

    solution_text = ft_response.reasoning_and_solution.strip()
    is_valid_response = (
        len(solution_text) >= 35 and "I cannot solve this error." not in solution_text
    )

    if is_valid_response:
        logger.info("Fine-tuned model provided a solution, using it.")
        return query + "\n" + "This is how you can fix the error: \n" + solution_text

    logger.info(
        "Fine-tuned model cannot answer (response too short or failure message), falling back to web search workflow."
    )
    system_prompt = prompt_stack_trace()

    messages = [append_message("user", "<query>\n" + query + "\n</query>")]
    logger.debug("Web search messages: %s", messages)

    response = call_llm(
        model=_DEVELOPER_TOOL_MODEL,
        system_instruction=system_prompt,
        messages=messages,
        enable_google_search=True,
        text_format=StackTraceSolution,
    )

    return (
        query
        + "\n"
        + "This is how you can fix the error: \n"
        + response.reasoning_and_solution.strip()
    )


def _ingest_images_for_llm(images: list[Path]) -> list[dict] | None:
    """Encode and format images for Gemini messages.

    Args:
        images: List of image paths to ingest

    Returns:
        List of formatted image messages ready to append, or None if no valid images
    """
    image_content = []
    for img_path in images:
        data_url = encode_image_to_data_url(str(img_path))
        if ";base64," in data_url:
            mime_part, b64_data = data_url.split(";base64,", 1)
            mime_type = mime_part.replace("data:", "")
            image_content.append(
                types.Part(
                    inline_data=types.Blob(
                        mime_type=mime_type, data=base64.b64decode(b64_data)
                    )
                )
            )

    if not image_content:
        return None

    return [{"role": "user", "parts": image_content}]


@weave.op()
def search_red_flags(
    description: str,
    context: str,
    images: list[Path] | None = None,
    train_stats: dict | None = None,
) -> str:
    """Stage 1: Direct analysis to identify red flags in the current approach.

    Args:
        description: Competition description
        context: Current code and logs
        images: Optional list of image paths (e.g., loss_curve.png, metric_curve.png)
        train_stats: Optional training statistics dict from train_stats.json
                    (must include: model_name, cv_scores, cv_mean, cv_std)

    Returns:
        Red flags response text (markdown format with structured analysis)
    """
    logger.info("Dispatching red flags identification via direct analysis")

    context_with_stats = context
    if train_stats:
        stats_text = "## Training Statistics (from train_stats.json)\n\n"
        stats_text += f"```json\n{json.dumps(train_stats, indent=2)}\n```\n\n"
        context_with_stats = stats_text + context
        logger.info("Added train_stats to context (%d keys)", len(train_stats))

    system_prompt = prompt_red_flags_system()
    user_prompt = prompt_red_flags_user(
        description=description,
        context=context_with_stats,
    )

    messages = [append_message("user", user_prompt)]

    if images:
        image_messages = _ingest_images_for_llm(images)
        if image_messages:
            messages.extend(image_messages)

    response = call_llm(
        model=_DEVELOPER_TOOL_MODEL,
        system_instruction=system_prompt,
        function_declarations=[],
        messages=messages,
        enable_google_search=True,
    )
    final_content = extract_text_from_response(response)

    logger.info("Red flags identification completed in single pass")
    return final_content


def _inject_user_guidance(input_list, guidance):
    text = f"[User guidance]: {guidance}"
    input_list.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=text)],
        )
    )


@weave.op()
def search_sota_suggestions(
    description: str,
    context: str,
    red_flags: str,
    executed_suggestion: str | None,
    failed_ideas: list[str],
    slug: str,
    data_path: str,
    later_recommendations: str | None = None,
    shared_suggestions: list[str] | None = None,
    external_data_listing: str | None = None,
    plan_content: str | None = None,
    attempt_number: int = 1,
    cpu_core_range: list[int] | None = None,
    gpu_identifier: str | None = None,
    file_suffix: str | None = None,
    max_tool_steps: int = 20,
    version: int = 1,
    images: list[Path] | None = None,
    train_stats: dict | None = None,
    hitl_sota: bool = False,
) -> str:
    """Stage 2: Use web search and tools to generate SOTA suggestions based on red flags."""
    logger.info(
        "Dispatching SOTA suggestions (Stage 2) with web search (attempt #%d)",
        attempt_number,
    )

    context_with_stats = context
    if train_stats:
        stats_text = "## Training Statistics (from train_stats.json)\n\n"
        stats_text += f"```json\n{json.dumps(train_stats, indent=2)}\n```\n\n"
        context_with_stats = stats_text + context
        logger.info("Added train_stats to context (%d keys)", len(train_stats))
    executed_suggestion_text = (
        executed_suggestion
        or "No previous suggestion executed; this is the first attempt."
    )
    failed_ideas_text = (
        "\n".join(f"- {idea}" for idea in failed_ideas)
        if failed_ideas
        else "No prior ideas are blacklisted."
    )

    # On even attempts, disable shared suggestions to force independent exploration
    if attempt_number % 2 == 0:
        logger.info(
            "Attempt #%d (even): Disabling shared suggestions to encourage novel exploration",
            attempt_number,
        )
        shared_suggestions_text = (
            "No shared suggestions provided for this attempt (exploring independently)."
        )
    else:
        shared_suggestions_text = (
            "\n".join(f"- {suggestion}" for suggestion in (shared_suggestions or []))
            if shared_suggestions
            else "No shared suggestions yet."
        )

    # On even attempts, strip "Validated Findings" section from plan to focus on other recommendations
    modified_plan_content = plan_content
    if attempt_number % 2 == 0 and plan_content:
        validated_start = "# Validated Findings (A/B Tested)"
        risks_start = "# Risks & Mitigations"

        start_idx = plan_content.find(validated_start)
        end_idx = plan_content.find(risks_start)

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            logger.info(
                "Attempt #%d (even): Stripping 'Validated Findings' section (%d chars)",
                attempt_number,
                end_idx - start_idx,
            )
            modified_plan_content = plan_content[:start_idx] + plan_content[end_idx:]
        else:
            logger.warning(
                "Could not find both section headers to strip Validated Findings"
            )

    plans_section = ""
    if modified_plan_content:
        plans_section += f"\n<plan>\n{modified_plan_content}\n</plan>\n"
    if later_recommendations:
        plans_section += f"\n<suggestions>\n{later_recommendations}\n</suggestions>\n"

    time_limit_minutes = int(_BASELINE_CODE_TIMEOUT / 60)

    system_prompt = prompt_sota_system(time_limit_minutes=time_limit_minutes)

    user_prompt = prompt_sota_user(
        description=description,
        plans_section=plans_section,
        red_flags=red_flags,
        failed_ideas_text=failed_ideas_text,
        executed_suggestion_text=executed_suggestion_text,
        context=context_with_stats,
        shared_suggestions_text=shared_suggestions_text,
        external_data_listing=external_data_listing
        or "No external data directories found.",
    )

    tools = get_tools()
    input_list = [append_message("user", user_prompt)]

    if images:
        image_messages = _ingest_images_for_llm(images)
        if image_messages:
            input_list.extend(image_messages)

    step_limit = None if hitl_sota else max_tool_steps
    step = 0
    while step_limit is None or step < step_limit:
        logger.info(
            "SOTA suggestion step %d/%s (tools: %s)",
            step + 1,
            "unlimited" if hitl_sota else str(step_limit),
            "enabled" if tools else "disabled",
        )

        response = call_llm(
            model=_DEVELOPER_TOOL_MODEL,
            system_instruction=system_prompt,
            function_declarations=tools if tools else [],
            messages=input_list,
            enable_google_search=True,
            text_format=SOTAResponse
            if (step_limit is not None and step == step_limit - 1)
            else None,
        )

        # text_format is SOTAResponse on last step (returns Pydantic), None otherwise (returns Gemini response)
        if hasattr(response, "suggestion"):
            logger.info(
                "SOTA suggestions completed at step %d (structured output present)",
                step + 1,
            )
            return response

        parts = response.candidates[0].content.parts
        has_function_calls = any(
            part.function_call for part in parts if hasattr(part, "function_call")
        )

        if not has_function_calls:
            logger.info(
                "SOTA suggestions completed at step %d (no function calls), requesting structured output",
                step + 1,
            )
            response = call_llm(
                model=_DEVELOPER_TOOL_MODEL,
                system_instruction=system_prompt,
                messages=input_list,
                enable_google_search=True,
                text_format=SOTAResponse,
            )
            return response

        function_responses = []

        for part in parts:
            if hasattr(part, "function_call") and part.function_call:
                tool_result_str = _execute_sota_tool_call(
                    item=part.function_call,
                    description=description,
                    data_path=data_path,
                    slug=slug,
                    cpu_core_range=cpu_core_range,
                    gpu_identifier=gpu_identifier,
                    file_suffix=file_suffix,
                    step=step,
                    version=version,
                )
                function_responses.append(
                    types.Part.from_function_response(
                        name=part.function_call.name,
                        response={"result": tool_result_str},
                    )
                )

        input_list.append(response.candidates[0].content)
        if function_responses:
            input_list.append(types.Content(role="function", parts=function_responses))

        # HITL: let user provide guidance for next step
        if hitl_sota:
            user_guidance = input(
                f"[HITL] Step {step + 1} | Enter to continue, guidance, or 'done': "
            ).strip()
            if user_guidance.lower() == "done":
                break
            if user_guidance:
                _inject_user_guidance(input_list, user_guidance)

        step += 1

    logger.warning("SOTA forcing final answer after %d steps", step)

    response = call_llm(
        model=_DEVELOPER_TOOL_MODEL,
        system_instruction=system_prompt
        + "\n\nYou have reached the maximum tool usage limit. Provide your final suggestion now based on the information gathered.",
        messages=input_list,
        enable_google_search=True,
        text_format=SOTAResponse,
    )
    return response  # Already parsed Pydantic object


def _execute_sota_tool_call(
    item,
    description,
    data_path,
    slug,
    cpu_core_range,
    gpu_identifier,
    file_suffix,
    step=0,
    version=1,
):
    """Execute a single SOTA tool call and return JSON result.

    Args:
        item: Gemini function_call object
        step: Current tool loop step (for unique filenames)
        version: Current developer version (for output directory)
    """
    from tools.researcher import scrape_web_page, read_research_paper

    args = dict(item.args)

    if item.name == "execute_python":
        code = args["code"]
        logger.info("SOTA tool: execute_python (code_len=%d, step=%d)", len(code), step)

        version_dir = Path(data_path) / _OUTPUTS_DIRNAME / file_suffix / str(version)
        version_dir.mkdir(parents=True, exist_ok=True)
        script_file = version_dir / f"execute_python_{step}.py"

        resource_header = _build_resource_header(cpu_core_range, gpu_identifier)
        script_file.write_text(resource_header + code)

        job = execute_code(str(script_file), timeout_seconds=300)
        output = job.result()
        return json.dumps({"output": output})

    elif item.name == "scrape_web_page":
        url = args["url"]
        logger.info("SOTA tool: scrape_web_page(%s)", url)
        result = scrape_web_page(url)
        return json.dumps({"content": result})

    elif item.name == "read_research_paper":
        arxiv_link = args["arxiv_link"]
        logger.info("SOTA tool: read_research_paper(%s)", arxiv_link)
        result = read_research_paper(arxiv_link)
        return json.dumps({"summary": result})

    else:
        raise ValueError(f"Unknown SOTA tool: {item.name}")


def _stream_reader(stream, buffer: list, timestamp_ref: list):
    """Read lines from a subprocess stream into a buffer, updating the last-output timestamp.

    Runs as a daemon thread to avoid pipe deadlock when capturing both stdout and stderr.
    """
    try:
        for line in stream:
            buffer.append(line)
            timestamp_ref[0] = time.monotonic()
    except ValueError:
        pass  # stream closed
    finally:
        try:
            stream.close()
        except Exception:
            pass


def _kill_process_group(pid: int):
    """Send SIGKILL to the entire process group rooted at pid."""
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except ProcessLookupError:
        pass  # already dead


class ExecutionJob:
    """Handle for a non-blocking code execution process.

    Wraps a subprocess.Popen with reader threads for stdout/stderr streaming,
    and exposes methods for monitoring, querying output, and killing.
    """

    def __init__(self, proc: subprocess.Popen, timeout_seconds: int, filepath: str):
        self._proc = proc
        self._timeout_seconds = timeout_seconds
        self._filepath = filepath
        self._start_time = time.monotonic()

        self._stdout_buf: list[str] = []
        self._stderr_buf: list[str] = []
        self._last_output_time: list[float] = [self._start_time]

        self._stdout_thread = threading.Thread(
            target=_stream_reader,
            args=(proc.stdout, self._stdout_buf, self._last_output_time),
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=_stream_reader,
            args=(proc.stderr, self._stderr_buf, self._last_output_time),
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

    def done(self) -> bool:
        """Check if the process has finished."""
        return self._proc.poll() is not None

    def result(self) -> str:
        """Get execution result. Blocks until the process finishes if not done.

        Returns stdout on success, or web_search_stack_trace(stderr) on error.
        """
        self._proc.wait()
        self._stdout_thread.join(timeout=5)
        self._stderr_thread.join(timeout=5)

        stdout_text = "".join(self._stdout_buf)
        stderr_text = "".join(self._stderr_buf)

        if self._proc.returncode == 0:
            logger.info("Execution succeeded for %s", self._filepath)
            return stdout_text

        logger.warning(
            "Execution failed for %s with return code %s",
            self._filepath,
            self._proc.returncode,
        )
        return web_search_stack_trace(stderr_text)

    def kill(self, reason: str) -> str:
        """Kill the process group and return a diagnostic message."""
        logger.warning("Killing execution of %s: %s", self._filepath, reason)
        _kill_process_group(self._proc.pid)
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        self._stdout_thread.join(timeout=5)
        self._stderr_thread.join(timeout=5)

        last_lines = "".join(self._stdout_buf[-20:])
        return f"Code execution killed: {reason}\n\nLast output:\n{last_lines}"

    def recent_output(self, n: int = 200) -> str:
        """Last n lines of combined stdout+stderr."""
        combined = self._stdout_buf + self._stderr_buf
        return "".join(combined[-n:])

    def idle_time(self) -> float:
        """Seconds since last output was received."""
        return time.monotonic() - self._last_output_time[0]

    def elapsed(self) -> float:
        """Total seconds since process started."""
        return time.monotonic() - self._start_time

    @property
    def pid(self) -> int:
        return self._proc.pid

    def check_timeout(self) -> bool:
        """True if the hard timeout has been exceeded."""
        return self.elapsed() > self._timeout_seconds


@weave.op()
def execute_code(
    filepath: str, timeout_seconds: int | None = None, conda_env: str | None = None
) -> "ExecutionJob":
    """Launch a Python file for execution and return a job handle immediately.

    The process runs in the background. Use the returned ExecutionJob to:
    - Check completion: job.done()
    - Get result (blocking): job.result()
    - Kill the process: job.kill(reason)
    - Inspect output: job.recent_output(), job.idle_time(), job.elapsed()

    Args:
        filepath: Path to the Python file to execute
        timeout_seconds: Hard timeout in seconds (default: baseline_code_timeout from config)
        conda_env: Conda environment name to use for execution (None = use current env)

    Returns:
        ExecutionJob handle for the running process
    """
    if timeout_seconds is None:
        timeout_seconds = _BASELINE_CODE_TIMEOUT

    conda_exe = os.environ.get("CONDA_EXE", "conda")

    if conda_env:
        # -u for unbuffered output on the inner python process
        cmd = [
            conda_exe,
            "run",
            "--no-capture-output",
            "-n",
            conda_env,
            "python",
            "-u",
            filepath,
        ]
        logger.info(
            "Executing in conda environment '%s': %s (timeout: %d seconds)",
            conda_env,
            filepath,
            timeout_seconds,
        )
    else:
        cmd = ["python", "-u", filepath]
        logger.info(
            "Executing generated script: %s (timeout: %d seconds)",
            filepath,
            timeout_seconds,
        )

    logger.debug("Running subprocess command: %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    return ExecutionJob(proc, timeout_seconds, filepath)


def _execute_monitor_tool_call(item) -> str:
    """Execute a monitor tool call (execute_bash) and return the result as JSON."""
    args = dict(item.args)

    if item.name == "execute_bash":
        command = args["command"]
        logger.info("Monitor tool: execute_bash(%s)", command)
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return json.dumps({"output": result.stdout + result.stderr})
    else:
        raise ValueError(f"Unknown monitor tool: {item.name}")


@weave.op()
def monitor_logs(
    log_output: str,
    seconds_since_last_output: float,
    total_elapsed_seconds: float,
    pid: int,
    max_tool_steps: int = 3,
) -> LogMonitorVerdict:
    """Analyze training logs and system state to decide if the process should be killed.

    Uses an LLM with access to an execute_bash tool for system diagnostics.
    Returns a LogMonitorVerdict with action ("continue" or "kill") and reason.
    """
    system_prompt = prompt_log_monitor_system()
    user_prompt = prompt_log_monitor_user(
        log_output=log_output,
        seconds_since_last_output=seconds_since_last_output,
        total_elapsed_seconds=total_elapsed_seconds,
        pid=pid,
    )

    tools = get_monitor_tools()
    input_list = [append_message("user", user_prompt)]

    for step in range(max_tool_steps):
        is_last_step = step == max_tool_steps - 1
        text_format = LogMonitorVerdict if is_last_step else None

        logger.info("Monitor step %d/%d", step + 1, max_tool_steps)

        response = call_llm(
            model=_DEVELOPER_TOOL_MODEL,
            system_instruction=system_prompt,
            function_declarations=tools if not is_last_step else [],
            messages=input_list,
            enable_google_search=False,
            text_format=text_format,
        )

        # text_format is LogMonitorVerdict on last step (returns Pydantic), None otherwise
        if hasattr(response, "action"):
            return response

        parts = response.candidates[0].content.parts
        has_function_calls = any(
            part.function_call for part in parts if hasattr(part, "function_call")
        )

        if not has_function_calls:
            response = call_llm(
                model=_DEVELOPER_TOOL_MODEL,
                system_instruction=system_prompt,
                messages=input_list,
                enable_google_search=False,
                text_format=LogMonitorVerdict,
            )
            return response

        function_responses = []
        for part in parts:
            if hasattr(part, "function_call") and part.function_call:
                tool_result_str = _execute_monitor_tool_call(part.function_call)
                function_responses.append(
                    types.Part.from_function_response(
                        name=part.function_call.name,
                        response={"result": tool_result_str},
                    )
                )
        input_list.append(response.candidates[0].content)
        if function_responses:
            input_list.append(types.Content(role="function", parts=function_responses))

    # Exhausted steps without a verdict — force one
    logger.warning("Monitor exhausted %d steps, forcing final verdict", max_tool_steps)
    response = call_llm(
        model=_DEVELOPER_TOOL_MODEL,
        system_instruction=system_prompt + "\n\nReturn your verdict now.",
        messages=input_list,
        enable_google_search=False,
        text_format=LogMonitorVerdict,
    )
    return response
