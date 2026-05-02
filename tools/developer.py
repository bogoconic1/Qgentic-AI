"""Utility helpers for executing generated code with rich logging."""

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
    append_message,
    get_monitor_tools,
)
from prompts.tools_developer import (
    build_stack_trace_prompt as prompt_stack_trace,
    log_monitor_system as prompt_log_monitor_system,
    log_monitor_user as prompt_log_monitor_user,
)
from guardrails.developer import (
    check_logging_basicconfig_order,
    check_solution_txt_filehandler,
)
from schemas.developer import LogMonitorVerdict, StackTraceSolution
from utils.output import truncate_for_llm
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
_RUNTIME_CFG = _CONFIG["runtime"]
_BASELINE_CODE_TIMEOUT = _RUNTIME_CFG["baseline_code_timeout"]
_LOG_MONITOR_INTERVAL = _RUNTIME_CFG["log_monitor_interval"]


SOLUTION_PY_SCAFFOLD = '''\
"""SOLUTION.py — agent-authored training script.

Edit below the logging stanza. Do not move or remove the basicConfig
call — guardrails enforce that it precedes all third-party imports
and registers a FileHandler for SOLUTION.txt.
"""

import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parent / "SOLUTION.txt", mode="w"),
    ],
    format="%(asctime)s %(levelname)s %(message)s",
)

logger = logging.getLogger(__name__)

# === third-party imports below this line ===
'''


@weave.op()
def web_search_stack_trace(query: str) -> str:
    """Research how to fix a bug based on the stack trace and error message.

    Dispatches a single web-search-grounded LLM call against the trace and
    returns the original query annotated with the model's suggested fix.
    """
    logger.info("Dispatching web search for stack trace remediation guidance.")
    trace_index = query.find("Traceback (most recent call last)")
    if trace_index != -1:
        query = query[trace_index:]
    logger.debug("Stack trace query: %s", query)

    system_prompt = prompt_stack_trace()
    messages = [append_message("user", "<query>\n" + query + "\n</query>")]

    response = call_llm(
        model=_DEVELOPER_TOOL_MODEL,
        system_instruction=system_prompt,
        messages=messages,
        enable_google_search=True,
        text_format=StackTraceSolution,
    )

    return truncate_for_llm(
        query
        + "\n"
        + "This is how you can fix the error: \n"
        + response.solution.strip()
    )


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

        Returns raw stdout on success, raw stderr on failure. Callers who want
        stack-trace enrichment can feed the stderr through
        ``web_search_stack_trace`` themselves.

        Honors ``self._timeout_seconds``: if the subprocess hasn't exited by
        then, it is killed via ``self.kill`` and the diagnostic string is
        returned. Without this, a hung snippet would block the parent
        indefinitely (see issue #256).
        """
        remaining = max(0.0, self._timeout_seconds - self.elapsed())
        try:
            self._proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            return self.kill(f"Hard timeout {self._timeout_seconds}s exceeded")

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
        return stderr_text

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


def execute_with_monitor(
    code_path: str | Path,
    *,
    timeout_seconds: int,
    log_monitor_interval: int,
    logger: logging.Logger,
    conda_env: str | None = None,
) -> str:
    """Launch a Python file with LLM-based log monitoring; return captured output.

    Polls the running ExecutionJob every ``log_monitor_interval`` seconds and
    calls ``monitor_logs`` to decide whether the process should be killed.
    Returns the output from ``ExecutionJob.result()`` — raw stdout on success,
    raw stderr on failure.
    """
    job = execute_code(
        str(code_path),
        timeout_seconds=timeout_seconds,
        conda_env=conda_env,
    )
    logger.info(
        "Launched execution for %s (pid=%d, timeout=%ds)",
        code_path,
        job.pid,
        timeout_seconds,
    )

    while not job.done():
        if job.check_timeout():
            logger.warning("Hard timeout reached for %s", code_path)
            return job.kill("Hard timeout exceeded")

        try:
            verdict = monitor_logs(
                log_output=job.recent_output(),
                seconds_since_last_output=job.idle_time(),
                total_elapsed_seconds=job.elapsed(),
                pid=job.pid,
            )
            logger.info(
                "Monitor verdict for %s: %s (%s)",
                code_path,
                verdict.action,
                verdict.reasoning,
            )
            if verdict.action == "kill":
                return job.kill(verdict.reasoning)
        except Exception:
            logger.exception(
                "Monitor call failed for %s — continuing execution", code_path
            )

        time.sleep(log_monitor_interval)

    output = job.result()
    logger.info("Execution output captured for %s", code_path)
    return output


# ---------------------------------------------------------------------------
# run_solution: agent-facing tool that wraps guardrails + execute_with_monitor
# ---------------------------------------------------------------------------


@weave.op()
def run_solution(version_dir: str | Path) -> str:
    """Execute the agent's SOLUTION.py at ``version_dir`` under guardrails + monitor.

    Pipeline: read SOLUTION.py → static guardrails (basicConfig order +
    SOLUTION.txt FileHandler) → execute_with_monitor (subprocess + LLM
    distress monitor) → parse SOLUTION.json. Returns a JSON string with
    ``{success, score?, stats?, elapsed_seconds, output_tail}`` on success
    or ``{success: False, error_kind, ...}`` on failure.

    The agent reads ``SOLUTION.txt`` separately via ``read_file`` for the
    curated log. ``output_tail`` here is a short slice of the captured
    stdout/stderr stream — useful for crashes that never reached the
    script's logger (e.g. import errors before basicConfig) and for the
    monitor's kill diagnostic when it fires.
    """
    version_dir = Path(version_dir)
    code_path = version_dir / "SOLUTION.py"

    if not code_path.exists():
        return json.dumps(
            {
                "success": False,
                "error_kind": "missing_solution_py",
                "error": (
                    f"SOLUTION.py not found at {code_path}. Author it via "
                    "write_file before calling run_solution."
                ),
            }
        )

    code = code_path.read_text(encoding="utf-8")

    g_basic = check_logging_basicconfig_order(code)
    if g_basic["status"] == "fail":
        return json.dumps(
            {
                "success": False,
                "error_kind": "guardrail_basicconfig",
                "violations": g_basic["violations"],
            }
        )

    g_handler = check_solution_txt_filehandler(code)
    if g_handler["status"] == "fail":
        return json.dumps(
            {
                "success": False,
                "error_kind": "guardrail_filehandler",
                "violations": g_handler["violations"],
            }
        )

    start = time.monotonic()
    output = execute_with_monitor(
        code_path=str(code_path),
        timeout_seconds=_BASELINE_CODE_TIMEOUT,
        log_monitor_interval=_LOG_MONITOR_INTERVAL,
        logger=logger,
    )
    elapsed = time.monotonic() - start
    output_tail = truncate_for_llm(output, max_chars=4_000)

    stats_path = version_dir / "SOLUTION.json"
    if not stats_path.exists():
        return json.dumps(
            {
                "success": False,
                "error_kind": "no_stats",
                "elapsed_seconds": elapsed,
                "output_tail": output_tail,
            }
        )

    try:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return json.dumps(
            {
                "success": False,
                "error_kind": "invalid_stats_json",
                "error": str(exc),
                "elapsed_seconds": elapsed,
                "output_tail": output_tail,
            }
        )

    score = stats.get("score") if isinstance(stats, dict) else None
    if (
        not isinstance(score, (int, float))
        or isinstance(score, bool)
        or score != score  # NaN check
        or score in (float("inf"), float("-inf"))
    ):
        return json.dumps(
            {
                "success": False,
                "error_kind": "missing_or_nonfinite_score",
                "stats": stats if isinstance(stats, dict) else None,
                "elapsed_seconds": elapsed,
                "output_tail": output_tail,
            }
        )

    return json.dumps(
        {
            "success": True,
            "score": float(score),
            "stats": stats,
            "elapsed_seconds": elapsed,
            "output_tail": output_tail,
        }
    )
