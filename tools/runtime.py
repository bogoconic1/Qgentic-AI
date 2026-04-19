"""Subprocess runtime for MainAgent's ``execute_python`` tool.

- ``execute_code(filepath)`` — launch a Python file and return an ``ExecutionJob``.
- ``ExecutionJob.result()`` — blocks; returns stdout on success, or stderr
  enriched with a web-searched remediation hint on failure.
- ``web_search_stack_trace(stderr)`` — one-shot LLM call (Gemini + Google search)
  that appends ``"This is how you can fix the error: …"`` to a traceback.
- ``_build_resource_header()`` — Python prefix that caps BLAS thread counts.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
import time

import weave
from dotenv import load_dotenv
from pydantic import BaseModel

from project_config import get_config
from tools.helpers import call_llm
from utils.llm_utils import append_message
from utils.output import truncate_for_llm


load_dotenv()


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
for _noisy in ("httpcore", "httpx", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


_CONFIG = get_config()
_MAIN_AGENT_MODEL = _CONFIG["llm"]["main_agent_model"]


# ---------------------------------------------------------------------------
# Stack-trace enrichment
# ---------------------------------------------------------------------------


class StackTraceSolution(BaseModel):
    """Structured verdict from the stack-trace debugging LLM."""

    reasoning: str
    web_search_findings: str
    solution: str


_STACK_TRACE_SYSTEM = """You are a Python debugging assistant. Analyze a traceback from the `<query>` field, using web search for supporting information.

## Constraints
- Do not recommend downgrading packages except as a last resort.
"""


@weave.op()
def web_search_stack_trace(stderr: str) -> str:
    """Annotate a traceback with a web-searched remediation hint.

    Dispatches a single Gemini call with Google-search grounding against the
    traceback in ``stderr``. Returns the original stderr followed by
    ``\\n\\nThis is how you can fix the error:\\n<solution>``.

    If the call fails for any reason, returns the raw stderr unchanged —
    enrichment is best-effort.
    """
    logger.info("Dispatching web search for stack trace remediation guidance.")
    trace_index = stderr.find("Traceback (most recent call last)")
    query = stderr[trace_index:] if trace_index != -1 else stderr
    logger.debug("Stack trace query: %s", query)

    try:
        response = call_llm(
            model=_MAIN_AGENT_MODEL,
            system_instruction=_STACK_TRACE_SYSTEM,
            messages=[append_message("user", "<query>\n" + query + "\n</query>")],
            enable_google_search=True,
            text_format=StackTraceSolution,
        )
    except Exception:
        logger.exception("web_search_stack_trace failed; returning raw stderr")
        return stderr

    return truncate_for_llm(
        stderr
        + "\n\nThis is how you can fix the error:\n"
        + response.solution.strip()
    )


# ---------------------------------------------------------------------------
# Subprocess execution
# ---------------------------------------------------------------------------


def _build_resource_header() -> str:
    """Python prefix that caps BLAS thread counts for subprocess scripts."""
    return 'import os\nos.environ["OPENBLAS_NUM_THREADS"] = "32"\n\n'


def _stream_reader(stream, buffer: list, timestamp_ref: list):
    """Read lines from a subprocess stream into a buffer, updating last-output time."""
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
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except ProcessLookupError:
        pass


class ExecutionJob:
    """Handle for a non-blocking subprocess execution."""

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
        return self._proc.poll() is not None

    def result(self) -> str:
        """Block until the process finishes.

        Returns stdout on success, or stderr enriched via
        ``web_search_stack_trace`` on failure.
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
        combined = self._stdout_buf + self._stderr_buf
        return "".join(combined[-n:])

    def idle_time(self) -> float:
        return time.monotonic() - self._last_output_time[0]

    def elapsed(self) -> float:
        return time.monotonic() - self._start_time

    @property
    def pid(self) -> int:
        return self._proc.pid

    def check_timeout(self) -> bool:
        return self.elapsed() > self._timeout_seconds


@weave.op()
def execute_code(filepath: str, timeout_seconds: int = 300) -> "ExecutionJob":
    """Launch a Python file in a new process group and return a job handle.

    Args:
        filepath: Path to the Python file to execute.
        timeout_seconds: Hard timeout (default 300s).
    """
    cmd = ["python", "-u", filepath]
    logger.info("Executing script: %s (timeout: %d seconds)", filepath, timeout_seconds)
    logger.debug("Subprocess command: %s", " ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    return ExecutionJob(proc, timeout_seconds, filepath)
