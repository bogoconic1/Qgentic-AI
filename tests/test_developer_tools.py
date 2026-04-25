"""Unit tests for shared developer tools (tools/developer.py)."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from tools.developer import (
    execute_code,
    monitor_logs,
    web_search_stack_trace,
    ExecutionJob,
)
from schemas.developer import LogMonitorVerdict


@pytest.fixture
def test_data_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def test_script_success():
    """Create a temporary Python script that succeeds."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("print('Success!')\n")
        f.write("print('Script completed')\n")
        script_path = f.name

    yield script_path
    Path(script_path).unlink(missing_ok=True)


@pytest.fixture
def test_script_error():
    """Create a temporary Python script that raises an error."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("raise ValueError('Test error message')\n")
        script_path = f.name

    yield script_path
    Path(script_path).unlink(missing_ok=True)


@pytest.fixture
def test_script_timeout():
    """Create a temporary Python script that times out."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("import time\n")
        f.write("time.sleep(10)\n")
        f.write("print('Done')\n")
        script_path = f.name

    yield script_path
    Path(script_path).unlink(missing_ok=True)


def test_execute_code_success(test_script_success):
    """Test successful code execution."""
    job = execute_code(test_script_success, timeout_seconds=10)
    assert isinstance(job, ExecutionJob)
    result = job.result()

    assert "Success!" in result
    assert "Script completed" in result


def test_execute_code_timeout(test_script_timeout):
    """Test code execution timeout via job.check_timeout() + kill()."""
    job = execute_code(test_script_timeout, timeout_seconds=3)

    import time

    time.sleep(4)
    assert job.check_timeout()
    result = job.kill("Hard timeout exceeded")

    assert "killed" in result.lower() or "timeout" in result.lower()


def test_execute_code_result_honors_timeout():
    """Regression for #256: a hung snippet must not block .result() forever.

    Before the fix, ExecutionJob.result() called self._proc.wait() with no
    timeout, so an infinite-loop snippet would pin the parent indefinitely.
    """
    import time

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("import time\nwhile True:\n    time.sleep(0.05)\n")
        script_path = f.name

    try:
        t0 = time.monotonic()
        job = execute_code(script_path, timeout_seconds=2)
        result = job.result()
        elapsed = time.monotonic() - t0

        assert "timeout" in result.lower(), (
            f"expected timeout diagnostic in output, got: {result[:200]!r}"
        )
        assert elapsed < 5, (
            f"result() should return within ~3s of the timeout; took {elapsed:.1f}s"
        )
    finally:
        Path(script_path).unlink(missing_ok=True)


def test_execute_code_error_returns_raw_stderr(test_script_error):
    """Test that a failing script returns raw stderr (no enrichment).

    Stack-trace enrichment via ``web_search_stack_trace`` is now the caller's
    responsibility; ``ExecutionJob.result()`` returns the unfiltered stream
    content.
    """
    job = execute_code(test_script_error, timeout_seconds=10)
    result = job.result()

    assert "ValueError" in result or "Test error message" in result
    assert "This is how you can fix the error" not in result


def test_web_search_stack_trace(monkeypatch):
    """Test web_search_stack_trace with mocked LLM call."""
    from schemas.developer import StackTraceSolution

    call_count = [0]

    def fake_call_llm(*args, **kwargs):
        call_count[0] += 1
        return StackTraceSolution(
            reasoning="The error is caused by X.",
            web_search_findings="Found solution on Stack Overflow",
            solution="The error is caused by X. You can fix it by doing Y.",
        )

    monkeypatch.setattr("tools.developer.call_llm", fake_call_llm)

    query = "Traceback (most recent call last):\n  File 'test.py', line 1\nValueError: test"
    result = web_search_stack_trace(query)

    assert call_count[0] == 1, "Should call the LLM exactly once (web-search path only)"
    assert "Traceback" in result
    assert "This is how you can fix the error" in result
    assert "doing Y" in result


def test_encode_image_to_data_url_basic(test_data_dir):
    """Test basic image encoding."""
    from utils.llm_utils import encode_image_to_data_url
    from PIL import Image

    img_path = Path(test_data_dir) / "test_image.png"
    img = Image.new("RGB", (100, 100), color="red")
    img.save(img_path)

    result = encode_image_to_data_url(str(img_path))

    assert result.startswith("data:image/png;base64,")
    assert len(result) > 50


# ---------------------------------------------------------------------------
# ExecutionJob + monitor_logs tests
# ---------------------------------------------------------------------------


def test_execute_code_returns_job():
    """execute_code() returns an ExecutionJob, not a string."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("print('hello')\n")
        script_path = f.name

    try:
        job = execute_code(script_path, timeout_seconds=10)
        assert isinstance(job, ExecutionJob)
        result = job.result()
        assert "hello" in result
    finally:
        Path(script_path).unlink(missing_ok=True)


def test_execution_job_done_and_result():
    """Job lifecycle: not done immediately, done after completion, result() returns output."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("import time\ntime.sleep(0.5)\nprint('finished')\n")
        script_path = f.name

    try:
        job = execute_code(script_path, timeout_seconds=10)
        import time

        time.sleep(0.1)
        result = job.result()
        assert job.done()
        assert "finished" in result
    finally:
        Path(script_path).unlink(missing_ok=True)


def test_execution_job_kill():
    """job.kill() terminates the process and returns diagnostic message."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("import time\nprint('running', flush=True)\ntime.sleep(300)\n")
        script_path = f.name

    try:
        job = execute_code(script_path, timeout_seconds=600)
        import time

        time.sleep(0.5)
        msg = job.kill("NaN loss detected")
        assert job.done()
        assert "NaN loss detected" in msg
        assert "running" in msg
    finally:
        Path(script_path).unlink(missing_ok=True)


def test_execution_job_recent_output_and_idle():
    """recent_output() streams in real time, idle_time() tracks silence."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(
            "import time\nprint('start', flush=True)\ntime.sleep(5)\nprint('end', flush=True)\n"
        )
        script_path = f.name

    try:
        job = execute_code(script_path, timeout_seconds=30)
        import time

        time.sleep(0.5)
        assert "start" in job.recent_output()

        time.sleep(2)
        idle = job.idle_time()
        assert idle > 1.0, f"Expected idle > 1.0s, got {idle:.2f}s"

        job.result()
        assert "end" in job.recent_output()
    finally:
        Path(script_path).unlink(missing_ok=True)


def test_execution_job_process_group_kill():
    """kill() kills child processes too via process group."""
    import os

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(
            "import subprocess, sys, time\n"
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])\n"
            "print(f'child_pid={child.pid}', flush=True)\n"
            "time.sleep(300)\n"
        )
        script_path = f.name

    try:
        job = execute_code(script_path, timeout_seconds=600)
        import time

        time.sleep(1)

        output = job.recent_output()
        child_pid = None
        for line in output.splitlines():
            if "child_pid=" in line:
                child_pid = int(line.split("=")[1])
                break
        assert child_pid is not None, "Could not find child PID"

        job.kill("test cleanup")

        time.sleep(0.5)
        try:
            os.kill(child_pid, 0)
            assert False, f"Child {child_pid} still alive"
        except ProcessLookupError:
            pass  # expected

    finally:
        Path(script_path).unlink(missing_ok=True)


def test_monitor_logs_kill(monkeypatch):
    """monitor_logs() returns kill verdict when LLM says kill."""
    kill_verdict = LogMonitorVerdict(
        reasoning="NaN loss detected at epoch 4",
        action="kill",
    )

    def fake_call_llm(*args, **kwargs):
        return kill_verdict

    monkeypatch.setattr("tools.developer.call_llm", fake_call_llm)

    verdict = monitor_logs(
        log_output="Epoch 4/100 - loss: nan - acc: 0.00\n",
        seconds_since_last_output=5.0,
        total_elapsed_seconds=240.0,
        pid=12345,
    )

    assert verdict.action == "kill"
    assert "NaN" in verdict.reasoning


def test_monitor_logs_continue(monkeypatch):
    """monitor_logs() returns continue verdict when training is healthy."""
    continue_verdict = LogMonitorVerdict(
        reasoning="Loss decreasing normally",
        action="continue",
    )

    def fake_call_llm(*args, **kwargs):
        return continue_verdict

    monkeypatch.setattr("tools.developer.call_llm", fake_call_llm)

    verdict = monitor_logs(
        log_output="Epoch 2/100 - loss: 1.876 - acc: 0.34\n",
        seconds_since_last_output=10.0,
        total_elapsed_seconds=120.0,
        pid=12345,
    )

    assert verdict.action == "continue"


def test_monitor_logs_with_bash_tool(monkeypatch):
    """monitor_logs() handles execute_bash tool calls from the LLM."""
    call_count = [0]

    def fake_call_llm(*args, **kwargs):
        call_count[0] += 1

        if call_count[0] == 1:
            mock_fc = MagicMock()
            mock_fc.name = "execute_bash"
            mock_fc.args = {"command": "echo ok"}
            mock_fc.id = "call_123"

            mock_part = MagicMock(spec=["function_call"])
            mock_part.function_call = mock_fc

            mock_content = MagicMock()
            mock_content.parts = [mock_part]

            mock_candidate = MagicMock()
            mock_candidate.content = mock_content

            mock_response = MagicMock(spec=["candidates"])
            mock_response.candidates = [mock_candidate]
            return mock_response

        return LogMonitorVerdict(
            reasoning="GPU at 0% utilization — process is deadlocked",
            action="kill",
        )

    monkeypatch.setattr("tools.developer.call_llm", fake_call_llm)

    verdict = monitor_logs(
        log_output="",
        seconds_since_last_output=600.0,
        total_elapsed_seconds=900.0,
        pid=12345,
    )

    assert call_count[0] >= 2
    assert verdict.action == "kill"
    assert "deadlocked" in verdict.reasoning.lower()
