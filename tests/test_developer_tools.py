"""Unit tests for shared developer tools (tools/developer.py)."""

import logging

import pytest
import tempfile
from pathlib import Path

from tools import developer
from tools.developer import (
    execute_code,
    web_search_stack_trace,
    ExecutionJob,
)


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

    query = (
        "Traceback (most recent call last):\n  File 'test.py', line 1\nValueError: test"
    )
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


def test_monitor_failure_never_kills_a_healthy_job(monkeypatch, tmp_path):
    """A monitor that cannot produce a verdict must not terminate the job.

    The monitor runs as a `codex exec` subprocess; it can fail because codex is
    missing, rate-limited, or returned something unparseable. None of those are
    evidence the training run is unhealthy, and killing on them would end a
    12-hour job for an unrelated reason.
    """
    script = tmp_path / "train.py"
    script.write_text(
        "import time\n"
        "for i in range(3):\n"
        "    print('epoch', i, 'loss', 1.0 / (i + 1), flush=True)\n"
        "    time.sleep(0.05)\n",
        encoding="utf-8",
    )

    def unavailable(**kwargs):
        raise developer.MonitorUnavailableError("codex exec exited 1")

    monkeypatch.setattr(developer, "monitor_logs", unavailable)

    output = developer.execute_with_monitor(
        code_path=script,
        timeout_seconds=30,
        log_monitor_interval=0,
        logger=logging.getLogger(__name__),
    )

    assert "epoch 2" in output, "job must run to completion despite monitor failure"
    assert "killed" not in output.lower()
