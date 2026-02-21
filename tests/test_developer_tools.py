"""Unit tests for DeveloperAgent tools."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from tools.developer import (
    execute_code,
    monitor_logs,
    web_search_stack_trace,
    search_red_flags,
    search_sota_suggestions,
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


def test_execute_code_error_with_web_search(test_script_error, monkeypatch):
    """Test code execution with error triggers web search."""

    def fake_web_search(query):
        return (
            query
            + "\nThis is how you can fix the error: \nMock solution for test error"
        )

    monkeypatch.setattr("tools.developer.web_search_stack_trace", fake_web_search)

    job = execute_code(test_script_error, timeout_seconds=10)
    result = job.result()

    assert "ValueError" in result or "Test error message" in result
    assert "This is how you can fix the error" in result
    assert "Mock solution" in result


def test_web_search_stack_trace(monkeypatch):
    """Test web_search_stack_trace with mocked LLM call."""
    from schemas.developer import StackTraceSolution

    def fake_call_llm(*args, **kwargs):
        return StackTraceSolution(
            checklist=["Check X", "Verify Y"],
            web_search_findings="Found solution on Stack Overflow",
            reasoning_and_solution="The error is caused by X. You can fix it by doing Y.",
            validation="Verify the fix works by running Z",
            further_steps="Consider refactoring the code",
        )

    monkeypatch.setattr("tools.developer.call_llm", fake_call_llm)

    query = "Traceback (most recent call last):\n  File 'test.py', line 1\nValueError: test"
    result = web_search_stack_trace(query)

    assert "Traceback" in result
    assert "This is how you can fix the error" in result
    assert "doing Y" in result


def test_web_search_stack_trace_fallback(monkeypatch):
    """Test web_search_stack_trace falls back to web search when fine-tuned model response is too short."""
    from schemas.developer import StackTraceSolution

    call_count = [0]

    def fake_call_llm(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # Fine-tuned model returns a too-short response (< 35 chars)
            return StackTraceSolution(
                checklist=[],
                web_search_findings="",
                reasoning_and_solution="I cannot solve this error.",
                validation="",
                further_steps="",
            )
        # Web search fallback returns a proper solution
        return StackTraceSolution(
            checklist=["Check types"],
            web_search_findings="Found on SO",
            reasoning_and_solution="Fallback solution text that is long enough to pass validation",
            validation="Run tests",
            further_steps="Refactor",
        )

    monkeypatch.setattr("tools.developer.call_llm", fake_call_llm)

    query = "Traceback (most recent call last):\nTypeError: cannot convert"
    result = web_search_stack_trace(query)

    assert call_count[0] == 2, "Should have called LLM twice (fine-tuned + fallback)"
    assert "Fallback solution text" in result
    assert "This is how you can fix the error" in result


def test_search_red_flags(monkeypatch):
    """Test search_red_flags with mocked LLM call."""
    red_flags_text = (
        "# Red Flags Analysis\n\n"
        "## Critical Issues\n"
        "1. Data leakage detected in preprocessing\n"
        "2. Overfitting on validation set\n"
    )

    def fake_call_llm(*args, **kwargs):
        mock_response = MagicMock()
        mock_response.text = red_flags_text
        return mock_response

    monkeypatch.setattr("tools.developer.call_llm", fake_call_llm)

    description = "Test competition description"
    context = "Current code context"

    result = search_red_flags(description, context)

    assert "Red Flags" in result or "Data leakage" in result
    assert len(result) > 0


def _make_no_tool_call_response():
    """Build a mock Gemini response with no function calls (triggers early exit to structured output)."""
    mock_part = MagicMock(spec=["function_call", "text"])
    mock_part.function_call = None
    mock_part.text = "Analyzing..."
    mock_content = MagicMock(spec=["parts"])
    mock_content.parts = [mock_part]
    mock_candidate = MagicMock(spec=["content"])
    mock_candidate.content = mock_content
    mock_response = MagicMock(spec=["candidates", "text"])
    mock_response.candidates = [mock_candidate]
    mock_response.text = "Analyzing..."
    return mock_response


def test_search_sota_suggestions(monkeypatch, test_data_dir):
    """Test search_sota_suggestions with mocked LLM call."""
    from schemas.developer import SOTAResponse

    def fake_call_llm(*args, **kwargs):
        if kwargs.get("text_format"):
            return SOTAResponse(
                blacklist=False,
                blacklist_reason="The previous suggestion is still valid",
                suggestion="Try using a transformer-based architecture with attention mechanisms",
                suggestion_reason="Transformers have shown SOTA results on similar tasks",
                suggestion_code="# Example transformer code\nmodel = TransformerModel()",
            )
        return _make_no_tool_call_response()

    monkeypatch.setattr("tools.developer.call_llm", fake_call_llm)

    result = search_sota_suggestions(
        description="Test competition",
        context="Current context",
        red_flags="Some red flags identified",
        executed_suggestion="Previous suggestion",
        failed_ideas=["Idea 1", "Idea 2"],
        slug="test-competition",
        data_path=test_data_dir,
        later_recommendations=None,
        shared_suggestions=None,
        external_data_listing=None,
        plan_content=None,
    )

    assert hasattr(result, "suggestion")
    assert "transformer" in result.suggestion.lower()


def test_search_sota_suggestions_with_plan_content(monkeypatch, test_data_dir):
    """Test search_sota_suggestions includes plan content in prompt."""
    from schemas.developer import SOTAResponse

    captured_kwargs = {}

    def fake_call_llm(*args, **kwargs):
        captured_kwargs.update(kwargs)
        if kwargs.get("text_format"):
            return SOTAResponse(
                blacklist=False,
                blacklist_reason="Continue with current approach",
                suggestion="Use the plan's recommended approach",
                suggestion_reason="Following the research plan systematically",
                suggestion_code="# Following the plan\nmodel = PlanRecommendedModel()",
            )
        return _make_no_tool_call_response()

    monkeypatch.setattr("tools.developer.call_llm", fake_call_llm)

    plan_content = (
        "# Research Plan\n- Step 1: Try XGBoost\n- Step 2: Try Neural Networks"
    )

    result = search_sota_suggestions(
        description="Test competition",
        context="Current context",
        red_flags="Red flags",
        executed_suggestion="Previous suggestion",
        failed_ideas=[],
        slug="test-competition",
        data_path=test_data_dir,
        later_recommendations="Try feature engineering",
        shared_suggestions=None,
        external_data_listing=None,
        plan_content=plan_content,
    )

    messages = captured_kwargs.get("messages", [])
    assert len(messages) > 0
    user_message = messages[0]["parts"][0]["text"]
    assert "<plan>" in user_message
    assert "XGBoost" in user_message
    assert "<suggestions>" in user_message
    assert "feature engineering" in user_message


def test_search_sota_suggestions_with_tools(monkeypatch, test_data_dir):
    """Test search_sota_suggestions with tool calling enabled."""
    from schemas.developer import SOTAResponse

    def fake_call_llm(*args, **kwargs):
        if kwargs.get("text_format"):
            return SOTAResponse(
                blacklist=False,
                blacklist_reason="Based on findings, previous approach is valid",
                suggestion="Add stratified sampling based on target distribution",
                suggestion_reason="Data shows imbalanced target, stratification will help",
                suggestion_code="# Stratified sampling\nfrom sklearn.model_selection import StratifiedKFold",
            )
        return _make_no_tool_call_response()

    monkeypatch.setattr("tools.developer.call_llm", fake_call_llm)

    result = search_sota_suggestions(
        description="Binary classification competition",
        context="Current model shows poor performance on class 1",
        red_flags="Class imbalance detected",
        executed_suggestion="Tried basic model",
        failed_ideas=[],
        slug="test-competition",
        data_path=test_data_dir,
        cpu_core_range=[0, 1],
        gpu_identifier="0",
    )

    assert hasattr(result, "suggestion")
    assert (
        "stratified" in result.suggestion.lower()
        or "sampling" in result.suggestion.lower()
    )


def test_search_sota_suggestions_early_exit_forces_structured_output(
    monkeypatch, test_data_dir
):
    """Test that early exit (no tool calls on step 2) still returns structured output."""
    from schemas.developer import SOTAResponse

    def fake_call_llm(*args, **kwargs):
        if kwargs.get("text_format"):
            return SOTAResponse(
                blacklist=False,
                blacklist_reason="Based on findings",
                suggestion="Use calibration based on EDA",
                suggestion_reason="Data shows miscalibration",
                suggestion_code="# Calibration code\nfrom sklearn.calibration import CalibratedClassifierCV",
            )
        return _make_no_tool_call_response()

    monkeypatch.setattr("tools.developer.call_llm", fake_call_llm)

    result = search_sota_suggestions(
        description="Test competition",
        context="Current context",
        red_flags="Class imbalance",
        executed_suggestion="Baseline",
        failed_ideas=[],
        slug="test-comp",
        data_path=test_data_dir,
    )

    assert hasattr(result, "suggestion")
    assert "calibration" in result.suggestion.lower()


def test_search_sota_suggestions_without_tools(monkeypatch, test_data_dir):
    """Test search_sota_suggestions without tool calling."""
    from schemas.developer import SOTAResponse

    call_count = [0]

    def fake_call_llm(*args, **kwargs):
        call_count[0] += 1
        if kwargs.get("text_format"):
            return SOTAResponse(
                blacklist=False,
                blacklist_reason="No tools available, using web search only",
                suggestion="Try ensemble methods",
                suggestion_reason="Common SOTA approach",
                suggestion_code="# Ensemble approach\nfrom sklearn.ensemble import VotingClassifier",
            )
        return _make_no_tool_call_response()

    monkeypatch.setattr("tools.developer.call_llm", fake_call_llm)

    result = search_sota_suggestions(
        description="Test competition",
        context="Current context",
        red_flags="Some issues",
        executed_suggestion="Previous",
        failed_ideas=[],
        slug="test-competition",
        data_path=test_data_dir,
    )

    assert hasattr(result, "suggestion")


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
        action="kill", reason="NaN loss detected at epoch 4"
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
    assert "NaN" in verdict.reason


def test_monitor_logs_continue(monkeypatch):
    """monitor_logs() returns continue verdict when training is healthy."""
    continue_verdict = LogMonitorVerdict(
        action="continue", reason="Loss decreasing normally"
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
            action="kill",
            reason="GPU at 0% utilization — process is deadlocked",
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
    assert "deadlocked" in verdict.reason.lower()
