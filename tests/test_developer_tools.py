"""Unit tests for DeveloperAgent tools."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import tempfile
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
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("print('Success!')\n")
        f.write("print('Script completed')\n")
        script_path = f.name

    yield script_path

    # Cleanup
    Path(script_path).unlink(missing_ok=True)


@pytest.fixture
def test_script_error():
    """Create a temporary Python script that raises an error."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("raise ValueError('Test error message')\n")
        script_path = f.name

    yield script_path

    # Cleanup
    Path(script_path).unlink(missing_ok=True)


@pytest.fixture
def test_script_timeout():
    """Create a temporary Python script that times out."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("import time\n")
        f.write("time.sleep(10)\n")  # Sleep 10 seconds
        f.write("print('Done')\n")
        script_path = f.name

    yield script_path

    # Cleanup
    Path(script_path).unlink(missing_ok=True)


def test_execute_code_success(test_script_success):
    """Test successful code execution."""
    job = execute_code(test_script_success, timeout_seconds=10)
    assert isinstance(job, ExecutionJob)
    result = job.result()

    assert "Success!" in result
    assert "Script completed" in result

    print("✅ execute_code with successful script works:")
    print(f"   - Output length: {len(result)} chars")


def test_execute_code_timeout(test_script_timeout):
    """Test code execution timeout via job.check_timeout() + kill()."""
    job = execute_code(test_script_timeout, timeout_seconds=3)

    import time
    # Wait for timeout to be reached
    time.sleep(4)
    assert job.check_timeout()
    result = job.kill("Hard timeout exceeded")

    assert "killed" in result.lower() or "timeout" in result.lower()

    print("✅ execute_code timeout detection works:")
    print(f"   - Timeout message: {result[:80]}")


def test_execute_code_error_with_web_search(test_script_error, monkeypatch):
    """Test code execution with error triggers web search."""
    def fake_web_search(query):
        return query + "\nThis is how you can fix the error: \nMock solution for test error"

    monkeypatch.setattr("tools.developer.web_search_stack_trace", fake_web_search)

    job = execute_code(test_script_error, timeout_seconds=10)
    result = job.result()

    assert "ValueError" in result or "Test error message" in result
    assert "This is how you can fix the error" in result
    assert "Mock solution" in result

    print("✅ execute_code with error and web search works:")
    print(f"   - Result length: {len(result)} chars")


def test_web_search_stack_trace(monkeypatch):
    """Test web_search_stack_trace with mocked LLM call."""
    from schemas.developer import StackTraceSolution

    def fake_call_llm(*args, **kwargs):
        return StackTraceSolution(
            checklist=["Check X", "Verify Y"],
            web_search_findings="Found solution on Stack Overflow",
            reasoning_and_solution="The error is caused by X. You can fix it by doing Y.",
            validation="Verify the fix works by running Z",
            further_steps="Consider refactoring the code"
        )

    monkeypatch.setattr("tools.developer.call_llm", fake_call_llm)

    query = "Traceback (most recent call last):\n  File 'test.py', line 1\nValueError: test"
    result = web_search_stack_trace(query)

    assert "Traceback" in result
    assert "This is how you can fix the error" in result
    assert "doing Y" in result

    print("✅ web_search_stack_trace works:")
    print(f"   - Result length: {len(result)} chars")


def test_web_search_stack_trace_fallback(monkeypatch):
    """Test web_search_stack_trace fallback when structured parsing fails."""
    def fake_call_llm_raise(*args, **kwargs):
        raise AttributeError("reasoning_and_solution")

    def fake_call_llm_fallback(*args, **kwargs):
        class FallbackResponse:
            text = "Fallback solution text"
        return FallbackResponse()

    # First call (fine-tuned model) raises, fallback call succeeds
    call_count = [0]
    def fake_call_llm(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise AttributeError("reasoning_and_solution")
        class FallbackResponse:
            text = "Fallback solution text"
        return FallbackResponse()

    monkeypatch.setattr("tools.developer.call_llm", fake_call_llm)

    query = "Traceback (most recent call last):\nTypeError: cannot convert"
    result = web_search_stack_trace(query)

    assert "Fallback solution text" in result
    assert "This is how you can fix the error" in result

    print("✅ web_search_stack_trace fallback works:")
    print(f"   - Used fallback text successfully")


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

    print("✅ search_red_flags works:")
    print(f"   - Result length: {len(result)} chars")


def test_search_sota_suggestions(monkeypatch):
    """Test search_sota_suggestions with mocked LLM call."""
    from schemas.developer import SOTAResponse

    def fake_call_llm(*args, **kwargs):
        # When text_format is provided, return the Pydantic object directly
        if kwargs.get('text_format'):
            return SOTAResponse(
                blacklist=False,
                blacklist_reason="The previous suggestion is still valid",
                suggestion="Try using a transformer-based architecture with attention mechanisms",
                suggestion_reason="Transformers have shown SOTA results on similar tasks",
                suggestion_code="# Example transformer code\nmodel = TransformerModel()"
            )
        # Return a Gemini-like response with no function calls
        mock_response = MagicMock(spec=[])
        mock_response.candidates = None  # No candidates triggers the else branch
        mock_response.text = "Mock SOTA suggestions"
        return mock_response

    monkeypatch.setattr("tools.developer.call_llm", fake_call_llm)

    result = search_sota_suggestions(
        description="Test competition",
        context="Current context",
        red_flags="Some red flags identified",
        executed_suggestion="Previous suggestion",
        failed_ideas=["Idea 1", "Idea 2"],
        later_recommendations=None,
        shared_suggestions=None,
        external_data_listing=None,
        plan_content=None
    )

    assert hasattr(result, 'suggestion')
    assert "transformer" in result.suggestion.lower()

    print("✅ search_sota_suggestions works:")
    print(f"   - Suggestion: {result.suggestion[:50]}...")


def test_search_sota_suggestions_with_plan_content(monkeypatch):
    """Test search_sota_suggestions includes plan content in prompt."""
    from schemas.developer import SOTAResponse

    captured_kwargs = {}

    def fake_call_llm(*args, **kwargs):
        captured_kwargs.update(kwargs)
        if kwargs.get('text_format'):
            return SOTAResponse(
                blacklist=False,
                blacklist_reason="Continue with current approach",
                suggestion="Use the plan's recommended approach",
                suggestion_reason="Following the research plan systematically",
                suggestion_code="# Following the plan\nmodel = PlanRecommendedModel()"
            )
        mock_response = MagicMock()
        mock_response.text = "Mock response"
        return mock_response

    monkeypatch.setattr("tools.developer.call_llm", fake_call_llm)

    plan_content = "# Research Plan\n- Step 1: Try XGBoost\n- Step 2: Try Neural Networks"

    result = search_sota_suggestions(
        description="Test competition",
        context="Current context",
        red_flags="Red flags",
        executed_suggestion="Previous suggestion",
        failed_ideas=[],
        later_recommendations="Try feature engineering",
        shared_suggestions=None,
        external_data_listing=None,
        plan_content=plan_content
    )

    messages = captured_kwargs.get('messages', [])
    assert len(messages) > 0
    # Gemini format uses 'parts'
    user_message = messages[0]['parts'][0]['text']
    assert "<plan>" in user_message
    assert "XGBoost" in user_message
    assert "<suggestions>" in user_message
    assert "feature engineering" in user_message

    print("✅ search_sota_suggestions with plan content works:")
    print(f"   - Plan content included in prompt")
    print(f"   - Later recommendations included in prompt")


def test_search_sota_suggestions_with_tools(monkeypatch, test_data_dir):
    """Test search_sota_suggestions with tool calling enabled."""
    from schemas.developer import SOTAResponse

    def fake_call_llm(*args, **kwargs):
        if kwargs.get('text_format'):
            return SOTAResponse(
                blacklist=False,
                blacklist_reason="Based on findings, previous approach is valid",
                suggestion="Add stratified sampling based on target distribution",
                suggestion_reason="Data shows imbalanced target, stratification will help",
                suggestion_code="# Stratified sampling\nfrom sklearn.model_selection import StratifiedKFold"
            )
        # Return response with no tool calls (text-only)
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock()]
        mock_response.candidates[0].content.parts[0].function_call = None
        mock_response.candidates[0].content.parts[0].text = "Analyzing data..."
        mock_response.text = "Analyzing data..."
        mock_response.suggestion = "Add stratified sampling based on target distribution"
        return mock_response

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

    assert hasattr(result, 'suggestion')
    assert "stratified" in result.suggestion.lower() or "sampling" in result.suggestion.lower()

    print("✅ search_sota_suggestions with tools works:")
    print(f"   - Final suggestion returned successfully")
    print(f"   - Result has expected structure")


def test_search_sota_suggestions_early_exit_forces_structured_output(monkeypatch):
    """Test that early exit (no tool calls on step 2) still returns structured output."""
    from schemas.developer import SOTAResponse

    def fake_call_llm(*args, **kwargs):
        if kwargs.get('text_format'):
            return SOTAResponse(
                blacklist=False,
                blacklist_reason="Based on findings",
                suggestion="Use calibration based on EDA",
                suggestion_reason="Data shows miscalibration",
                suggestion_code="# Calibration code\nfrom sklearn.calibration import CalibratedClassifierCV"
            )
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock()]
        mock_response.candidates[0].content.parts[0].function_call = None
        mock_response.candidates[0].content.parts[0].text = "No tools needed."
        mock_response.text = "No tools needed."
        mock_response.suggestion = "Use calibration based on EDA"
        return mock_response

    monkeypatch.setattr("tools.developer.call_llm", fake_call_llm)

    result = search_sota_suggestions(
        description="Test competition",
        context="Current context",
        red_flags="Class imbalance",
        executed_suggestion="Baseline",
        failed_ideas=[],
        slug="test-comp",
        data_path="/tmp/test",
    )

    assert hasattr(result, 'suggestion')
    assert "calibration" in result.suggestion.lower()

    print("✅ Early exit correctly forces structured output:")
    print(f"   - Structured output obtained successfully")


def test_search_sota_suggestions_without_tools(monkeypatch):
    """Test search_sota_suggestions without slug/data_path (tools disabled)."""
    from schemas.developer import SOTAResponse

    call_count = [0]

    def fake_call_llm(*args, **kwargs):
        call_count[0] += 1
        if kwargs.get('text_format'):
            return SOTAResponse(
                blacklist=False,
                blacklist_reason="No tools available, using web search only",
                suggestion="Try ensemble methods",
                suggestion_reason="Common SOTA approach",
                suggestion_code="# Ensemble approach\nfrom sklearn.ensemble import VotingClassifier"
            )
        mock_response = MagicMock()
        mock_response.text = "Suggestion based on web search"
        return mock_response

    monkeypatch.setattr("tools.developer.call_llm", fake_call_llm)

    result = search_sota_suggestions(
        description="Test competition",
        context="Current context",
        red_flags="Some issues",
        executed_suggestion="Previous",
        failed_ideas=[],
    )

    assert hasattr(result, 'suggestion')

    print("✅ search_sota_suggestions without tools works:")
    print(f"   - LLM calls made: {call_count[0]}")


def test_encode_image_to_data_url_basic(test_data_dir):
    """Test basic image encoding."""
    from utils.llm_utils import encode_image_to_data_url
    from PIL import Image

    img_path = Path(test_data_dir) / "test_image.png"
    img = Image.new('RGB', (100, 100), color='red')
    img.save(img_path)

    result = encode_image_to_data_url(str(img_path))

    assert result.startswith("data:image/png;base64,")
    assert len(result) > 50

    print("✅ encode_image_to_data_url basic encoding works")


# ---------------------------------------------------------------------------
# ExecutionJob + monitor_logs tests
# ---------------------------------------------------------------------------

def test_execute_code_returns_job():
    """execute_code() returns an ExecutionJob, not a string."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("print('hello')\n")
        script_path = f.name

    try:
        job = execute_code(script_path, timeout_seconds=10)
        assert isinstance(job, ExecutionJob)
        result = job.result()
        assert "hello" in result
    finally:
        Path(script_path).unlink(missing_ok=True)

    print("✅ execute_code returns ExecutionJob")


def test_execution_job_done_and_result():
    """Job lifecycle: not done immediately, done after completion, result() returns output."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
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

    print("✅ ExecutionJob done() and result() work correctly")


def test_execution_job_kill():
    """job.kill() terminates the process and returns diagnostic message."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
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

    print("✅ ExecutionJob kill() works")


def test_execution_job_recent_output_and_idle():
    """recent_output() streams in real time, idle_time() tracks silence."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("import time\nprint('start', flush=True)\ntime.sleep(5)\nprint('end', flush=True)\n")
        script_path = f.name

    try:
        job = execute_code(script_path, timeout_seconds=30)
        import time
        time.sleep(0.5)  # let 'start' arrive
        assert "start" in job.recent_output()

        time.sleep(2)  # wait through the silence (script sleeps 5s, so 'end' hasn't printed)
        idle = job.idle_time()
        assert idle > 1.0, f"Expected idle > 1.0s, got {idle:.2f}s"

        job.result()  # wait for completion
        assert "end" in job.recent_output()
    finally:
        Path(script_path).unlink(missing_ok=True)

    print("✅ ExecutionJob recent_output() and idle_time() work")


def test_execution_job_process_group_kill():
    """kill() kills child processes too via process group."""
    import os

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
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

    print(f"✅ ExecutionJob kills child processes (child {child_pid} killed)")


def test_monitor_logs_kill(monkeypatch):
    """monitor_logs() returns kill verdict when LLM says kill."""
    kill_verdict = LogMonitorVerdict(action="kill", reason="NaN loss detected at epoch 4")

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

    print("✅ monitor_logs returns kill verdict")


def test_monitor_logs_continue(monkeypatch):
    """monitor_logs() returns continue verdict when training is healthy."""
    continue_verdict = LogMonitorVerdict(action="continue", reason="Loss decreasing normally")

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

    print("✅ monitor_logs returns continue verdict")


def test_monitor_logs_with_bash_tool(monkeypatch):
    """monitor_logs() handles execute_bash tool calls from the LLM."""
    call_count = [0]

    def fake_call_llm(*args, **kwargs):
        call_count[0] += 1

        # First call: LLM wants to use execute_bash tool
        if call_count[0] == 1:
            mock_fc = MagicMock()
            mock_fc.name = "execute_bash"
            mock_fc.args = {"command": "echo ok"}
            mock_fc.id = "call_123"

            mock_part = MagicMock(spec=['function_call'])
            mock_part.function_call = mock_fc

            mock_content = MagicMock()
            mock_content.parts = [mock_part]

            mock_candidate = MagicMock()
            mock_candidate.content = mock_content

            mock_response = MagicMock(spec=['candidates'])
            mock_response.candidates = [mock_candidate]
            return mock_response

        # Second call: LLM returns verdict after seeing tool output
        return LogMonitorVerdict(
            action="kill",
            reason="GPU at 0% utilization — process is deadlocked"
        )

    monkeypatch.setattr("tools.developer.call_llm", fake_call_llm)

    verdict = monitor_logs(
        log_output="",
        seconds_since_last_output=600.0,
        total_elapsed_seconds=900.0,
        pid=12345,
    )

    assert call_count[0] >= 2, f"Expected at least 2 LLM calls (tool use + verdict), got {call_count[0]}"
    assert verdict.action == "kill"
    assert "deadlocked" in verdict.reason.lower()

    print("✅ monitor_logs handles execute_bash tool calls")


def test_monitor_logs_error_defaults_to_continue(monkeypatch):
    """monitor_logs() defaults to continue when the LLM call fails."""
    def fake_call_llm(*args, **kwargs):
        raise RuntimeError("API error")

    monkeypatch.setattr("tools.developer.call_llm", fake_call_llm)

    verdict = monitor_logs(
        log_output="Epoch 1/100 - loss: 2.345\n",
        seconds_since_last_output=5.0,
        total_elapsed_seconds=60.0,
        pid=12345,
    )

    assert verdict.action == "continue"

    print("✅ monitor_logs defaults to continue on error")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
