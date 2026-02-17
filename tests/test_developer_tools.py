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

    # Verify output contains success messages
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
    # Mock web_search_stack_trace to avoid actual web search
    def fake_web_search(query):
        return query + "\nThis is how you can fix the error: \nMock solution for test error"

    monkeypatch.setattr("tools.developer.web_search_stack_trace", fake_web_search)

    job = execute_code(test_script_error, timeout_seconds=10)
    result = job.result()

    # Verify error is in output
    assert "ValueError" in result or "Test error message" in result
    # Verify web search result is appended
    assert "This is how you can fix the error" in result
    assert "Mock solution" in result

    print("✅ execute_code with error and web search works:")
    print(f"   - Result length: {len(result)} chars")


def test_web_search_stack_trace(monkeypatch):
    """Test web_search_stack_trace with mocked LLM call."""
    from schemas.developer import StackTraceSolution

    def fake_call_llm(*args, **kwargs):
        # For Anthropic/Google providers, return the Pydantic object directly
        return StackTraceSolution(
            checklist=["Check X", "Verify Y"],
            web_search_findings="Found solution on Stack Overflow",
            reasoning_and_solution="The error is caused by X. You can fix it by doing Y.",
            validation="Verify the fix works by running Z",
            further_steps="Consider refactoring the code"
        )

    # Mock all provider-specific LLM calls - IMPORTANT: also mock in tools.helpers
    # because web_search_stack_trace does inline import from tools.helpers
    monkeypatch.setattr("tools.developer.call_llm_with_retry", fake_call_llm)
    monkeypatch.setattr("tools.developer.call_llm_with_retry_anthropic", fake_call_llm)
    monkeypatch.setattr("tools.developer.call_llm_with_retry_google", fake_call_llm)
    monkeypatch.setattr("tools.helpers.call_llm_with_retry_google", fake_call_llm)
    monkeypatch.setattr("tools.developer.detect_provider", lambda x: "openai")

    query = "Traceback (most recent call last):\n  File 'test.py', line 1\nValueError: test"
    result = web_search_stack_trace(query)

    # Verify result format
    assert "Traceback" in result
    assert "This is how you can fix the error" in result
    assert "doing Y" in result

    print("✅ web_search_stack_trace works:")
    print(f"   - Result length: {len(result)} chars")


def test_web_search_stack_trace_fallback(monkeypatch):
    """Test web_search_stack_trace fallback when structured parsing fails."""
    def fake_call_llm_google(*args, **kwargs):
        # Simulate fine-tuned model failing with an attribute error
        raise AttributeError("reasoning_and_solution")

    def fake_call_llm_fallback(*args, **kwargs):
        # For the fallback path, return an object that only has output_text
        # (no reasoning_and_solution) to trigger the fallback to raw content
        class FallbackResponse:
            output_text = "Fallback solution text"
        return FallbackResponse()

    # Mock all provider-specific LLM calls - IMPORTANT: also mock in tools.helpers
    # because web_search_stack_trace does inline import from tools.helpers
    monkeypatch.setattr("tools.developer.call_llm_with_retry", fake_call_llm_fallback)
    monkeypatch.setattr("tools.developer.call_llm_with_retry_anthropic", fake_call_llm_fallback)
    monkeypatch.setattr("tools.developer.call_llm_with_retry_google", fake_call_llm_fallback)
    # Google helper for the fine-tuned model call should fail
    monkeypatch.setattr("tools.helpers.call_llm_with_retry_google", fake_call_llm_google)
    monkeypatch.setattr("tools.developer.detect_provider", lambda x: "openai")

    query = "Traceback (most recent call last):\nTypeError: cannot convert"
    result = web_search_stack_trace(query)

    # Verify fallback to output_text
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
        # Return object with output_text for OpenAI provider path
        mock_response = MagicMock()
        mock_response.output_text = red_flags_text
        return mock_response

    # Force OpenAI provider to use output_text path
    monkeypatch.setattr("tools.developer.detect_provider", lambda x: "openai")
    monkeypatch.setattr("tools.developer.call_llm_with_retry", fake_call_llm)
    monkeypatch.setattr("tools.developer.call_llm_with_retry_anthropic", fake_call_llm)
    monkeypatch.setattr("tools.developer.call_llm_with_retry_google", fake_call_llm)

    description = "Test competition description"
    context = "Current code context"

    result = search_red_flags(description, context)

    # Verify result
    assert "Red Flags" in result or "Data leakage" in result
    assert len(result) > 0

    print("✅ search_red_flags works:")
    print(f"   - Result length: {len(result)} chars")


def test_search_sota_suggestions(monkeypatch):
    """Test search_sota_suggestions with mocked LLM call."""
    from schemas.developer import SOTAResponse

    def fake_call_llm(*args, **kwargs):
        mock_response = MagicMock()
        mock_response.output = []
        # Return structured SOTAResponse with all required fields
        mock_response.output_parsed = SOTAResponse(
            blacklist=False,
            blacklist_reason="The previous suggestion is still valid",
            suggestion="Try using a transformer-based architecture with attention mechanisms",
            suggestion_reason="Transformers have shown SOTA results on similar tasks",
            suggestion_code="# Example transformer code\nmodel = TransformerModel()"
        )
        mock_response.output_text = "Mock SOTA suggestions"
        return mock_response

    # Mock all provider-specific LLM calls
    monkeypatch.setattr("tools.developer.call_llm_with_retry", fake_call_llm)
    monkeypatch.setattr("tools.developer.call_llm_with_retry_anthropic", fake_call_llm)
    monkeypatch.setattr("tools.developer.call_llm_with_retry_google", fake_call_llm)

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

    # Verify result is response object with structured output
    assert hasattr(result, 'output_parsed')
    assert result.output_parsed.blacklist == False
    assert "transformer" in result.output_parsed.suggestion

    print("✅ search_sota_suggestions works:")
    print(f"   - Blacklist: {result.output_parsed.blacklist}")
    print(f"   - Suggestion: {result.output_parsed.suggestion[:50]}...")


def test_search_sota_suggestions_with_plan_content(monkeypatch):
    """Test search_sota_suggestions includes plan content in prompt."""
    from schemas.developer import SOTAResponse

    captured_kwargs = {}

    def fake_call_llm(*args, **kwargs):
        # Capture kwargs to verify plan_content is included
        captured_kwargs.update(kwargs)
        mock_response = MagicMock()
        mock_response.output = []
        mock_response.output_parsed = SOTAResponse(
            blacklist=False,
            blacklist_reason="Continue with current approach",
            suggestion="Use the plan's recommended approach",
            suggestion_reason="Following the research plan systematically",
            suggestion_code="# Following the plan\nmodel = PlanRecommendedModel()"
        )
        mock_response.output_text = "Mock response"
        return mock_response

    # Mock all provider-specific LLM calls
    monkeypatch.setattr("tools.developer.call_llm_with_retry", fake_call_llm)
    monkeypatch.setattr("tools.developer.call_llm_with_retry_anthropic", fake_call_llm)
    monkeypatch.setattr("tools.developer.call_llm_with_retry_google", fake_call_llm)

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

    # Verify plan content was included in the prompt
    messages = captured_kwargs.get('messages', [])
    assert len(messages) > 0
    # Handle both Google ('parts') and OpenAI/Anthropic ('content') formats
    if 'parts' in messages[0]:
        user_message = messages[0]['parts'][0]['text']
    else:
        user_message = messages[0]['content']
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

    def fake_call_llm_anthropic(*args, **kwargs):
        # When text_format is provided (last step), return the Pydantic object directly
        if kwargs.get('text_format'):
            return SOTAResponse(
                blacklist=False,
                blacklist_reason="Based on findings, previous approach is valid",
                suggestion="Add stratified sampling based on target distribution",
                suggestion_reason="Data shows imbalanced target, stratification will help",
                suggestion_code="# Stratified sampling\nfrom sklearn.model_selection import StratifiedKFold"
            )
        # Otherwise return response with stop_reason="end_turn" (no tool use)
        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = []
        # Include suggestion attribute so it returns early
        mock_response.suggestion = "Add stratified sampling based on target distribution"
        return mock_response

    # Force Anthropic provider (when slug/data_path provided, uses Anthropic with tools)
    monkeypatch.setattr("tools.developer.detect_provider", lambda x: "anthropic")
    monkeypatch.setattr("tools.developer.call_llm_with_retry", fake_call_llm_anthropic)
    monkeypatch.setattr("tools.developer.call_llm_with_retry_anthropic", fake_call_llm_anthropic)
    monkeypatch.setattr("tools.developer.call_llm_with_retry_google", fake_call_llm_anthropic)

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

    # Verify final result is SOTAResponse Pydantic object or has suggestion attribute
    assert hasattr(result, 'suggestion')
    assert "stratified" in result.suggestion.lower() or "sampling" in result.suggestion.lower()

    print("✅ search_sota_suggestions with tools works:")
    print(f"   - Final suggestion returned successfully")
    print(f"   - Result has expected structure")


def test_search_sota_suggestions_early_exit_forces_structured_output(monkeypatch):
    """Test that early exit (no tool calls on step 2) still returns structured output."""
    from schemas.developer import SOTAResponse

    def fake_call_llm_anthropic(*args, **kwargs):
        # When text_format is provided (last step), return the Pydantic object directly
        if kwargs.get('text_format'):
            return SOTAResponse(
                blacklist=False,
                blacklist_reason="Based on findings",
                suggestion="Use calibration based on EDA",
                suggestion_reason="Data shows miscalibration",
                suggestion_code="# Calibration code\nfrom sklearn.calibration import CalibratedClassifierCV"
            )
        # Otherwise return response with stop_reason="end_turn" (no tool use)
        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = []
        # Include suggestion attribute so it returns early
        mock_response.suggestion = "Use calibration based on EDA"
        return mock_response

    # Force Anthropic provider (when slug/data_path provided, uses Anthropic with tools)
    monkeypatch.setattr("tools.developer.detect_provider", lambda x: "anthropic")
    monkeypatch.setattr("tools.developer.call_llm_with_retry", fake_call_llm_anthropic)
    monkeypatch.setattr("tools.developer.call_llm_with_retry_anthropic", fake_call_llm_anthropic)
    monkeypatch.setattr("tools.developer.call_llm_with_retry_google", fake_call_llm_anthropic)

    result = search_sota_suggestions(
        description="Test competition",
        context="Current context",
        red_flags="Class imbalance",
        executed_suggestion="Baseline",
        failed_ideas=[],
        slug="test-comp",
        data_path="/tmp/test",
    )

    # Final result should have suggestion attribute
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
        mock_response = MagicMock()
        mock_response.output = []
        mock_response.output_parsed = SOTAResponse(
            blacklist=False,
            blacklist_reason="No tools available, using web search only",
            suggestion="Try ensemble methods",
            suggestion_reason="Common SOTA approach",
            suggestion_code="# Ensemble approach\nfrom sklearn.ensemble import VotingClassifier"
        )
        mock_response.output_text = "Suggestion based on web search"

        # Verify tools are empty when slug/data_path not provided
        assert kwargs.get('tools', []) == []

        return mock_response

    # Mock all provider-specific LLM calls
    monkeypatch.setattr("tools.developer.call_llm_with_retry", fake_call_llm)
    monkeypatch.setattr("tools.developer.call_llm_with_retry_anthropic", fake_call_llm)
    monkeypatch.setattr("tools.developer.call_llm_with_retry_google", fake_call_llm)

    # Call without slug/data_path - tools should be disabled
    result = search_sota_suggestions(
        description="Test competition",
        context="Current context",
        red_flags="Some issues",
        executed_suggestion="Previous",
        failed_ideas=[],
        # No slug/data_path provided
    )

    # Should only make one call (no tool loop)
    assert call_count[0] == 1
    assert hasattr(result, 'output_parsed')

    print("✅ search_sota_suggestions without tools works:")
    print(f"   - Single LLM call made (no tool loop)")
    print(f"   - Tools correctly disabled when slug/data_path missing")


def test_encode_image_to_data_url_basic(test_data_dir):
    """Test basic image encoding without resize."""
    from utils.llm_utils import encode_image_to_data_url
    from PIL import Image

    # Create a small test image
    img_path = Path(test_data_dir) / "test_image.png"
    img = Image.new('RGB', (100, 100), color='red')
    img.save(img_path)

    result = encode_image_to_data_url(str(img_path))

    assert result.startswith("data:image/png;base64,")
    assert len(result) > 50  # Should have base64 data

    print("✅ encode_image_to_data_url basic encoding works")


def test_encode_image_to_data_url_resize_for_anthropic(test_data_dir):
    """Test image encoding with resize for Anthropic's size limits."""
    from utils.llm_utils import encode_image_to_data_url
    from PIL import Image

    # Create a large test image (3000x2000)
    img_path = Path(test_data_dir) / "large_image.png"
    img = Image.new('RGB', (3000, 2000), color='blue')
    img.save(img_path)

    result = encode_image_to_data_url(str(img_path), resize_for_anthropic=True)

    # Should be resized and converted to JPEG
    assert result.startswith("data:image/jpeg;base64,")

    # Decode and verify dimensions are within limits
    import base64
    import io
    b64_data = result.split(";base64,")[1]
    decoded = base64.b64decode(b64_data)
    resized_img = Image.open(io.BytesIO(decoded))

    assert resized_img.width <= 2000
    assert resized_img.height <= 2000

    print("✅ encode_image_to_data_url resize for Anthropic works")
    print(f"   - Original: 3000x2000, Resized: {resized_img.width}x{resized_img.height}")


def test_encode_image_to_data_url_compression(test_data_dir):
    """Test image compression to meet size limits."""
    from utils.llm_utils import encode_image_to_data_url
    from PIL import Image
    import numpy as np

    # Create a large noisy image that won't compress well
    img_path = Path(test_data_dir) / "noisy_image.png"
    # Create random noise image
    noise = np.random.randint(0, 255, (2000, 2000, 3), dtype=np.uint8)
    img = Image.fromarray(noise)
    img.save(img_path)

    # Set a small max size to force compression
    result = encode_image_to_data_url(
        str(img_path),
        resize_for_anthropic=True,
        max_size_bytes=500_000  # 500KB limit
    )

    # Verify the result is under the limit
    import base64
    b64_data = result.split(";base64,")[1]
    decoded_size = len(base64.b64decode(b64_data))

    assert decoded_size <= 500_000

    print("✅ encode_image_to_data_url compression works")
    print(f"   - Compressed to: {decoded_size / 1024:.1f}KB")


def test_ingest_images_for_llm_anthropic(test_data_dir, monkeypatch):
    """Test _ingest_images_for_llm passes resize flag for Anthropic."""
    from tools.developer import _ingest_images_for_llm
    from PIL import Image

    # Create a test image
    img_path = Path(test_data_dir) / "test_chart.png"
    img = Image.new('RGB', (500, 500), color='green')
    img.save(img_path)

    # Track if resize_for_anthropic is passed
    captured_calls = []

    def mock_encode(path, resize_for_anthropic=False, max_size_bytes=4_500_000):
        captured_calls.append({
            'path': path,
            'resize_for_anthropic': resize_for_anthropic,
            'max_size_bytes': max_size_bytes
        })
        # Return a valid data URL
        return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

    # Monkeypatch at the source module since it's imported inside the function
    monkeypatch.setattr("utils.llm_utils.encode_image_to_data_url", mock_encode)

    result = _ingest_images_for_llm([img_path], provider="anthropic")

    # Verify resize flag was passed for Anthropic
    assert len(captured_calls) == 1
    assert captured_calls[0]['resize_for_anthropic'] == True

    print("✅ _ingest_images_for_llm passes resize flag for Anthropic")


def test_ingest_images_for_llm_openai_no_resize(test_data_dir, monkeypatch):
    """Test _ingest_images_for_llm does not resize for OpenAI."""
    from tools.developer import _ingest_images_for_llm
    from PIL import Image

    # Create a test image
    img_path = Path(test_data_dir) / "test_chart.png"
    img = Image.new('RGB', (500, 500), color='yellow')
    img.save(img_path)

    # Track if resize_for_anthropic is passed
    captured_calls = []

    def mock_encode(path, resize_for_anthropic=False, max_size_bytes=4_500_000):
        captured_calls.append({
            'path': path,
            'resize_for_anthropic': resize_for_anthropic
        })
        return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

    # Monkeypatch at the source module since it's imported inside the function
    monkeypatch.setattr("utils.llm_utils.encode_image_to_data_url", mock_encode)

    result = _ingest_images_for_llm([img_path], provider="openai")

    # Verify resize flag was NOT passed for OpenAI
    assert len(captured_calls) == 1
    assert captured_calls[0]['resize_for_anthropic'] == False

    print("✅ _ingest_images_for_llm does not resize for OpenAI")


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
        # Should not be done immediately
        import time
        time.sleep(0.1)
        # Process might or might not be done yet, but result() should block and return
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
        if kwargs.get('text_format') == LogMonitorVerdict:
            return kill_verdict
        mock = MagicMock()
        mock.output = []
        mock.stop_reason = "end_turn"
        mock.content = []
        mock.action = "kill"
        mock.reason = "NaN loss detected at epoch 4"
        return kill_verdict

    monkeypatch.setattr("tools.developer.detect_provider", lambda x: "openai")
    monkeypatch.setattr("tools.developer.call_llm_with_retry", fake_call_llm)

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

    monkeypatch.setattr("tools.developer.detect_provider", lambda x: "openai")
    monkeypatch.setattr("tools.developer.call_llm_with_retry", fake_call_llm)

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
            mock_response = MagicMock(spec=[])  # no auto-attributes
            mock_item = MagicMock(spec=[])
            mock_item.type = "function_call"
            mock_item.name = "execute_bash"
            mock_item.arguments = '{"command": "nvidia-smi"}'
            mock_item.call_id = "call_123"
            mock_response.output = [mock_item]
            return mock_response

        # Second call: LLM returns verdict after seeing tool output
        return LogMonitorVerdict(
            action="kill",
            reason="GPU at 0% utilization — process is deadlocked"
        )

    monkeypatch.setattr("tools.developer.detect_provider", lambda x: "openai")
    monkeypatch.setattr("tools.developer.call_llm_with_retry", fake_call_llm)

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

    monkeypatch.setattr("tools.developer.detect_provider", lambda x: "openai")
    monkeypatch.setattr("tools.developer.call_llm_with_retry", fake_call_llm)

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
