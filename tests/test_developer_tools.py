"""Unit tests for DeveloperAgent tools."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import tempfile
from unittest.mock import MagicMock

from tools.developer import (
    execute_code,
    web_search_stack_trace,
    search_red_flags,
    search_sota_suggestions
)


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
    result = execute_code(test_script_success, timeout_seconds=10)

    # Verify output contains success messages
    assert "Success!" in result
    assert "Script completed" in result

    print("✅ execute_code with successful script works:")
    print(f"   - Output length: {len(result)} chars")


def test_execute_code_timeout(test_script_timeout):
    """Test code execution timeout."""
    result = execute_code(test_script_timeout, timeout_seconds=3)

    # Verify timeout message
    assert "timed out" in result.lower()
    assert "3 second" in result.lower()

    print("✅ execute_code timeout detection works:")
    print(f"   - Timeout message: {result}")


def test_execute_code_error_with_web_search(test_script_error, monkeypatch):
    """Test code execution with error triggers web search."""
    # Mock web_search_stack_trace to avoid actual web search
    def fake_web_search(query):
        return query + "\nThis is how you can fix the error: \nMock solution for test error"

    monkeypatch.setattr("tools.developer.web_search_stack_trace", fake_web_search)

    result = execute_code(test_script_error, timeout_seconds=10)

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
        is_ensemble=False,
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
        is_ensemble=False,
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


def test_search_sota_suggestions_ensemble_mode(monkeypatch):
    """Test search_sota_suggestions with ensemble mode."""
    from schemas.developer import SOTAResponse

    captured_kwargs = {}

    def fake_call_llm(*args, **kwargs):
        captured_kwargs.update(kwargs)
        mock_response = MagicMock()
        mock_response.output = []
        mock_response.output_parsed = SOTAResponse(
            blacklist=False,
            blacklist_reason="Ensemble approach is promising",
            suggestion="Ensemble the top models using stacking",
            suggestion_reason="Stacking typically provides better generalization",
            suggestion_code="# Ensemble code\nfrom sklearn.ensemble import StackingClassifier"
        )
        mock_response.output_text = "Mock ensemble response"
        return mock_response

    # Mock all provider-specific LLM calls
    monkeypatch.setattr("tools.developer.call_llm_with_retry", fake_call_llm)
    monkeypatch.setattr("tools.developer.call_llm_with_retry_anthropic", fake_call_llm)
    monkeypatch.setattr("tools.developer.call_llm_with_retry_google", fake_call_llm)

    result = search_sota_suggestions(
        description="Test competition",
        context="Current context",
        red_flags="Red flags",
        executed_suggestion="Previous suggestion",
        failed_ideas=[],
        later_recommendations=None,
        shared_suggestions=None,
        is_ensemble=True,  # Ensemble mode
        external_data_listing=None,
        plan_content=None
    )

    # Verify ensemble mode is handled
    assert hasattr(result, 'output_parsed')
    assert "ensemble" in result.output_parsed.suggestion.lower()

    print("✅ search_sota_suggestions with ensemble mode works:")
    print(f"   - Ensemble-specific suggestion generated")


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
