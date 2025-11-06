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
        mock_response = MagicMock()
        mock_response.output = []
        # Return structured output with all required fields
        mock_response.output_parsed = StackTraceSolution(
            checklist=["Check X", "Verify Y"],
            web_search_findings="Found solution on Stack Overflow",
            reasoning_and_solution="The error is caused by X. You can fix it by doing Y.",
            validation="Verify the fix works by running Z",
            further_steps="Consider refactoring the code"
        )
        mock_response.output_text = "Mock LLM response"
        return mock_response

    monkeypatch.setattr("tools.developer.call_llm_with_retry", fake_call_llm)

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
    def fake_call_llm(*args, **kwargs):
        mock_response = MagicMock()
        mock_response.output = []
        # Return None for output_parsed to trigger fallback
        mock_response.output_parsed = None
        mock_response.output_text = "Fallback solution text"
        return mock_response

    monkeypatch.setattr("tools.developer.call_llm_with_retry", fake_call_llm)

    query = "Traceback (most recent call last):\nTypeError: cannot convert"
    result = web_search_stack_trace(query)

    # Verify fallback to output_text
    assert "Fallback solution text" in result
    assert "This is how you can fix the error" in result

    print("✅ web_search_stack_trace fallback works:")
    print(f"   - Used fallback text successfully")


def test_search_red_flags(monkeypatch):
    """Test search_red_flags with mocked LLM call."""
    def fake_call_llm(*args, **kwargs):
        mock_response = MagicMock()
        mock_response.output = []
        mock_response.output_text = (
            "# Red Flags Analysis\n\n"
            "## Critical Issues\n"
            "1. Data leakage detected in preprocessing\n"
            "2. Overfitting on validation set\n"
        )
        return mock_response

    monkeypatch.setattr("tools.developer.call_llm_with_retry", fake_call_llm)

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
            suggestion_reason="Transformers have shown SOTA results on similar tasks"
        )
        mock_response.output_text = "Mock SOTA suggestions"
        return mock_response

    monkeypatch.setattr("tools.developer.call_llm_with_retry", fake_call_llm)

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
            suggestion_reason="Following the research plan systematically"
        )
        mock_response.output_text = "Mock response"
        return mock_response

    monkeypatch.setattr("tools.developer.call_llm_with_retry", fake_call_llm)

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
            suggestion_reason="Stacking typically provides better generalization"
        )
        mock_response.output_text = "Mock ensemble response"
        return mock_response

    monkeypatch.setattr("tools.developer.call_llm_with_retry", fake_call_llm)

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
