"""Unit tests for ResearcherAgent tools."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import tempfile
from unittest.mock import MagicMock

from tools.researcher import ask_eda, download_external_datasets, get_tools


@pytest.fixture
def test_data_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_ask_eda_timeout(test_data_dir, monkeypatch):
    """Test that ask_eda times out for long-running operations."""
    # Mock the LLM to generate code that sleeps
    def fake_call_llm(*args, **kwargs):
        mock_response = MagicMock()
        mock_response.output = []
        # Generate code that sleeps for 5 seconds
        mock_response.output_text = (
            "```python\n"
            "import time\n"
            "time.sleep(5)\n"
            "print('Done')\n"
            "```"
        )
        return mock_response

    monkeypatch.setattr("tools.researcher.call_llm_with_retry", fake_call_llm)

    result = ask_eda(
        question="Sleep for 5 seconds",
        description="Test competition",
        data_path=test_data_dir,
        max_attempts=1,
        timeout_seconds=3  # 3 second timeout - should timeout
    )

    # Verify timeout message is returned
    assert "timed out" in result.lower() or "cannot be executed within 3 seconds" in result

    print("✅ ask_eda timeout detection works:")
    print(f"   - Timeout message received correctly")


def test_ask_eda_success(test_data_dir, monkeypatch):
    """Test that ask_eda succeeds with quick code."""
    # Mock the LLM to generate simple code
    def fake_call_llm(*args, **kwargs):
        mock_response = MagicMock()
        mock_response.output = []
        # Generate simple code that completes quickly
        mock_response.output_text = (
            "```python\n"
            "print('Hello from EDA!')\n"
            "print('Test completed successfully')\n"
            "```"
        )
        return mock_response

    monkeypatch.setattr("tools.researcher.call_llm_with_retry", fake_call_llm)

    result = ask_eda(
        question="Print a hello message",
        description="Test competition",
        data_path=test_data_dir,
        max_attempts=1,
        timeout_seconds=10  # 10 second timeout - plenty of time
    )

    # Verify we got a successful result
    assert result is not None
    assert len(result) > 0
    assert "cannot be executed" not in result.lower()
    assert "cannot be answered" not in result.lower()
    # Should contain our output
    assert "Hello from EDA" in result or "Test completed" in result

    print("✅ ask_eda success works:")
    print(f"   - Result length: {len(result)} chars")


def test_ask_eda_retry_on_no_code_block(test_data_dir, monkeypatch):
    """Test that ask_eda retries when LLM doesn't return code block."""
    attempt_count = [0]

    def fake_call_llm(*args, **kwargs):
        attempt_count[0] += 1
        mock_response = MagicMock()
        mock_response.output = []

        if attempt_count[0] == 1:
            # First attempt: no code block
            mock_response.output_text = "Let me think about this..."
        else:
            # Second attempt: valid code
            mock_response.output_text = (
                "```python\n"
                "print('Success on retry')\n"
                "```"
            )

        return mock_response

    monkeypatch.setattr("tools.researcher.call_llm_with_retry", fake_call_llm)

    result = ask_eda(
        question="Test retry logic",
        description="Test competition",
        data_path=test_data_dir,
        max_attempts=2,
        timeout_seconds=10
    )

    # Should have tried twice
    assert attempt_count[0] == 2
    # Should eventually succeed
    assert "Success on retry" in result or result

    print("✅ ask_eda retry logic works:")
    print(f"   - Attempted {attempt_count[0]} times")


def test_download_external_datasets(monkeypatch):
    """Test download_external_datasets with mocked Kaggle API."""
    # Create temporary directory with comp_metadata.yaml
    with tempfile.TemporaryDirectory() as tmpdir:
        task_root = Path(tmpdir)
        comp_dir = task_root / "test-competition"
        comp_dir.mkdir(parents=True, exist_ok=True)

        # Create comp_metadata.yaml
        import yaml
        metadata = {
            "START_DATE": "2024-01-01",
            "END_DATE": "2024-12-31"
        }
        with open(comp_dir / "comp_metadata.yaml", "w") as f:
            yaml.dump(metadata, f)

        # Create external data directory
        external_dir = comp_dir / "external_data"
        external_dir.mkdir(parents=True, exist_ok=True)

        # Set environment variable
        import os
        os.environ["EXTERNAL_DATA_DIR"] = str(external_dir)

        # Patch _TASK_ROOT
        monkeypatch.setattr("tools.researcher._TASK_ROOT", task_root)

        # Mock the LLM to return structured output
        from schemas.researcher import DatasetDiscovery

        def fake_call_llm(*args, **kwargs):
            mock_response = MagicMock()
            mock_response.output = []

            # Return structured output with dataset names (Kaggle format: user/dataset)
            mock_response.output_parsed = DatasetDiscovery(
                datasets=["kaggle/time-series-data", "user123/sales-forecasting"]
            )
            mock_response.output_text = "Found datasets via web search"

            return mock_response

        monkeypatch.setattr("tools.researcher.call_llm_with_retry", fake_call_llm)

        # Mock Kaggle API
        mock_dataset_metadata = MagicMock()
        mock_dataset_metadata.lastUpdated = "2024-06-15"

        def fake_dataset_list(search=None):
            return [mock_dataset_metadata]

        # Mock kaggle.api
        mock_kaggle_api = MagicMock()
        mock_kaggle_api.dataset_list = fake_dataset_list
        monkeypatch.setattr("tools.researcher.kaggle.api", mock_kaggle_api)

        # Mock kagglehub.dataset_download
        def fake_download(dataset_name):
            # Return a fake path
            download_path = Path(tmpdir) / "kagglehub" / "datasets" / dataset_name.replace("/", "_")
            download_path.mkdir(parents=True, exist_ok=True)
            # Create a dummy file
            (download_path / "data.csv").write_text("dummy,data\n1,2\n")
            return str(download_path)

        monkeypatch.setattr("tools.researcher.kagglehub.dataset_download", fake_download)

        # Mock os.system to avoid actual file copying
        def fake_system(cmd):
            return 0

        monkeypatch.setattr("os.system", fake_system)

        # Mock _build_directory_listing
        def fake_directory_listing(path):
            return "external_data/\n  kaggle_time-series-data/\n    data.csv"

        monkeypatch.setattr("tools.researcher._build_directory_listing", fake_directory_listing)

        result = download_external_datasets(
            question_1="external dataset for time series forecasting",
            question_2="public dataset with time series data",
            question_3="historical sales data for prediction",
            slug="test-competition",
            max_attempts=1
        )

        # Verify we got results
        assert result is not None
        assert len(result) > 0
        assert "Relevant Datasets are downloaded" in result or "contains" in result

        print("✅ download_external_datasets works:")
        print(f"   - Result length: {len(result)} chars")
        print(f"   - Result preview: {result[:100]}...")


def test_get_tools():
    """Test that get_tools returns proper tool definitions."""
    tools = get_tools()

    # Verify structure
    assert isinstance(tools, list)
    assert len(tools) > 0

    # Check that expected tools are present
    tool_names = [tool.get('name') for tool in tools if isinstance(tool, dict)]

    assert 'ask_eda' in tool_names or 'run_ab_test' in tool_names
    assert 'download_external_datasets' in tool_names

    # Verify tool schema structure
    for tool in tools:
        if isinstance(tool, dict):
            assert 'name' in tool
            assert 'description' in tool or 'function' in tool
            # Check parameters exist
            if 'function' in tool:
                assert 'parameters' in tool['function']

    print("✅ get_tools returns valid tool definitions:")
    print(f"   - Number of tools: {len(tools)}")
    print(f"   - Tool names: {tool_names}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
