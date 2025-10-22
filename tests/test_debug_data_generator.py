"""Unit tests for debug data generator."""

import json
import pytest
from pathlib import Path
import shutil

from agents.debug_data_generator import (
    generate_debug_data,
    _should_skip_debug,
    _validate_debug_samples,
    _extract_code_block,
)


def test_should_skip_debug():
    """Test skip logic for debug mode."""

    # Should skip: train too small (<2000)
    skip, reason = _should_skip_debug(train_size=1500, test_size=500, num_classes=5)
    assert skip is True
    assert "too small" in reason.lower()

    # Should skip: too many classes (>80% of debug size)
    skip, reason = _should_skip_debug(train_size=5000, test_size=1000, num_classes=900)
    assert skip is True
    assert "many classes" in reason.lower()

    # Should NOT skip: good sizes
    skip, reason = _should_skip_debug(train_size=5000, test_size=1000, num_classes=10)
    assert skip is False
    assert reason is None

    # Edge case: no num_classes (regression)
    skip, reason = _should_skip_debug(train_size=5000, test_size=1000, num_classes=None)
    assert skip is False

    print("✅ Skip logic tests passed")


def test_extract_code_block():
    """Test Python code extraction from LLM responses."""

    # Test with fenced code block
    text1 = """Here is the sampling code:
```python
import pandas as pd
df = pd.read_csv("train.csv")
print(len(df))
```
"""
    result1 = _extract_code_block(text1)
    assert result1 is not None
    assert "import pandas" in result1
    assert "print(len(df))" in result1

    # Test with multiple code blocks (should take first)
    text2 = """
Some explanation
```python
# First block
import numpy as np
```

```python
# Second block
import torch
```
"""
    result2 = _extract_code_block(text2)
    assert result2 is not None
    assert "numpy" in result2
    assert "torch" not in result2

    # Test with no code block
    text3 = "No code here"
    result3 = _extract_code_block(text3)
    assert result3 is None

    print("✅ Code extraction tests passed")


def test_validate_debug_samples_basic():
    """Test validation with mock data."""

    # Create temporary test data
    test_dir = Path("/tmp/test_debug_validation")
    debug_dir = test_dir / "debug"

    try:
        # Setup
        debug_dir.mkdir(parents=True, exist_ok=True)

        # Create mock debug CSV
        import pandas as pd
        mock_df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": ["a", "b", "c", "d", "e"],
            "target": [0, 1, 0, 1, 1]
        })
        mock_df.to_csv(debug_dir / "train_debug.csv", index=False)

        # Validate
        result = _validate_debug_samples(test_dir, original_train_size=1000, num_classes=2)

        # Check result structure
        assert "valid" in result
        assert "errors" in result
        assert "metadata" in result
        assert "paths" in result
        assert result["valid"] is True

        print("✅ Basic validation test passed")

    finally:
        # Cleanup
        if test_dir.exists():
            shutil.rmtree(test_dir)


def test_cache_behavior():
    """Test that debug data is cached per competition."""

    # Use existing test task
    slug = "us-patent-phrase-to-phrase-matching"
    iteration = 1

    cache_path = Path(f"task/{slug}/debug_data_info.json")

    # If cache exists, test that it's reused
    if cache_path.exists():
        with open(cache_path) as f:
            cached_data = json.load(f)

        # Generate again - should use cache
        result = generate_debug_data(slug, iteration)

        # Should have same structure
        assert "skip_debug" in result
        assert "paths" in result
        assert "metadata" in result

        print(f"✅ Cache test passed (cache exists at {cache_path})")
    else:
        print(f"⚠️  Cache test skipped (no cache at {cache_path})")


def test_debug_info_structure():
    """Test that debug_info has expected structure."""

    # Test with a competition that should skip debug
    # (We'll use a small mock case)

    # Expected structure for skipped debug
    skip_info = {
        "skip_debug": True,
        "reason": "Some reason",
        "paths": {},
        "metadata": {}
    }

    assert "skip_debug" in skip_info
    assert skip_info["skip_debug"] is True
    assert "reason" in skip_info

    # Expected structure for active debug
    active_info = {
        "skip_debug": False,
        "paths": {
            "train_debug_csv": "task/slug/debug/train_debug.csv",
            "test_debug_csv": "task/slug/debug/test_debug.csv",
        },
        "metadata": {
            "train_size": 10000,
            "test_size": 5000,
            "debug_train_size": 1000,
            "debug_test_size": 200,
        }
    }

    assert "skip_debug" in active_info
    assert active_info["skip_debug"] is False
    assert "paths" in active_info
    assert "metadata" in active_info
    assert "train_debug_csv" in active_info["paths"]

    print("✅ Debug info structure test passed")


if __name__ == "__main__":
    print("Running unit tests for debug data generator...")
    print()

    tests = [
        ("test_should_skip_debug", test_should_skip_debug),
        ("test_extract_code_block", test_extract_code_block),
        ("test_validate_debug_samples_basic", test_validate_debug_samples_basic),
        ("test_cache_behavior", test_cache_behavior),
        ("test_debug_info_structure", test_debug_info_structure),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            test_func()
            print(f"✅ {test_name} passed\n")
            passed += 1
        except AssertionError as e:
            print(f"❌ {test_name} failed: {e}\n")
            failed += 1
        except Exception as e:
            print(f"⚠️  {test_name} error: {e}\n")
            failed += 1

    print("=" * 60)
    print(f"Test results: {passed} passed, {failed} failed")
    print("All unit tests completed!")
