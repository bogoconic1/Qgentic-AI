"""Test script to verify ask_eda timeout functionality."""

import sys
import os
import tempfile
import logging
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, '/workspace/Qgentic-AI')

from tools.researcher import ask_eda

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

def test_timeout_short():
    """Test that timeout works with a short 5-second limit."""
    logger.info("=" * 60)
    logger.info("TEST 1: Testing timeout with 5-second limit")
    logger.info("=" * 60)

    # Create a temporary directory for test
    with tempfile.TemporaryDirectory() as tmpdir:
        test_data_path = Path(tmpdir)

        # Create a dummy CSV file
        import pandas as pd
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': ['a', 'b', 'c', 'd', 'e']
        })
        csv_path = test_data_path / "test_data.csv"
        df.to_csv(csv_path, index=False)

        logger.info("Created test data at %s", csv_path)

        # Ask a question that will cause a long-running operation (infinite loop)
        question = "Load test_data.csv and then run an infinite loop: while True: pass"
        description = "Test competition for timeout verification"

        result = ask_eda(
            question=question,
            description=description,
            data_path=str(test_data_path),
            max_attempts=1,  # Only try once
            timeout_seconds=5  # 5 second timeout
        )

        logger.info("Result: %s", result)

        # Check if timeout message is returned
        if "cannot be executed within 5 seconds" in result:
            logger.info("✅ TEST PASSED: Timeout message received correctly")
            return True
        else:
            logger.error("❌ TEST FAILED: Expected timeout message, got: %s", result)
            return False

def test_timeout_success():
    """Test that quick operations complete successfully."""
    logger.info("=" * 60)
    logger.info("TEST 2: Testing successful quick execution")
    logger.info("=" * 60)

    # Create a temporary directory for test
    with tempfile.TemporaryDirectory() as tmpdir:
        test_data_path = Path(tmpdir)

        # Create a dummy CSV file
        import pandas as pd
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': ['a', 'b', 'c', 'd', 'e']
        })
        csv_path = test_data_path / "test_data.csv"
        df.to_csv(csv_path, index=False)

        logger.info("Created test data at %s", csv_path)

        # Ask a simple question that should complete quickly
        question = "Load test_data.csv and print the shape"
        description = "Test competition for success verification"

        result = ask_eda(
            question=question,
            description=description,
            data_path=str(test_data_path),
            max_attempts=1,
            timeout_seconds=30  # 30 second timeout (plenty of time)
        )

        logger.info("Result: %s", result)

        # Check if we got a valid result (not a timeout or error message)
        if "cannot be executed" not in result and "cannot be answered" not in result:
            logger.info("✅ TEST PASSED: Quick operation completed successfully")
            return True
        else:
            logger.error("❌ TEST FAILED: Expected successful result, got: %s", result)
            return False

if __name__ == "__main__":
    logger.info("Starting timeout tests...")
    logger.info("")

    test1_passed = test_timeout_short()
    logger.info("")

    test2_passed = test_timeout_success()
    logger.info("")

    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info("Test 1 (Timeout detection): %s", "PASSED ✅" if test1_passed else "FAILED ❌")
    logger.info("Test 2 (Quick execution): %s", "PASSED ✅" if test2_passed else "FAILED ❌")
    logger.info("")

    if test1_passed and test2_passed:
        logger.info("🎉 ALL TESTS PASSED!")
        sys.exit(0)
    else:
        logger.error("❌ SOME TESTS FAILED")
        sys.exit(1)
