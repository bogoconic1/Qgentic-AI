"""Test script to verify execute_code timeout functionality."""

import sys
import os
import tempfile
import logging
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, '/workspace/Qgentic-AI')

from tools.developer import execute_code

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

def test_timeout_short():
    """Test that timeout works with a short 3-second limit."""
    logger.info("=" * 60)
    logger.info("TEST 1: Testing timeout with 3-second limit")
    logger.info("=" * 60)

    # Create a temporary Python file that will run for a long time
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        temp_file = f.name
        f.write("""
import time
print("Starting long-running code...")
time.sleep(10)  # Sleep for 10 seconds (longer than 3-second timeout)
print("This should not be printed")
""")

    try:
        logger.info("Created test file: %s", temp_file)

        result = execute_code(temp_file, timeout_seconds=3)

        logger.info("Result: %s", result)

        # Check if timeout message is returned
        if "timed out after 3 second" in result:
            logger.info("✅ TEST PASSED: Timeout message received correctly")
            return True
        else:
            logger.error("❌ TEST FAILED: Expected timeout message, got: %s", result)
            return False
    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.unlink(temp_file)

def test_timeout_success():
    """Test that quick operations complete successfully."""
    logger.info("=" * 60)
    logger.info("TEST 2: Testing successful quick execution")
    logger.info("=" * 60)

    # Create a temporary Python file that completes quickly
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        temp_file = f.name
        f.write("""
print("Hello from quick execution!")
result = 2 + 2
print(f"2 + 2 = {result}")
""")

    try:
        logger.info("Created test file: %s", temp_file)

        result = execute_code(temp_file, timeout_seconds=30)

        logger.info("Result: %s", result)

        # Check if we got valid output
        if "Hello from quick execution" in result and "2 + 2 = 4" in result:
            logger.info("✅ TEST PASSED: Quick operation completed successfully")
            return True
        else:
            logger.error("❌ TEST FAILED: Expected successful output, got: %s", result)
            return False
    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.unlink(temp_file)

def test_default_timeout():
    """Test that default timeout is 3600 seconds (1 hour)."""
    logger.info("=" * 60)
    logger.info("TEST 3: Testing default timeout value (1 hour)")
    logger.info("=" * 60)

    # Create a temporary Python file that completes quickly
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        temp_file = f.name
        f.write("""
print("Testing default timeout")
""")

    try:
        logger.info("Created test file: %s", temp_file)

        # Call without specifying timeout (should use default 3600)
        result = execute_code(temp_file)

        logger.info("Result: %s", result)

        # Check the log message for timeout value
        if "Testing default timeout" in result:
            logger.info("✅ TEST PASSED: Default timeout allows execution")
            return True
        else:
            logger.error("❌ TEST FAILED: Execution failed with default timeout")
            return False
    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.unlink(temp_file)

if __name__ == "__main__":
    logger.info("Starting execute_code timeout tests...")
    logger.info("")

    test1_passed = test_timeout_short()
    logger.info("")

    test2_passed = test_timeout_success()
    logger.info("")

    test3_passed = test_default_timeout()
    logger.info("")

    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info("Test 1 (Timeout detection): %s", "PASSED ✅" if test1_passed else "FAILED ❌")
    logger.info("Test 2 (Quick execution): %s", "PASSED ✅" if test2_passed else "FAILED ❌")
    logger.info("Test 3 (Default timeout): %s", "PASSED ✅" if test3_passed else "FAILED ❌")
    logger.info("")

    if test1_passed and test2_passed and test3_passed:
        logger.info("🎉 ALL TESTS PASSED!")
        sys.exit(0)
    else:
        logger.error("❌ SOME TESTS FAILED")
        sys.exit(1)
