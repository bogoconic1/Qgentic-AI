"""Unit tests for ModelRecommenderAgent."""

import json
import pytest
from pathlib import Path

from agents.model_recommender import ModelRecommenderAgent


def test_extract_json_block():
    """Test JSON extraction from LLM responses."""

    # Test with fenced JSON block
    text1 = """Here is the output:
```json
{
    "loss_function": "CrossEntropyLoss",
    "explanation": "Good for classification"
}
```
"""
    result1 = ModelRecommenderAgent._extract_json_block(text1)
    assert result1 is not None
    parsed1 = json.loads(result1)
    assert parsed1["loss_function"] == "CrossEntropyLoss"

    # Test with just braces
    text2 = """Some text before
{"preprocessing": [{"strategy": "normalize", "explanation": "scale features"}]}
Some text after"""
    result2 = ModelRecommenderAgent._extract_json_block(text2)
    assert result2 is not None
    parsed2 = json.loads(result2)
    assert "preprocessing" in parsed2

    # Test with no JSON
    text3 = "No JSON here"
    result3 = ModelRecommenderAgent._extract_json_block(text3)
    assert result3 is None


def test_load_inputs():
    """Test input loading from existing task."""
    # Use existing task directory
    slug = "us-patent-phrase-to-phrase-matching"
    iteration = 1

    agent = ModelRecommenderAgent(slug, iteration)

    # Check that inputs were loaded
    assert "description" in agent.inputs
    assert "task_type" in agent.inputs
    assert "task_summary" in agent.inputs
    assert "plan" in agent.inputs

    # Check task_type is loaded correctly
    assert agent.inputs["task_type"] == "Natural Language Processing"

    # Check description is not empty
    assert len(agent.inputs["description"]) > 0

    print(f"✅ Loaded inputs successfully:")
    print(f"   - Task type: {agent.inputs['task_type']}")
    print(f"   - Description length: {len(agent.inputs['description'])} chars")
    print(f"   - Plan loaded: {agent.inputs.get('plan') is not None}")


def test_agent_initialization():
    """Test agent initialization."""
    slug = "us-patent-phrase-to-phrase-matching"
    iteration = 1

    agent = ModelRecommenderAgent(slug, iteration)

    # Check paths are set correctly
    assert agent.slug == slug
    assert agent.iteration == iteration
    assert agent.outputs_dir.exists()
    assert agent.json_path.name == "model_recommendations.json"

    print(f"✅ Agent initialized successfully:")
    print(f"   - Output dir: {agent.outputs_dir}")
    print(f"   - JSON path: {agent.json_path}")


if __name__ == "__main__":
    print("Running unit tests for ModelRecommenderAgent...")
    print()

    try:
        test_extract_json_block()
        print("✅ test_extract_json_block passed\n")
    except AssertionError as e:
        print(f"❌ test_extract_json_block failed: {e}\n")

    try:
        test_load_inputs()
        print("✅ test_load_inputs passed\n")
    except AssertionError as e:
        print(f"❌ test_load_inputs failed: {e}\n")

    try:
        test_agent_initialization()
        print("✅ test_agent_initialization passed\n")
    except AssertionError as e:
        print(f"❌ test_agent_initialization failed: {e}\n")

    print("=" * 60)
    print("All unit tests completed!")
