"""Test updated orchestrator flow."""

import json
from pathlib import Path
from agents.orchestrator import _build_example_details_from_recommendations


def test_build_example_details():
    """Test that recommendations are formatted correctly for developer."""

    # Sample recommendations
    recommendations = {
        "preprocessing": {
            "preprocessing": [
                {"strategy": "Normalize text", "explanation": "Lowercase and remove special chars"},
                {"strategy": "Remove stopwords", "explanation": "Filter common words"}
            ],
            "tokenization": [
                {"strategy": "Use BPE tokenization", "explanation": "Better for subword handling"}
            ]
        },
        "loss_function": {
            "loss_function": "MSELoss with custom QWK wrapper",
            "explanation": "Aligns with QWK competition metric"
        },
        "hyperparameters": {
            "hyperparameters": [
                {"hyperparameter": "learning_rate: 2e-5", "explanation": "Standard for transformers"},
                {"hyperparameter": "batch_size: 16", "explanation": "Balance memory and training speed"}
            ],
            "architectures": [
                {"architecture": "Add dropout layer (0.1)", "explanation": "Reduce overfitting"}
            ]
        },
        "inference_strategies": {
            "inference_strategies": [
                {"strategy": "Test-time augmentation", "explanation": "Improve robustness"},
                {"strategy": "Threshold tuning", "explanation": "Optimize for metric"}
            ]
        }
    }

    # Build example details
    details = _build_example_details_from_recommendations(recommendations)

    # Verify structure
    assert "## Preprocessing Strategies" in details
    assert "Normalize text" in details
    assert "## Loss Function" in details
    assert "MSELoss with custom QWK wrapper" in details
    assert "## Hyperparameters" in details
    assert "learning_rate: 2e-5" in details
    assert "## Inference Strategies" in details
    assert "Test-time augmentation" in details

    print("✅ Example details formatted correctly:")
    print()
    print(details)
    print()


def test_empty_recommendations():
    """Test handling of empty recommendations."""
    empty_recs = {}
    details = _build_example_details_from_recommendations(empty_recs)
    assert details == "No specific recommendations available."
    print("✅ Empty recommendations handled correctly")


def test_partial_recommendations():
    """Test handling of partial recommendations."""
    partial_recs = {
        "loss_function": {
            "loss_function": "CrossEntropyLoss",
            "explanation": "Good for classification"
        }
    }
    details = _build_example_details_from_recommendations(partial_recs)
    assert "## Loss Function" in details
    assert "CrossEntropyLoss" in details
    assert "## Preprocessing Strategies" not in details
    print("✅ Partial recommendations handled correctly")


if __name__ == "__main__":
    print("Testing updated orchestrator functionality...")
    print("=" * 60)
    print()

    try:
        test_build_example_details()
        print()
    except AssertionError as e:
        print(f"❌ test_build_example_details failed: {e}\n")

    try:
        test_empty_recommendations()
        print()
    except AssertionError as e:
        print(f"❌ test_empty_recommendations failed: {e}\n")

    try:
        test_partial_recommendations()
        print()
    except AssertionError as e:
        print(f"❌ test_partial_recommendations failed: {e}\n")

    print("=" * 60)
    print("All orchestrator tests completed!")
