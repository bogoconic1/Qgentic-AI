"""Test updated orchestrator flow with NOW/LATER structure."""

import json
from pathlib import Path
from agents.orchestrator import _format_recommendations_for_developer


def test_format_recommendations_now_later():
    """Test that NOW recommendations are formatted correctly from unified strategy."""

    # Sample recommendations with NOW/LATER structure
    recommendations = {
        "preprocessing": {
            "NOW": [
                {"strategy": "NFKC normalization", "explanation": "Handle unicode properly"},
                {"strategy": "Whitespace cleanup", "explanation": "Remove extra spaces"},
                {"strategy": "Basic deduplication", "explanation": "Remove exact duplicates"},
            ],
            "LATER": [
                {"strategy": "Heavy augmentation", "explanation": "Try after baseline works", "expected_improvement": "1-2%"},
                {"strategy": "Mixup/Cutmix", "explanation": "Advanced augmentation"},
            ]
        },
        "loss_function": {
            "NOW": {
                "loss": "MSELoss",
                "explanation": "Standard loss for regression baseline"
            },
            "LATER": [
                {"loss": "Pearson correlation with z-score", "explanation": "Try after MSE baseline", "when_to_try": "Iteration 2"}
            ]
        },
        "hyperparameters": {
            "NOW": {
                "core_hyperparameters": {
                    "learning_rate": "2e-5",
                    "batch_size": "32",
                    "epochs": "3",
                    "optimizer": "AdamW",
                    "scheduler": "cosine with warmup",
                    "weight_decay": "0.01",
                    "dropout": "0.1"
                },
                "architecture": "Standard [CLS] pooling → Linear(768, 1)"
            },
            "LATER": {
                "training_enhancements": [
                    {"technique": "EMA (decay 0.999)", "explanation": "Smooths training noise"},
                    {"technique": "Layerwise LR decay 0.9", "explanation": "Preserve pretrained weights"},
                ],
                "architecture_enhancements": [
                    {"technique": "Layer-weighted pooling", "explanation": "Blend last 4 layers"},
                    {"technique": "Multi-sample dropout", "explanation": "Ensemble regularization"},
                ]
            }
        },
        "inference": {
            "NOW": [
                {"strategy": "Direct prediction", "explanation": "Forward pass only"},
                {"strategy": "Clip to [0, 1]", "explanation": "Ensure valid range"},
            ],
            "LATER": [
                {"strategy": "5-fold averaging", "explanation": "Requires K-Fold training", "depends_on": "K-Fold"},
                {"strategy": "TTA (anchor-target swap)", "explanation": "Symmetric similarity"},
            ]
        }
    }

    # Format recommendations for developer (should extract only NOW items)
    details = _format_recommendations_for_developer(recommendations)

    # Verify NOW items are included
    assert "NFKC normalization" in details
    assert "Whitespace cleanup" in details
    assert "Basic deduplication" in details
    assert "MSELoss" in details
    assert "learning_rate: 2e-5" in details
    assert "batch_size: 32" in details
    assert "Standard [CLS] pooling" in details
    assert "Direct prediction" in details
    assert "Clip to [0, 1]" in details

    # Verify LATER items are NOT included
    assert "Heavy augmentation" not in details
    assert "Mixup/Cutmix" not in details
    assert "Pearson correlation" not in details
    assert "EMA" not in details
    assert "Layer-weighted pooling" not in details
    assert "5-fold averaging" not in details
    assert "TTA" not in details

    print("✅ NOW recommendations formatted correctly:")
    print(f"   - 3 preprocessing strategies (NOW only)")
    print(f"   - 1 loss function (NOW only)")
    print(f"   - 7 hyperparameters + architecture (NOW only)")
    print(f"   - 2 inference strategies (NOW only)")
    print()
    print(details)
    print()


def test_empty_recommendations():
    """Test handling of empty recommendations."""
    empty_recs = {}
    details = _format_recommendations_for_developer(empty_recs)
    assert details == "No specific recommendations available."
    print("✅ Empty recommendations handled correctly")


def test_partial_recommendations():
    """Test handling of partial recommendations with NOW/LATER."""
    partial_recs = {
        "loss_function": {
            "NOW": {
                "loss": "CrossEntropyLoss",
                "explanation": "Good for classification"
            }
        }
    }
    details = _format_recommendations_for_developer(partial_recs)
    assert "## Loss Function" in details
    assert "CrossEntropyLoss" in details
    assert "## Preprocessing Strategies" not in details
    print("✅ Partial recommendations handled correctly")


def test_only_later_recommendations():
    """Test handling when only LATER recommendations exist (should return empty)."""
    only_later = {
        "preprocessing": {
            "LATER": [
                {"strategy": "Advanced technique", "explanation": "For later"}
            ]
        }
    }
    details = _format_recommendations_for_developer(only_later)
    assert details == "No specific recommendations available."
    print("✅ LATER-only recommendations correctly excluded")


if __name__ == "__main__":
    print("Testing updated orchestrator functionality with NOW/LATER structure...")
    print("=" * 60)
    print()

    try:
        test_format_recommendations_now_later()
        print()
    except AssertionError as e:
        print(f"❌ test_format_recommendations_now_later failed: {e}\n")

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

    try:
        test_only_later_recommendations()
        print()
    except AssertionError as e:
        print(f"❌ test_only_later_recommendations failed: {e}\n")

    print("=" * 60)
    print("All orchestrator tests completed!")
