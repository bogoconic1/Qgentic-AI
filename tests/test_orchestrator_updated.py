"""Test updated orchestrator flow."""

from agents.orchestrator import _format_recommendations_for_developer


def test_format_recommendations():
    """Test that ALL recommendations are formatted correctly without filtering."""
    recommendations = {
        "preprocessing": {
            "preprocessing": {
                "MUST_HAVE": [
                    {"strategy": "Strategy 1", "explanation": "Explanation 1"},
                    {"strategy": "Strategy 2", "explanation": "Explanation 2"},
                    {"strategy": "Strategy 3", "explanation": "Explanation 3"},
                    {"strategy": "Strategy 4", "explanation": "Explanation 4"},
                    {"strategy": "Strategy 5", "explanation": "Explanation 5"},
                ],
                "NICE_TO_HAVE": [],
            },
            "tokenization": {
                "MUST_HAVE": [
                    {
                        "strategy": "Use BPE tokenization",
                        "explanation": "Better for subword handling",
                    }
                ],
                "NICE_TO_HAVE": [],
            },
        },
        "loss_function": {
            "MUST_HAVE": {
                "loss_function": "MSELoss with custom QWK wrapper",
                "explanation": "Aligns with QWK competition metric",
            },
            "NICE_TO_HAVE": [],
        },
        "hyperparameters": {
            "MUST_HAVE": {
                "hyperparameters": [
                    {
                        "hyperparameter": "learning_rate: 2e-5",
                        "explanation": "Standard for transformers",
                    },
                    {
                        "hyperparameter": "batch_size: 16",
                        "explanation": "Balance memory and training speed",
                    },
                    {
                        "hyperparameter": "epochs: 5",
                        "explanation": "Prevent overfitting",
                    },
                    {
                        "hyperparameter": "warmup_steps: 100",
                        "explanation": "Stabilize training",
                    },
                    {
                        "hyperparameter": "weight_decay: 0.01",
                        "explanation": "Regularization",
                    },
                    {
                        "hyperparameter": "max_grad_norm: 1.0",
                        "explanation": "Gradient clipping",
                    },
                ],
                "architectures": [
                    {
                        "architecture": "Add dropout layer (0.1)",
                        "explanation": "Reduce overfitting",
                    },
                    {
                        "architecture": "Multi-sample dropout",
                        "explanation": "Better regularization",
                    },
                    {
                        "architecture": "Layerwise learning rate decay",
                        "explanation": "Fine-tune pretrained layers",
                    },
                    {
                        "architecture": "EMA weights",
                        "explanation": "Stabilize predictions",
                    },
                ],
            },
            "NICE_TO_HAVE": {"hyperparameters": [], "architectures": []},
        },
        "inference_strategies": {
            "MUST_HAVE": {
                "inference_strategies": [
                    {
                        "strategy": "Test-time augmentation",
                        "explanation": "Improve robustness",
                    },
                    {
                        "strategy": "Threshold tuning",
                        "explanation": "Optimize for metric",
                    },
                    {
                        "strategy": "Fold averaging",
                        "explanation": "Reduce variance",
                    },
                    {
                        "strategy": "Monte Carlo dropout",
                        "explanation": "Uncertainty estimation",
                    },
                ]
            },
            "NICE_TO_HAVE": {"inference_strategies": []},
        },
    }

    details = _format_recommendations_for_developer(recommendations)

    assert "Strategy 1" in details
    assert "Strategy 2" in details
    assert "Strategy 3" in details
    assert "Strategy 4" in details
    assert "Strategy 5" in details

    assert "learning_rate: 2e-5" in details
    assert "batch_size: 16" in details
    assert "epochs: 5" in details
    assert "warmup_steps: 100" in details
    assert "weight_decay: 0.01" in details
    assert "max_grad_norm: 1.0" in details

    assert "Add dropout layer (0.1)" in details
    assert "Multi-sample dropout" in details
    assert "Layerwise learning rate decay" in details
    assert "EMA weights" in details

    assert "Test-time augmentation" in details
    assert "Threshold tuning" in details
    assert "Fold averaging" in details
    assert "Monte Carlo dropout" in details


def test_empty_preprocessing_and_inference():
    """Test handling of empty preprocessing and inference lists."""
    recs = {
        "preprocessing": {},
        "loss_function": {
            "MUST_HAVE": {
                "loss_function": "CrossEntropyLoss",
                "explanation": "Good for classification",
            },
            "NICE_TO_HAVE": [],
        },
        "hyperparameters": {
            "MUST_HAVE": {"hyperparameters": [], "architectures": []},
            "NICE_TO_HAVE": {"hyperparameters": [], "architectures": []},
        },
        "inference_strategies": {
            "MUST_HAVE": {"inference_strategies": []},
            "NICE_TO_HAVE": {"inference_strategies": []},
        },
    }
    details = _format_recommendations_for_developer(recs)
    assert "## Loss Function" in details
    assert "CrossEntropyLoss" in details
    assert "## Preprocessing Strategies" not in details
    assert "## Inference Strategies" not in details
