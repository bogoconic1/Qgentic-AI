"""Unit tests for ModelRecommenderAgent."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import json
import tempfile
from unittest.mock import MagicMock
from pydantic import BaseModel

from agents.model_recommender import ModelRecommenderAgent


@pytest.fixture(autouse=True)
def patch_llm_calls(monkeypatch):
    """Mock all LLM API calls to avoid actual network requests."""

    def fake_call_llm(*args, **kwargs):
        """Return a mock response based on the text_format (Pydantic schema).

        When text_format is provided, the helper returns the parsed Pydantic object directly
        (not wrapped in a response object).
        """
        from schemas.model_recommender import (
            PreprocessingRecommendations,
            LossFunctionRecommendations,
            LossFunctionMustHave,
            LossFunctionNiceToHave,
            HyperparameterRecommendations,
            HyperparameterSection,
            HyperparameterItem,
            ArchitectureItem,
            InferenceStrategyRecommendations,
            InferenceStrategySection,
            InferenceStrategyItem,
            CategoryRecommendations,
            StrategyItem,
            ModelSelection,
        )

        text_format = kwargs.get('text_format')

        if text_format and issubclass(text_format, BaseModel):
            if text_format.__name__ == 'PreprocessingRecommendations':
                return text_format(
                    preprocessing=CategoryRecommendations(
                        MUST_HAVE=[StrategyItem(strategy="Tokenize using BERT", explanation="Required for transformer models")],
                        NICE_TO_HAVE=[StrategyItem(strategy="Lowercase text", explanation="Helps with generalization")]
                    )
                )
            elif text_format.__name__ == 'LossFunctionRecommendations':
                return text_format(
                    MUST_HAVE=LossFunctionMustHave(
                        loss_function="CrossEntropyLoss",
                        explanation="Suitable for multi-class classification"
                    ),
                    NICE_TO_HAVE=[
                        LossFunctionNiceToHave(
                            loss_function="FocalLoss",
                            explanation="Helps with class imbalance"
                        )
                    ]
                )
            elif text_format.__name__ == 'HyperparameterRecommendations':
                return text_format(
                    MUST_HAVE=HyperparameterSection(
                        hyperparameters=[
                            HyperparameterItem(hyperparameter="learning_rate=2e-5", explanation="Standard for BERT fine-tuning")
                        ],
                        architectures=[
                            ArchitectureItem(architecture="3 layer encoder", explanation="Balanced complexity")
                        ]
                    ),
                    NICE_TO_HAVE=HyperparameterSection(
                        hyperparameters=[
                            HyperparameterItem(hyperparameter="warmup_steps=1000", explanation="Helps with training stability")
                        ],
                        architectures=[]
                    )
                )
            elif text_format.__name__ == 'InferenceStrategyRecommendations':
                return text_format(
                    MUST_HAVE=InferenceStrategySection(
                        inference_strategies=[
                            InferenceStrategyItem(strategy="Test-time augmentation", explanation="Improves robustness")
                        ]
                    ),
                    NICE_TO_HAVE=InferenceStrategySection(
                        inference_strategies=[
                            InferenceStrategyItem(strategy="Ensemble voting", explanation="Better predictions")
                        ]
                    )
                )
            elif text_format.__name__ == 'ModelSelection':
                return text_format(
                    recommended_models=[],
                )
            else:
                return text_format()
        else:
            mock_response = MagicMock()
            mock_response.text = """```json
{
    "preprocessing": {
        "MUST_HAVE": [
            {"strategy": "Tokenize using BERT", "explanation": "Required for transformer models"}
        ],
        "NICE_TO_HAVE": [
            {"strategy": "Lowercase text", "explanation": "Helps with generalization"}
        ]
    }
}
```"""
            return mock_response

    monkeypatch.setattr("agents.model_recommender.call_llm", fake_call_llm)


@pytest.fixture(scope='module')
def test_task_dir():
    """Create a temporary task directory with dummy data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        task_root = Path(tmpdir)
        slug = "test-competition"
        iteration = 1

        # Create directory structure
        task_dir = task_root / slug
        outputs_dir = task_dir / "outputs" / str(iteration)
        outputs_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy description.md
        (task_dir / "description.md").write_text("# Test Competition\nThis is a test competition.")

        # Create dummy starter_suggestions.json
        starter_data = {
            "task_types": ["nlp"],
            "task_summary": "Text classification task"
        }
        (outputs_dir / "starter_suggestions.json").write_text(json.dumps(starter_data))

        # Create dummy plan.md
        (outputs_dir / "plan.md").write_text("# Research Plan\nTest plan content.")

        yield {
            'task_root': task_root,
            'slug': slug,
            'iteration': iteration
        }


def test_agent_initialization(test_task_dir, monkeypatch):
    """Test agent initialization with dummy data."""
    monkeypatch.setattr("agents.model_recommender._TASK_ROOT", test_task_dir['task_root'])

    agent = ModelRecommenderAgent(test_task_dir['slug'], test_task_dir['iteration'])

    assert agent.slug == test_task_dir['slug']
    assert agent.iteration == test_task_dir['iteration']
    assert agent.outputs_dir.exists()
    assert agent.json_path.name == "model_recommendations.json"

    assert "description" in agent.inputs
    assert "task_type" in agent.inputs
    assert "task_summary" in agent.inputs
    assert "plan" in agent.inputs

    assert isinstance(agent.inputs["task_type"], list)
    assert "nlp" in agent.inputs["task_type"]
    assert len(agent.inputs["description"]) > 0

    print(f"✅ Agent initialized successfully:")
    print(f"   - Task types: {agent.inputs['task_type']}")
    print(f"   - Description length: {len(agent.inputs['description'])} chars")
    print(f"   - Plan loaded: {agent.inputs.get('plan') is not None}")


def test_recommend_for_model(test_task_dir, monkeypatch):
    """Test generating recommendations for a single model."""
    monkeypatch.setattr("agents.model_recommender._TASK_ROOT", test_task_dir['task_root'])

    agent = ModelRecommenderAgent(test_task_dir['slug'], test_task_dir['iteration'])

    recommendations = agent._recommend_for_model("deberta-v3-large")

    assert isinstance(recommendations, dict)
    assert len(recommendations) > 0

    for key, value in recommendations.items():
        assert isinstance(value, dict), f"Value for {key} should be dict"

    expected_some_of = ["preprocessing", "loss_function", "hyperparameters", "inference_strategies", "data_augmentation"]
    found_keys = [k for k in expected_some_of if k in recommendations]
    assert len(found_keys) > 0, f"Expected at least one of {expected_some_of}, got {list(recommendations.keys())}"

    print("✅ Model recommendations generated successfully:")
    print(f"   - Keys found: {list(recommendations.keys())}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
