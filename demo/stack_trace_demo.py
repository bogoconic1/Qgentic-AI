"""Demo of fine-tuned Gemini code API model with fallback to web search."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.developer import web_search_stack_trace

# Test queries
QUERY_1_XGBOOST = """Traceback (most recent call last):
  File "/lambda/nfs/workspace/Qgentic-AI/task/playground-series-s5e11/outputs/10_1/code_10_1_v1.py", line 460, in <module>
    run_pipeline(debug_mode=True)
  File "/lambda/nfs/workspace/Qgentic-AI/task/playground-series-s5e11/outputs/10_1/code_10_1_v1.py", line 394, in run_pipeline
    model_baseline = fit_xgb_classifier(X_tr, y_tr, X_va, y_va, base_params, early_stopping_rounds=100)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/lambda/nfs/workspace/Qgentic-AI/task/playground-series-s5e11/outputs/10_1/code_10_1_v1.py", line 253, in fit_xgb_classifier
    model.fit(
  File "/home/ubuntu/miniconda3/envs/qgentic-model-1/lib/python3.12/site-packages/xgboost/core.py", line 774, in inner_f
    return func(**kwargs)
           ^^^^^^^^^^^^^^
TypeError: XGBClassifier.fit() got an unexpected keyword argument 'early_stopping_rounds'"""

QUERY_2_CATBOOST = """Traceback (most recent call last):
  File "/lambda/nfs/workspace/Qgentic-AI/task/playground-series-s5e11/outputs/10_3/code_10_3_v2.py", line 452, in <module>
    debug_auc, debug_meta = run_pipeline(debug=True)
                            ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/lambda/nfs/workspace/Qgentic-AI/task/playground-series-s5e11/outputs/10_3/code_10_3_v2.py", line 390, in run_pipeline
    model_base, auc_base, best_iter_base, info_base = train_one_fold(
                                                      ^^^^^^^^^^^^^^^
  File "/lambda/nfs/workspace/Qgentic-AI/task/playground-series-s5e11/outputs/10_3/code_10_3_v2.py", line 255, in train_one_fold
    model.fit(train_pool, eval_set=valid_pool, verbose=fit_verbose)
  File "/home/ubuntu/miniconda3/envs/qgentic-model-3/lib/python3.12/site-packages/catboost/core.py", line 5245, in fit
    self._fit(X, y, cat_features, text_features, embedding_features, None, graph, sample_weight, None, None, None, None, baseline, use_best_model,
  File "/home/ubuntu/miniconda3/envs/qgentic-model-3/lib/python3.12/site-packages/catboost/core.py", line 2395, in _fit
    train_params = self._prepare_train_params(
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/miniconda3/envs/qgentic-model-3/lib/python3.12/site-packages/catboost/core.py", line 2321, in _prepare_train_params
    _check_train_params(params)
  File "_catboost.pyx", line 6601, in _catboost._check_train_params
  File "_catboost.pyx", line 6623, in _catboost._check_train_params
_catboost.CatBoostError: catboost/private/libs/options/catboost_options.cpp:637: Error: rsm on GPU is supported for pairwise modes only"""


def test_query(query_name: str, query: str):
    """Test a single query with web_search_stack_trace."""
    print("\n" + "=" * 80)
    print(f"Testing Query: {query_name}")
    print("=" * 80)
    print(f"\nInput:\n{query}")
    print("\n" + "-" * 80)
    print("Processing...")
    print("-" * 80 + "\n")

    result = web_search_stack_trace(query)

    print("Result:")
    print("=" * 80)
    print(result)
    print("=" * 80)


def main():
    """Demo the fine-tuned code API model with fallback to web search."""
    print("=" * 80)
    print("Fine-tuned Gemini Code API Model Demo")
    print("Testing with XGBoost (should use fine-tuned model) and")
    print("CatBoost (should fall back to web search)")
    print("=" * 80)

    # Test 1: XGBoost error (should be handled by fine-tuned model)
    test_query("Query 1: XGBoost Error", QUERY_1_XGBOOST)

    # Test 2: CatBoost error (should fall back to web search)
    test_query("Query 2: CatBoost Error", QUERY_2_CATBOOST)

    print("\n" + "=" * 80)
    print("Demo Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
