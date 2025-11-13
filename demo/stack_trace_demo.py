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

QUERY_3_PANDAS = """Traceback (most recent call last):
  File "/lambda/nfs/workspace/Qgentic-AI/task/learning-agency-lab-automated-essay-scoring-2/eda_temp_7.py", line 183, in <module>
    XB_va = z_norm(dva["log_wc"], dva["len_band"])
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/lambda/nfs/workspace/Qgentic-AI/task/learning-agency-lab-automated-essay-scoring-2/eda_temp_7.py", line 171, in z_norm
    mu = band_stats.loc[band_idx, "mu"].to_numpy()
         ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^
  File "/home/ubuntu/miniconda3/envs/qgentic-ai/lib/python3.12/site-packages/pandas/core/indexing.py", line 1185, in __getitem__
    return self._getitem_tuple(key)
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/miniconda3/envs/qgentic-ai/lib/python3.12/site-packages/pandas/core/indexing.py", line 1369, in _getitem_tuple
    return self._getitem_lowerdim(tup)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/miniconda3/envs/qgentic-ai/lib/python3.12/site-packages/pandas/core/indexing.py", line 1090, in _getitem_lowerdim
    return getattr(section, self.name)[new_key]
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^
  File "/home/ubuntu/miniconda3/envs/qgentic-ai/lib/python3.12/site-packages/pandas/core/indexing.py", line 1192, in __getitem__
    return self._getitem_axis(maybe_callable, axis=axis)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/miniconda3/envs/qgentic-ai/lib/python3.12/site-packages/pandas/core/indexing.py", line 1421, in _getitem_axis
    return self._getitem_iterable(key, axis=axis)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/miniconda3/envs/qgentic-ai/lib/python3.12/site-packages/pandas/core/indexing.py", line 1361, in _getitem_iterable
    keyarr, indexer = self._get_listlike_indexer(key, axis)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/miniconda3/envs/qgentic-ai/lib/python3.12/site-packages/pandas/core/indexing.py", line 1559, in _get_listlike_indexer
    keyarr, indexer = ax._get_indexer_strict(key, axis_name)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ubuntu/miniconda3/envs/qgentic-ai/lib/python3.12/site-packages/pandas/core/indexes/base.py", line 6212, in _get_indexer_strict
    self._raise_if_missing(keyarr, indexer, axis_name)
  File "/home/ubuntu/miniconda3/envs/qgentic-ai/lib/python3.12/site-packages/pandas/core/indexes/base.py", line 6264, in _raise_if_missing
    raise KeyError(f"{not_found} not in index")
KeyError: '[nan] not in index'"""


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
    print("Testing with XGBoost, CatBoost, and Pandas errors")
    print("=" * 80)

    # Test 1: XGBoost error (should be handled by fine-tuned model)
    test_query("Query 1: XGBoost Error", QUERY_1_XGBOOST)

    # Test 2: CatBoost error (should be handled by fine-tuned model)
    test_query("Query 2: CatBoost Error", QUERY_2_CATBOOST)

    # Test 3: Pandas KeyError with NaN (should be handled by fine-tuned model)
    test_query("Query 3: Pandas KeyError", QUERY_3_PANDAS)

    print("\n" + "=" * 80)
    print("Demo Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
