# Filename: code_3_v1.py
import os
import re
import gc
import sys
import time
import json
import math
import logging
import warnings
from typing import Tuple, List

import numpy as np
import pandas as pd
from scipy import sparse as sp
from scipy.optimize import minimize

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")

# Optional GPU detection
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

# ------------------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------
BASE_DIR = "task/learning-agency-lab-automated-essay-scoring-2"
TRAIN_PATH = os.path.join(BASE_DIR, "train.csv")
TEST_PATH = os.path.join(BASE_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_DIR, "sample_submission.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "3")
SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")


# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------
def seed_everything(seed: int = 42):
    try:
        import random
        random.seed(seed)
    except Exception:
        pass
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        try:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        except Exception:
            pass


def detect_cuda() -> bool:
    gpu_torch = False
    if TORCH_AVAILABLE:
        try:
            gpu_torch = torch.cuda.is_available()
        except Exception:
            gpu_torch = False
    logger.info(f"PyTorch available: {TORCH_AVAILABLE}, CUDA available via torch: {gpu_torch}")

    # Verify if XGBoost GPU is available by a quick dummy training
    xgb_gpu = False
    if XGB_AVAILABLE:
        try:
            dtrain = xgb.DMatrix(np.random.randn(100, 10), label=np.random.randint(0, 2, 100))
            params = {"objective": "binary:logistic", "tree_method": "gpu_hist", "verbosity": 0}
            xgb.train(params, dtrain, num_boost_round=1)
            xgb_gpu = True
        except Exception as e:
            logger.info(f"XGBoost GPU not usable: {e}")
            xgb_gpu = False
    logger.info(f"XGBoost available: {XGB_AVAILABLE}, GPU usable in XGBoost: {xgb_gpu}")
    return gpu_torch or xgb_gpu


# ------------------------------------------------------------------------------
# Text preprocessing and feature engineering
# ------------------------------------------------------------------------------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    text = re.sub(r"<[^>]+>", " ", text)             # remove HTML tags
    text = re.sub(r"http\S+|www\.\S+", " ", text)    # remove URLs
    text = re.sub(r"\.{3,}", ".", text)              # normalize ellipses
    text = re.sub(r"\s+", " ", text)                 # normalize spaces
    text = text.lower().strip()
    return text


def split_sentences(text: str) -> List[str]:
    # Simple sentence splitter on punctuation
    sents = re.split(r"[.!?]+", text)
    sents = [s.strip() for s in sents if len(s.strip()) > 0]
    if len(sents) == 0:
        sents = [text.strip()] if text.strip() else [""]
    return sents


def basic_stats(text: str) -> dict:
    sents = split_sentences(text)
    sentence_count = max(1, len(sents))
    words = re.findall(r"\b\w+\b", text)
    word_count = len(words)
    char_count = len(text)
    avg_word_length = float(np.mean([len(w) for w in words])) if word_count > 0 else 0.0

    comma_count = text.count(",")
    period_count = text.count(".")
    qmark_count = text.count("?")
    exclam_count = text.count("!")

    comma_per_sentence = comma_count / sentence_count
    period_per_sentence = period_count / sentence_count
    question_mark_per_sentence = qmark_count / sentence_count
    exclam_per_sentence = exclam_count / sentence_count

    complex_words = [w for w in words if len(w) > 6]
    complex_word_ratio = len(complex_words) / word_count if word_count > 0 else 0.0

    first_sentence_length = len(re.findall(r"\b\w+\b", sents[0])) if len(sents) > 0 else 0
    last_sentence_length = len(re.findall(r"\b\w+\b", sents[-1])) if len(sents) > 0 else 0

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "char_count": char_count,
        "avg_word_length": avg_word_length,
        "comma_per_sentence": comma_per_sentence,
        "period_per_sentence": period_per_sentence,
        "question_mark_per_sentence": question_mark_per_sentence,
        "exclam_per_sentence": exclam_per_sentence,
        "complex_word_ratio": complex_word_ratio,
        "first_sentence_length": first_sentence_length,
        "last_sentence_length": last_sentence_length,
    }


def build_numeric_features(texts: List[str]) -> np.ndarray:
    feats = []
    for t in texts:
        feats.append(basic_stats(t))
    df = pd.DataFrame(feats)
    cols = df.columns.tolist()
    logger.info(f"Numeric feature columns: {cols}")
    X_num = df.values.astype(np.float32)
    return X_num


# ------------------------------------------------------------------------------
# Metrics: Quadratic Weighted Kappa
# ------------------------------------------------------------------------------
def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray, min_rating: int = 1, max_rating: int = 6) -> float:
    # Ensure integer labels within range
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    assert y_true.shape == y_pred.shape

    N = max_rating - min_rating + 1
    # O: observed matrix
    O = np.zeros((N, N), dtype=np.float64)
    for a, p in zip(y_true, y_pred):
        O[a - min_rating, p - min_rating] += 1.0

    # histograms
    act_hist = np.sum(O, axis=1)
    pred_hist = np.sum(O, axis=0)

    # Expected matrix E
    E = np.outer(act_hist, pred_hist) / np.sum(O)

    # Weights
    W = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            W[i, j] = ((i - j) ** 2) / ((N - 1) ** 2)

    num = np.sum(W * O)
    den = np.sum(W * E)
    kappa = 1.0 - num / den if den != 0 else 0.0
    return kappa


# ------------------------------------------------------------------------------
# Optimized Rounder for continuous-to-ordinal mapping
# ------------------------------------------------------------------------------
class OptimizedRounder:
    def __init__(self, initial_thresholds: List[float] = None):
        if initial_thresholds is None:
            initial_thresholds = [1.5, 2.5, 3.5, 4.5, 5.5]
        self.coef_ = np.array(initial_thresholds, dtype=np.float64)

    @staticmethod
    def _apply_thresholds(y_cont: np.ndarray, thresholds: List[float]) -> np.ndarray:
        thresholds = np.sort(np.asarray(thresholds, dtype=np.float64))
        bins = [-np.inf] + thresholds.tolist() + [np.inf]
        # labels 1..6
        y_class = np.digitize(y_cont, bins)  # returns 1..6
        return y_class

    def _loss(self, coef: np.ndarray, y_true: np.ndarray, y_cont: np.ndarray) -> float:
        y_pred = self._apply_thresholds(y_cont, coef)
        # Minimize negative kappa (maximize kappa)
        return -quadratic_weighted_kappa(y_true, y_pred, min_rating=1, max_rating=6)

    def fit(self, y_true: np.ndarray, y_cont: np.ndarray):
        logger.info("Starting threshold optimization for QWK...")
        init = np.array(self.coef_, dtype=np.float64)
        bounds = [(1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 5.0), (5.0, 6.0)]
        # Powell doesn't use bounds; alternative: Nelder-Mead. We rely on sorting inside loss.
        res = minimize(self._loss, init, args=(y_true, y_cont), method="Powell", options={"maxiter": 2000, "disp": False})
        self.coef_ = np.sort(res.x)
        logger.info(f"Optimized thresholds: {self.coef_.tolist()} | Success: {res.success}")
        return self

    def predict(self, y_cont: np.ndarray) -> np.ndarray:
        return self._apply_thresholds(y_cont, self.coef_)


# ------------------------------------------------------------------------------
# Model training using XGBoost (GPU if available)
# ------------------------------------------------------------------------------
def train_xgb_cv(
    X: sp.csr_matrix,
    y: np.ndarray,
    X_test: sp.csr_matrix,
    n_splits: int = 5,
    random_state: int = 42,
    use_gpu: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    logger.info("Starting XGBoost CV training...")
    y_zero_based = y - 1  # 0..5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Compute sample weights inversely proportional to class counts
    class_counts = np.bincount(y_zero_based, minlength=6).astype(np.float64)
    inv_freq = 1.0 / np.maximum(class_counts, 1.0)
    sample_weight = inv_freq[y_zero_based]
    sample_weight = sample_weight / np.mean(sample_weight)

    logger.info(f"Class counts: {class_counts.tolist()}")
    logger.info(f"Sample weight stats: mean={sample_weight.mean():.4f}, min={sample_weight.min():.4f}, max={sample_weight.max():.4f}")

    oof_proba = np.zeros((X.shape[0], 6), dtype=np.float32)
    test_proba = np.zeros((X_test.shape[0], 6), dtype=np.float32)
    fold_metrics = []

    # Base params
    params = {
        "objective": "multi:softprob",
        "num_class": 6,
        "eval_metric": "mlogloss",
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.6,
        "min_child_weight": 2.0,
        "lambda": 1.0,
        "alpha": 0.0,
        "verbosity": 1,
        "random_state": random_state,
        "nthread": max(1, os.cpu_count() or 4),
    }

    if use_gpu:
        params["tree_method"] = "gpu_hist"
        params["predictor"] = "gpu_predictor"
        logger.info("Configured XGBoost to use GPU.")
    else:
        params["tree_method"] = "hist"
        params["predictor"] = "auto"
        logger.info("Configured XGBoost to use CPU.")

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        logger.info(f"Fold {fold} - Train size: {len(tr_idx)}, Valid size: {len(va_idx)}")

        X_tr = X[tr_idx]
        X_va = X[va_idx]
        y_tr = y_zero_based[tr_idx]
        y_va = y_zero_based[va_idx]
        w_tr = sample_weight[tr_idx]
        w_va = sample_weight[va_idx]

        dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
        dvalid = xgb.DMatrix(X_va, label=y_va, weight=w_va)
        dtest = xgb.DMatrix(X_test)

        watchlist = [(dtrain, "train"), (dvalid, "valid")]

        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=5000,
            evals=watchlist,
            early_stopping_rounds=100,
            verbose_eval=False,
        )

        best_iter = booster.best_iteration
        logger.info(f"Fold {fold} - Best iteration: {best_iter}")

        proba_va = booster.predict(dvalid, iteration_range=(0, best_iter + 1))
        proba_te = booster.predict(dtest, iteration_range=(0, best_iter + 1))

        oof_proba[va_idx] = proba_va
        test_proba += proba_te / n_splits

        # Evaluate QWK on validation with expected-score rounding
        expected_va = (proba_va * np.arange(1, 7, dtype=np.float32)).sum(axis=1)
        base_preds = np.rint(expected_va).clip(1, 6).astype(int)
        kappa_base = quadratic_weighted_kappa(y[va_idx], base_preds)
        logger.info(f"Fold {fold} - QWK (simple round of expected value): {kappa_base:.6f}")

        fold_metrics.append({"fold": fold, "best_iteration": int(best_iter), "qwk_simple": float(kappa_base)})

        del X_tr, X_va, y_tr, y_va, w_tr, w_va, dtrain, dvalid, dtest, booster, proba_va, proba_te
        gc.collect()

    logger.info("Completed CV training.")
    return oof_proba, test_proba, fold_metrics


# ------------------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------------------
def main():
    start_time = time.time()
    seed_everything(42)
    logger.info("Script started.")

    # Ensure output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"Output directory ensured at: {OUTPUT_DIR}")

    # Read data
    logger.info(f"Reading training data from: {TRAIN_PATH}")
    train_df = pd.read_csv(TRAIN_PATH)
    logger.info(f"Train shape: {train_df.shape}")

    logger.info(f"Reading test data from: {TEST_PATH}")
    test_df = pd.read_csv(TEST_PATH)
    logger.info(f"Test shape: {test_df.shape}")

    # Clean text
    logger.info("Cleaning text fields...")
    train_texts = train_df["full_text"].astype(str).apply(clean_text).tolist()
    test_texts = test_df["full_text"].astype(str).apply(clean_text).tolist()

    # Feature engineering - numeric
    logger.info("Building numeric features...")
    X_train_num = build_numeric_features(train_texts)
    X_test_num = build_numeric_features(test_texts)
    logger.info(f"Numeric feature shapes - train: {X_train_num.shape}, test: {X_test_num.shape}")

    # TF-IDF vectorizer
    logger.info("Fitting TF-IDF vectorizer...")
    tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=20000,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    X_train_tfidf = tfidf.fit_transform(train_texts)
    X_test_tfidf = tfidf.transform(test_texts)
    logger.info(f"TF-IDF shapes - train: {X_train_tfidf.shape}, test: {X_test_tfidf.shape}")

    # Combine features (sparse hstack)
    logger.info("Combining TF-IDF and numeric features...")
    X_train_num_sp = sp.csr_matrix(X_train_num)
    X_test_num_sp = sp.csr_matrix(X_test_num)
    X_train = sp.hstack([X_train_tfidf, X_train_num_sp], format="csr")
    X_test = sp.hstack([X_test_tfidf, X_test_num_sp], format="csr")
    logger.info(f"Combined feature shapes - train: {X_train.shape}, test: {X_test.shape}")

    # Labels
    y = train_df["score"].astype(int).values
    logger.info(f"Label distribution: {np.unique(y, return_counts=True)}")

    # Detect CUDA support
    cuda_available = detect_cuda()
    logger.info(f"CUDA available overall: {cuda_available}")

    # Train model with CV
    oof_proba, test_proba, fold_metrics = train_xgb_cv(
        X=X_train,
        y=y,
        X_test=X_test,
        n_splits=5,
        random_state=42,
        use_gpu=cuda_available,
    )

    logger.info(f"Fold metrics: {json.dumps(fold_metrics, indent=2)}")

    # Post-processing: expected value + threshold optimization
    logger.info("Post-processing predictions with threshold optimization...")
    oof_expected = (oof_proba * np.arange(1, 7, dtype=np.float32)).sum(axis=1)
    initial_pred = np.rint(oof_expected).clip(1, 6).astype(int)
    base_qwk = quadratic_weighted_kappa(y, initial_pred)
    logger.info(f"OOF QWK with simple rounding: {base_qwk:.6f}")

    opt_rounder = OptimizedRounder()
    opt_rounder.fit(y_true=y, y_cont=oof_expected)
    oof_preds_opt = opt_rounder.predict(oof_expected)
    oof_qwk_opt = quadratic_weighted_kappa(y, oof_preds_opt)
    logger.info(f"OOF QWK with optimized thresholds: {oof_qwk_opt:.6f}")

    # Apply to test
    test_expected = (test_proba * np.arange(1, 7, dtype=np.float32)).sum(axis=1)
    test_preds = opt_rounder.predict(test_expected).astype(int)
    test_preds = np.clip(test_preds, 1, 6)

    # Prepare submission
    logger.info("Preparing submission file...")
    sub = pd.DataFrame({
        "essay_id": test_df["essay_id"].astype(str).values,
        "score": test_preds
    })
    logger.info(f"Submission head:\n{sub.head()}")

    # Ensure correct path and write
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sub.to_csv(SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to: {SUBMISSION_PATH}")

    elapsed = time.time() - start_time
    logger.info(f"Script finished in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()