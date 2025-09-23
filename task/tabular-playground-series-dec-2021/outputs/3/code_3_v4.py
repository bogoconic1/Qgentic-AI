import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import logging

# Prepare output directories before configuring logging
BASE_DIR = "task/tabular-playground-series-dec-2021"
OUT_DIR = os.path.join(BASE_DIR, "outputs", "3")
os.makedirs(OUT_DIR, exist_ok=True)
LOG_FILE = os.path.join(OUT_DIR, "code_3_v4.txt")
SUBMISSION_FILE = os.path.join(OUT_DIR, "submission_4.csv")

# Configure logging at the very beginning (before any logging.info calls)
logging.basicConfig(
    filename=LOG_FILE,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logging.info("Logging initialized.")
logging.info(f"Outputs directory: {OUT_DIR}")
logging.info(f"Log file: {LOG_FILE}")
logging.info(f"Submission path will be: {SUBMISSION_FILE}")

import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

import xgboost as xgb
from catboost import CatBoostClassifier, Pool

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
logging.info(f"Random seed set to {SEED}.")

# Log library versions for reproducibility
logging.info(f"XGBoost version: {getattr(xgb, '__version__', 'unknown')}")
try:
    import catboost
    logging.info(f"CatBoost version: {getattr(catboost, '__version__', 'unknown')}")
except Exception as _:
    logging.info("CatBoost version: unknown")

# -----------------------------
# Configurable pipeline options
# -----------------------------
config = {
    # Columns confirmed zero-variance
    "remove_zero_variance_cols": ["Soil_Type7", "Soil_Type15"],

    # Physically implausible corrections
    "distance_cols_to_clip_nonneg": [
        "Horizontal_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Horizontal_Distance_To_Fire_Points",
    ],
    "hillshade_cols": ["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"],
    "aspect_col": "Aspect",

    # One-hot categorical groups
    "wilderness_cols": [f"Wilderness_Area{i}" for i in range(1, 5)],
    "soil_cols": [f"Soil_Type{i}" for i in range(1, 41)],

    # General options
    "drop_id_from_features": True,
    "train_val_splits": 5,  # Use only the first split (single-fold as requested)
    "epsilon_div": 1e-3,    # For safe division in engineered ratios

    # XGBoost (CUDA) params
    "xgb_params": {
        "tree_method": "gpu_hist",   # CUDA acceleration
        "predictor": "gpu_predictor",
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "learning_rate": 0.05,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "n_estimators": 2000,
        "random_state": SEED,
        "gpu_id": 0,
    },
    "xgb_early_stopping": 200,

    # CatBoost (CUDA) params
    "cat_params": {
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "task_type": "GPU",      # CUDA acceleration
        "devices": "0",
        "learning_rate": 0.05,
        "depth": 8,
        "l2_leaf_reg": 3.0,
        "iterations": 2000,
        "random_seed": SEED,
        "verbose": 100,
    },
}

logging.info("Configuration prepared.")
for k, v in config.items():
    logging.info(f"CONFIG {k}: {v}")

# -----------------------------
# Utility functions
# -----------------------------
def load_data(base_dir: str):
    train_path = os.path.join(base_dir, "train.csv")
    test_path = os.path.join(base_dir, "test.csv")
    sample_sub_path = os.path.join(base_dir, "sample_submission.csv")
    logging.info(f"Loading data from {base_dir}")
    logging.info(f"Train path: {train_path}")
    logging.info(f"Test path: {test_path}")
    logging.info(f"Sample submission path: {sample_sub_path}")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample_sub = pd.read_csv(sample_sub_path)
    logging.info(f"Train shape: {train_df.shape}")
    logging.info(f"Test shape: {test_df.shape}")
    return train_df, test_df, sample_sub

def preprocess(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    logging.info(f"Starting preprocessing. is_train={is_train} | initial shape={df.shape}")
    df = df.copy()

    # 1) Drop zero-variance columns
    for col in config["remove_zero_variance_cols"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
            logging.info(f"Dropped zero-variance column: {col}")

    # 2) Clip horizontal distance features to be non-negative
    for col in config["distance_cols_to_clip_nonneg"]:
        if col in df.columns:
            min_before = df[col].min()
            df[col] = df[col].clip(lower=0)
            min_after = df[col].min()
            logging.info(f"Clipped negatives in {col}: min_before={min_before}, min_after={min_after}")

    # 3) Clip hillshade to [0, 255]
    for col in config["hillshade_cols"]:
        if col in df.columns:
            min_before, max_before = df[col].min(), df[col].max()
            df[col] = df[col].clip(lower=0, upper=255)
            min_after, max_after = df[col].min(), df[col].max()
            logging.info(f"Clipped {col} to [0,255]: before(min,max)=({min_before},{max_before}) -> after({min_after},{max_after})")

    # 4) Wrap Aspect to [0, 360) and add sin/cos
    aspect_col = config["aspect_col"]
    if aspect_col in df.columns:
        min_before, max_before = df[aspect_col].min(), df[aspect_col].max()
        df[aspect_col] = np.mod(df[aspect_col], 360)
        min_after, max_after = df[aspect_col].min(), df[aspect_col].max()
        logging.info(f"Wrapped {aspect_col} to [0,360): before(min,max)=({min_before},{max_before}) -> after({min_after},{max_after})")
        rad = np.deg2rad(df[aspect_col].values)
        df["Aspect_Sin"] = np.sin(rad)
        df["Aspect_Cos"] = np.cos(rad)
        logging.info("Added Aspect_Sin and Aspect_Cos.")

    # 5) Wilderness_Area one-hot consistency: ensure exactly one '1'
    w_cols = [c for c in config["wilderness_cols"] if c in df.columns]
    if len(w_cols) == 4:
        sums = df[w_cols].sum(axis=1)
        n_zero = int((sums == 0).sum())
        n_multi = int((sums > 1).sum())
        logging.info(f"Wilderness_Area consistency: rows with sum==0: {n_zero}; sum>1: {n_multi}")

        # Assign sum==0 to the most frequent wilderness globally
        col_means = df[w_cols].mean().values
        default_idx = int(np.argmax(col_means))
        default_col = w_cols[default_idx]
        zero_idx = df.index[sums == 0]
        if len(zero_idx) > 0:
            df.loc[zero_idx, w_cols] = 0
            df.loc[zero_idx, default_col] = 1
            logging.info(f"Assigned {len(zero_idx)} rows with sum==0 to {default_col} (global most frequent).")

        # For rows with sum>1, keep argmax and zero others
        multi_idx = df.index[sums > 1]
        if len(multi_idx) > 0:
            sub = df.loc[multi_idx, w_cols].to_numpy()
            keep_indices = np.argmax(sub, axis=1)
            new_sub = np.zeros_like(sub, dtype=sub.dtype)
            new_sub[np.arange(new_sub.shape[0]), keep_indices] = 1
            df.loc[multi_idx, w_cols] = new_sub
            logging.info(f"Resolved {len(multi_idx)} rows with multiple wilderness flags by keeping argmax only.")

    # 6) Feature engineering
    # a) Ratio: Vertical_Distance_To_Hydrology / Elevation
    if "Vertical_Distance_To_Hydrology" in df.columns and "Elevation" in df.columns:
        df["VDistHydro_by_Elev"] = df["Vertical_Distance_To_Hydrology"] / (df["Elevation"] + config["epsilon_div"])
        logging.info("Created feature VDistHydro_by_Elev.")

    # b) Straight-line distance to water
    if "Horizontal_Distance_To_Hydrology" in df.columns and "Vertical_Distance_To_Hydrology" in df.columns:
        df["Straight_Dist_To_Hydrology"] = np.sqrt(
            df["Horizontal_Distance_To_Hydrology"] ** 2 + df["Vertical_Distance_To_Hydrology"] ** 2
        )
        logging.info("Created feature Straight_Dist_To_Hydrology.")

    # c) Hillshade mean
    if all(col in df.columns for col in config["hillshade_cols"]):
        df["Hillshade_Mean"] = df[config["hillshade_cols"]].mean(axis=1)
        logging.info("Created feature Hillshade_Mean.")

    # d) Soil_Type_Kurtosis proxy (normalized sum of soil one-hots)
    soil_cols_present = [c for c in config["soil_cols"] if c in df.columns]
    if len(soil_cols_present) > 0:
        df["Soil_Type_Kurtosis"] = df[soil_cols_present].sum(axis=1) / 40.0
        logging.info("Created feature Soil_Type_Kurtosis (normalized sum of Soil_Type one-hots).")

    logging.info(f"Finished preprocessing. final shape={df.shape}")
    return df

def build_label_mappings(y: pd.Series):
    """Drop class 5 and map remaining labels to contiguous indices."""
    logging.info("Building label mappings (dropping class 5).")
    original_counts = y.value_counts().sort_index()
    logging.info(f"Original class distribution:\n{original_counts.to_string()}")

    keep_mask = y != 5
    dropped = int((~keep_mask).sum())
    logging.info(f"Dropping {dropped} samples from class 5 (unlearnable singletons).")

    y_kept = y[keep_mask]
    classes_present = sorted(y_kept.unique().tolist())
    label_to_index = {label: idx for idx, label in enumerate(classes_present)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}

    logging.info(f"Classes after drop: {classes_present}")
    logging.info(f"Label->Index mapping: {label_to_index}")
    return keep_mask, label_to_index, index_to_label

def compute_sample_weights(y_indices: np.ndarray, n_classes: int):
    """Compute per-sample weights inversely proportional to class frequency."""
    logging.info("Computing per-sample class-balanced weights.")
    counts = np.bincount(y_indices, minlength=n_classes)
    total = y_indices.shape[0]
    weights_per_class = total / (n_classes * np.maximum(counts, 1))
    weights = weights_per_class[y_indices]
    logging.info(f"Class counts: {counts.tolist()}")
    logging.info(f"Per-class weights: {weights_per_class.tolist()}")
    return weights, weights_per_class

def align_columns(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Ensure train and test have identical columns in the same order."""
    logging.info("Aligning train/test columns.")
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    common = train_cols & test_cols
    only_train = sorted(list(train_cols - test_cols))
    only_test = sorted(list(test_cols - train_cols))
    logging.info(f"Columns only in train (will drop): {only_train}")
    logging.info(f"Columns only in test (will drop): {only_test}")

    train_df_aligned = train_df.drop(columns=only_train)
    test_df_aligned = test_df.drop(columns=only_test)
    common_sorted = sorted(common)
    train_df_aligned = train_df_aligned[common_sorted]
    test_df_aligned = test_df_aligned[common_sorted]
    logging.info(f"Aligned shapes -> train: {train_df_aligned.shape}, test: {test_df_aligned.shape}")
    return train_df_aligned, test_df_aligned

# -----------------------------
# Load and preprocess data
# -----------------------------
train_df, test_df, sample_sub = load_data(BASE_DIR)

# Separate target
target_col = "Cover_Type"
y = train_df[target_col]
train_df_features = train_df.drop(columns=[target_col])
logging.info(f"Separated target column '{target_col}' from features.")

# Drop ID if configured
id_col = "Id" if "Id" in train_df_features.columns else None
if config["drop_id_from_features"] and id_col is not None:
    logging.info("Dropping Id from training features.")
    train_df_features = train_df_features.drop(columns=[id_col])

# For test, keep Id for submission but not as feature
test_ids = test_df["Id"].values if "Id" in test_df.columns else np.arange(len(test_df))
if config["drop_id_from_features"] and "Id" in test_df.columns:
    logging.info("Dropping Id from test features (kept separately for submission).")
    test_df_features = test_df.drop(columns=["Id"])
else:
    test_df_features = test_df.copy()

# Preprocess
train_df_proc = preprocess(train_df_features, is_train=True)
test_df_proc = preprocess(test_df_features, is_train=False)

# Align columns
train_df_proc, test_df_proc = align_columns(train_df_proc, test_df_proc)

# Build label mappings (drop class 5)
keep_mask, label_to_index, index_to_label = build_label_mappings(y)
y_kept = y[keep_mask].map(label_to_index).values
X_kept = train_df_proc.loc[keep_mask].copy()
logging.info(f"After dropping class 5: X_kept shape={X_kept.shape}, y_kept length={len(y_kept)}")

# Determine class count after drop
num_classes = len(index_to_label)
logging.info(f"Number of classes after drop: {num_classes}")

# Final feature matrix (NumPy for speed)
feature_cols = X_kept.columns.tolist()
logging.info(f"Number of features: {len(feature_cols)}")
X_all = X_kept[feature_cols].values
X_test = test_df_proc[feature_cols].values

# -----------------------------
# Single-fold validation split
# -----------------------------
logging.info("Creating stratified folds (will use only the first fold for single-fold early iteration).")
skf = StratifiedKFold(n_splits=config["train_val_splits"], shuffle=True, random_state=SEED)
train_idx, valid_idx = next(iter(skf.split(X_all, y_kept)))
logging.info(f"Fold indices prepared: train size={len(train_idx)}, valid size={len(valid_idx)}")

X_tr, X_va = X_all[train_idx], X_all[valid_idx]
y_tr, y_va = y_kept[train_idx], y_kept[valid_idx]
w_tr, class_weights = compute_sample_weights(y_tr, n_classes=num_classes)
w_va, _ = compute_sample_weights(y_va, n_classes=num_classes)

# -----------------------------
# XGBoost (CUDA)
# -----------------------------
logging.info("Training XGBoost (CUDA).")
xgb_params = config["xgb_params"].copy()
xgb_params["num_class"] = num_classes

xgb_model = xgb.XGBClassifier(**xgb_params)
xgb_model.fit(
    X_tr, y_tr,
    sample_weight=w_tr,
    eval_set=[(X_tr, y_tr), (X_va, y_va)],
    sample_weight_eval_set=[w_tr, w_va],
    verbose=True,
    early_stopping_rounds=config["xgb_early_stopping"],
)
xgb_va_proba = xgb_model.predict_proba(X_va)
xgb_va_pred = np.argmax(xgb_va_proba, axis=1)
xgb_va_acc = accuracy_score(y_va, xgb_va_pred)
logging.info(f"XGBoost validation accuracy: {xgb_va_acc:.6f}")

# -----------------------------
# CatBoost (CUDA)
# -----------------------------
logging.info("Training CatBoost (CUDA).")
cat_params = config["cat_params"].copy()
cat_params["classes_count"] = num_classes
# CatBoost expects per-class weights aligned to class indices
cat_params["class_weights"] = class_weights.tolist()

cat_train = Pool(X_tr, y_tr, weight=w_tr, feature_names=feature_cols)
cat_valid = Pool(X_va, y_va, weight=w_va, feature_names=feature_cols)

cat_model = CatBoostClassifier(**cat_params)
cat_model.fit(cat_train, eval_set=cat_valid, use_best_model=True)
cat_va_proba = cat_model.predict_proba(cat_valid)
cat_va_pred = np.argmax(cat_va_proba, axis=1)
cat_va_acc = accuracy_score(y_va, cat_va_pred)
logging.info(f"CatBoost validation accuracy: {cat_va_acc:.6f}")

# -----------------------------
# Ensemble (soft-voting)
# -----------------------------
logging.info("Ensembling validation predictions (soft average) from XGBoost and CatBoost.")
va_proba_stack = np.stack([xgb_va_proba, cat_va_proba], axis=0)
ens_va_proba = va_proba_stack.mean(axis=0)
ens_va_pred = np.argmax(ens_va_proba, axis=1)
ens_va_acc = accuracy_score(y_va, ens_va_pred)
logging.info(f"Ensemble validation accuracy: {ens_va_acc:.6f}")

# Per-class report for the ensemble (final validation results)
report_text = classification_report(y_va, ens_va_pred, digits=6)
logging.info("Final Ensemble validation classification report (indices correspond to remapped classes):")
for line in report_text.splitlines():
    logging.info(line)

# Also log a concise summary line
logging.info(
    f"FINAL VAL SUMMARY | XGB: {xgb_va_acc:.6f} | CAT: {cat_va_acc:.6f} | ENSEMBLE: {ens_va_acc:.6f}"
)

# -----------------------------
# Predict on test set
# -----------------------------
logging.info("Generating test predictions from each CUDA model.")
xgb_test_proba = xgb_model.predict_proba(X_test)
cat_test_proba = cat_model.predict_proba(X_test)

logging.info("Ensembling test predictions (soft average).")
test_proba_stack = np.stack([xgb_test_proba, cat_test_proba], axis=0)
ens_test_proba = test_proba_stack.mean(axis=0)
ens_test_pred_idx = np.argmax(ens_test_proba, axis=1)

# Map indices back to original labels (note: class 5 will never be predicted)
pred_labels = np.vectorize(lambda i: int(index_to_label[i]))(ens_test_pred_idx)

# -----------------------------
# Build and save submission
# -----------------------------
submission = pd.DataFrame({
    "Id": test_ids,
    "Cover_Type": pred_labels.astype(int),
})
submission.sort_values("Id", inplace=True)
submission.to_csv(SUBMISSION_FILE, index=False)
logging.info(f"Submission saved to: {SUBMISSION_FILE}")
logging.info("Pipeline complete. Logged final validation results above.")