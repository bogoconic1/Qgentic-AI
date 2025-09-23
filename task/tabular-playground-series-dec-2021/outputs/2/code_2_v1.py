import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import logging
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# =========================
# Global configuration
# =========================
CONFIG = {
    "base_dir": "task/tabular-playground-series-dec-2021",
    "output_dir": "task/tabular-playground-series-dec-2021/outputs/2",
    "log_file": "task/tabular-playground-series-dec-2021/outputs/2/code_2_v1.txt",
    "submission_path": "task/tabular-playground-series-dec-2021/outputs/2/submission_1.csv",
    "seed": 42,
    "use_single_fold_index": 0,  # early iteration: use only a single fold
    "n_splits": 5,
    "enable_models": {
        "lightgbm": True,
        "xgboost": True,
        "catboost": True
    },
    "pseudo_labeling": False,           # customizable toggle (disabled for this iteration)
    "augment_with_original": False,     # customizable toggle (disabled for this iteration)
    "drop_zero_var_cols": ["Soil_Type7", "Soil_Type15"],
    "horizontal_distance_cols": [
        "Horizontal_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Horizontal_Distance_To_Fire_Points"
    ],
    "hillshade_cols": [
        "Hillshade_9am",
        "Hillshade_Noon",
        "Hillshade_3pm"
    ],
    "aspect_col": "Aspect",
    "wilderness_cols": [f"Wilderness_Area{i}" for i in range(1, 5)],
    "soil_cols": [f"Soil_Type{i}" for i in range(1, 41)],
    "target_col": "Cover_Type",
    "id_col": "Id",
    "removed_class": 5,  # remove the lone sample of class 5 from train as per plan
}

np.random.seed(CONFIG["seed"])


# =========================
# Utilities
# =========================
def ensure_dirs():
    logging.info("Ensuring output directories exist.")
    os.makedirs(CONFIG["output_dir"], exist_ok=True)


def setup_logging():
    ensure_dirs()
    logging.basicConfig(
        filename=CONFIG["log_file"],
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    logging.info("Logging initialized. Writing logs to %s", CONFIG["log_file"])


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = CONFIG["base_dir"]
    train_path = os.path.join(base, "train.csv")
    test_path = os.path.join(base, "test.csv")
    sample_sub_path = os.path.join(base, "sample_submission.csv")

    logging.info("Loading train from %s", train_path)
    train = pd.read_csv(train_path)
    logging.info("Loading test from %s", test_path)
    test = pd.read_csv(test_path)
    logging.info("Loading sample submission from %s", sample_sub_path)
    sample_sub = pd.read_csv(sample_sub_path)

    logging.info("Train shape: %s", str(train.shape))
    logging.info("Test shape: %s", str(test.shape))
    return train, test, sample_sub


def drop_zero_variance_columns(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logging.info("Dropping zero-variance columns: %s", CONFIG["drop_zero_var_cols"])
    for col in CONFIG["drop_zero_var_cols"]:
        if col in train.columns:
            train = train.drop(columns=[col])
        if col in test.columns:
            test = test.drop(columns=[col])
    logging.info("Shapes after dropping zero-variance columns — train: %s, test: %s", str(train.shape), str(test.shape))
    return train, test


def clip_and_wrap_numeric(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Clipping negative horizontal distance values to >= 0.")
    for col in CONFIG["horizontal_distance_cols"]:
        if col in df.columns:
            before_neg = (df[col] < 0).sum()
            logging.info("Column %s has %d negative entries before clip.", col, before_neg)
            df[col] = df[col].clip(lower=0)

    logging.info("Clipping hillshade columns to [0, 255].")
    for col in CONFIG["hillshade_cols"]:
        if col in df.columns:
            below = (df[col] < 0).sum()
            above = (df[col] > 255).sum()
            logging.info("Column %s has %d < 0 and %d > 255 before clip.", col, below, above)
            df[col] = df[col].clip(lower=0, upper=255)

    if CONFIG["aspect_col"] in df.columns:
        logging.info("Wrapping aspect to [0, 360) and adding sin/cos transforms.")
        df[CONFIG["aspect_col"]] = df[CONFIG["aspect_col"]] % 360.0
        radians = np.deg2rad(df[CONFIG["aspect_col"]].astype(float))
        df["Aspect_sin"] = np.sin(radians)
        df["Aspect_cos"] = np.cos(radians)
    return df


def fix_wilderness_areas_fit(train: pd.DataFrame) -> Dict:
    logging.info("Fitting Wilderness_Area correction logic from training data.")
    wcols = [c for c in CONFIG["wilderness_cols"] if c in train.columns]
    area_sums = train[wcols].sum(axis=0)
    area_mode_idx = int(np.argmax(area_sums.values))
    area_mode_col = wcols[area_mode_idx]
    logging.info("Most frequent Wilderness_Area column in training: %s", area_mode_col)
    return {"wilderness_cols": wcols, "area_mode_idx": area_mode_idx, "area_mode_col": area_mode_col}


def fix_wilderness_areas_apply(df: pd.DataFrame, stats: Dict) -> pd.DataFrame:
    wcols = stats["wilderness_cols"]
    area_mode_idx = stats["area_mode_idx"]

    logging.info("Applying Wilderness_Area correction to dataset with shape %s", str(df.shape))
    row_sums = df[wcols].sum(axis=1)
    invalid_zero = (row_sums == 0).sum()
    invalid_multi = (row_sums > 1).sum()
    logging.info("Rows with no wilderness area assigned (sum==0): %d", invalid_zero)
    logging.info("Rows with multiple wilderness areas assigned (sum>1): %d", invalid_multi)

    # If sum==0, set the mode area to 1
    mask_zero = (row_sums == 0).values
    if mask_zero.any():
        df.loc[mask_zero, wcols] = 0
        df.loc[mask_zero, wcols[wcols.index(wcols[area_mode_idx])]] = 1

    # If sum>1, keep only the argmax column as 1, others 0
    mask_multi = (row_sums > 1).values
    if mask_multi.any():
        sub = df.loc[mask_multi, wcols]
        argmax_idx = sub.values.argmax(axis=1)
        df.loc[mask_multi, wcols] = 0
        for i, ridx in enumerate(sub.index):
            df.loc[ridx, wcols[argmax_idx[i]]] = 1

    # Validate
    row_sums_after = df[wcols].sum(axis=1)
    assert (row_sums_after == 1).all()
    logging.info("Wilderness_Area correction applied. All rows now have exactly one active wilderness area.")
    return df


def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Adding engineered features.")
    # Straight-line distance to hydrology
    if "Horizontal_Distance_To_Hydrology" in df.columns and "Vertical_Distance_To_Hydrology" in df.columns:
        df["Euclid_Dist_To_Hydrology"] = np.sqrt(
            (df["Horizontal_Distance_To_Hydrology"].astype(float) ** 2) +
            (df["Vertical_Distance_To_Hydrology"].astype(float) ** 2)
        )

    # Normalized vertical distance to hydrology by elevation
    if "Vertical_Distance_To_Hydrology" in df.columns and "Elevation" in df.columns:
        elev = df["Elevation"].replace(0, 1).astype(float)
        df["VertHydro_by_Elev"] = df["Vertical_Distance_To_Hydrology"].astype(float) / elev

    # Hillshade mean
    hcols = [c for c in CONFIG["hillshade_cols"] if c in df.columns]
    if len(hcols) == 3:
        df["Hillshade_Mean"] = df[hcols].mean(axis=1)

    # Distance sums/diffs (common in CoverType)
    if "Horizontal_Distance_To_Roadways" in df.columns and "Horizontal_Distance_To_Fire_Points" in df.columns:
        df["Roadways_minus_FirePts"] = df["Horizontal_Distance_To_Roadways"] - df["Horizontal_Distance_To_Fire_Points"]
        df["Roadways_plus_FirePts"] = df["Horizontal_Distance_To_Roadways"] + df["Horizontal_Distance_To_Fire_Points"]

    if "Horizontal_Distance_To_Roadways" in df.columns and "Horizontal_Distance_To_Hydrology" in df.columns:
        df["Roadways_minus_Hydro"] = df["Horizontal_Distance_To_Roadways"] - df["Horizontal_Distance_To_Hydrology"]
        df["Roadways_plus_Hydro"] = df["Horizontal_Distance_To_Roadways"] + df["Horizontal_Distance_To_Hydrology"]

    if "Horizontal_Distance_To_Fire_Points" in df.columns and "Horizontal_Distance_To_Hydrology" in df.columns:
        df["FirePts_minus_Hydro"] = df["Horizontal_Distance_To_Fire_Points"] - df["Horizontal_Distance_To_Hydrology"]
        df["FirePts_plus_Hydro"] = df["Horizontal_Distance_To_Fire_Points"] + df["Horizontal_Distance_To_Hydrology"]

    # Soil diversity proxy (fraction of soil one-hots activated; should be 1 ideally)
    soil_cols = [c for c in CONFIG["soil_cols"] if c in df.columns]
    if len(soil_cols) > 0:
        df["Soil_OneHot_Sum"] = df[soil_cols].sum(axis=1)
    return df


def preprocess(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Dict]:
    logging.info("Starting preprocessing pipeline.")
    # Drop known zero-variance columns
    train, test = drop_zero_variance_columns(train, test)

    # Remove the single sample of class 5 from training
    if CONFIG["target_col"] in train.columns:
        removed_class = CONFIG["removed_class"]
        num_class5 = (train[CONFIG["target_col"]] == removed_class).sum()
        logging.info("Removing class %d samples from training: %d rows", removed_class, num_class5)
        train = train.loc[train[CONFIG["target_col"]] != removed_class].copy()

    # Clip and wrap numeric features
    train = clip_and_wrap_numeric(train)
    test = clip_and_wrap_numeric(test)

    # Wilderness correction
    w_stats = fix_wilderness_areas_fit(train)
    train = fix_wilderness_areas_apply(train, w_stats)
    test = fix_wilderness_areas_apply(test, w_stats)

    # Feature engineering
    train = add_feature_engineering(train)
    test = add_feature_engineering(test)

    # Align columns between train and test (without target)
    drop_cols = [CONFIG["target_col"], CONFIG["id_col"]]
    feature_cols = [c for c in train.columns if c not in drop_cols]
    feature_cols_test = [c for c in test.columns if c != CONFIG["id_col"]]
    logging.info("Number of initial train features: %d, test features: %d", len(feature_cols), len(feature_cols_test))

    # Make sure both sets share identical columns and ordering
    common_cols = sorted(list(set(feature_cols).intersection(set(feature_cols_test))))
    logging.info("Number of common features retained: %d", len(common_cols))

    X = train[common_cols].copy()
    y = train[CONFIG["target_col"]].copy()
    X_test = test[common_cols].copy()
    test_ids = test[CONFIG["id_col"]].copy()

    logging.info("Preprocessing completed. X shape: %s, X_test shape: %s", str(X.shape), str(X_test.shape))
    return X, y, X_test, test_ids, {"feature_cols": common_cols, "wilderness_stats": w_stats}


def compute_class_weights(y: pd.Series) -> Dict[int, float]:
    logging.info("Computing class weights (inverse frequency).")
    classes, counts = np.unique(y, return_counts=True)
    num_classes = len(classes)
    total = len(y)
    weights = {}
    for c, cnt in zip(classes, counts):
        weights[int(c)] = total / (num_classes * cnt)
        logging.info("Class %d count=%d, weight=%.6f", c, cnt, weights[int(c)])
    return weights


def map_labels(y_orig: pd.Series, ordered_original_labels: List[int]) -> Tuple[np.ndarray, Dict[int, int], Dict[int, int]]:
    logging.info("Mapping original labels to model indices (compact 0..K-1).")
    orig_to_model = {lab: i for i, lab in enumerate(ordered_original_labels)}
    model_to_orig = {i: lab for lab, i in orig_to_model.items()}
    y_mapped = np.array([orig_to_model[int(v)] for v in y_orig.values])
    logging.info("Label mapping: %s", str(orig_to_model))
    return y_mapped, orig_to_model, model_to_orig


def get_train_valid_indices(y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    logging.info("Creating single-fold stratified split using StratifiedKFold with n_splits=%d.", CONFIG["n_splits"])
    skf = StratifiedKFold(n_splits=CONFIG["n_splits"], shuffle=True, random_state=CONFIG["seed"])
    fold_idx = CONFIG["use_single_fold_index"]
    for i, (tr, va) in enumerate(skf.split(np.zeros_like(y), y)):
        if i == fold_idx:
            logging.info("Selected fold index: %d (train size=%d, valid size=%d)", i, len(tr), len(va))
            return tr, va
    # Should never reach here given valid fold_idx; returning last if needed
    logging.info("Fallback to last fold index due to index out of range in loop (this should not happen).")
    return tr, va


def prepare_sample_weights(y_train_orig: np.ndarray, class_weights_orig: Dict[int, float]) -> np.ndarray:
    logging.info("Preparing per-sample weights based on original labels.")
    weights = np.array([class_weights_orig[int(lbl)] for lbl in y_train_orig])
    logging.info("Sample weights stats -> mean=%.6f, min=%.6f, max=%.6f", weights.mean(), weights.min(), weights.max())
    return weights


def train_lightgbm(X_tr, y_tr, X_va, y_va, sample_weight_tr, num_classes: int) -> lgb.LGBMClassifier:
    logging.info("Training LightGBM (GPU) multiclass model.")
    lgbm = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=num_classes,
        n_estimators=10000,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=256,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        min_child_samples=40,
        random_state=CONFIG["seed"],
        device="gpu",          # ensure GPU
        max_bin=255
    )
    lgbm.fit(
        X_tr, y_tr,
        sample_weight=sample_weight_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=200)]
    )
    logging.info("LightGBM best iteration: %d", lgbm.best_iteration_)
    logging.info("LightGBM training completed.")
    return lgbm


def train_xgboost(X_tr, y_tr, X_va, y_va, sample_weight_tr, num_classes: int) -> xgb.XGBClassifier:
    logging.info("Training XGBoost (GPU) multiclass model.")
    xgb_clf = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        n_estimators=20000,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        tree_method="gpu_hist",
        predictor="gpu_predictor",
        gpu_id=0,
        random_state=CONFIG["seed"],
        min_child_weight=2,
        max_bin=256,
        verbosity=1,
    )
    xgb_clf.fit(
        X_tr, y_tr,
        sample_weight=sample_weight_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="mlogloss",
        early_stopping_rounds=300,
        verbose=False
    )
    logging.info("XGBoost best iteration: %s", str(xgb_clf.best_iteration))
    logging.info("XGBoost training completed.")
    return xgb_clf


def train_catboost(X_tr, y_tr, X_va, y_va, sample_weight_tr, num_classes: int) -> CatBoostClassifier:
    logging.info("Training CatBoost (GPU) multiclass model.")
    cat = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=20000,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3.0,
        random_seed=CONFIG["seed"],
        task_type="GPU",
        devices="0",
        bootstrap_type="Bernoulli",
        subsample=0.8,
        rsm=0.8,
        od_type="Iter",
        od_wait=300,
        verbose=False
    )
    cat.fit(
        X_tr, y_tr,
        sample_weight=sample_weight_tr,
        eval_set=(X_va, y_va),
        use_best_model=True
    )
    logging.info("CatBoost best iteration: %d", cat.get_best_iteration())
    logging.info("CatBoost training completed.")
    return cat


def probs_to_seven_classes(probs_6: np.ndarray, ordered_original_labels: List[int]) -> np.ndarray:
    # Map 6-class probs (for labels [1,2,3,4,6,7]) into 7-class ordering [1..7], inserting zeros for class 5.
    logging.info("Expanding probability matrix from 6 classes to 7 classes with zero-prob for class 5.")
    n = probs_6.shape[0]
    probs_7 = np.zeros((n, 7), dtype=np.float32)
    # Original label -> column index in 7-class array: (label-1)
    for model_idx, orig_label in enumerate(ordered_original_labels):
        col_idx = orig_label - 1
        probs_7[:, col_idx] = probs_6[:, model_idx]
    # class 5 (index 4) remains zeros
    row_sums = probs_7.sum(axis=1, keepdims=True)
    # Normalize to sum 1 (since we inserted zeros)
    row_sums[row_sums == 0] = 1.0
    probs_7 = probs_7 / row_sums
    return probs_7


def evaluate_and_log(y_true_orig: np.ndarray, pred_probs_6: np.ndarray, ordered_original_labels: List[int], tag: str):
    logging.info("Evaluating %s on validation set.", tag)
    probs_7 = probs_to_seven_classes(pred_probs_6, ordered_original_labels)
    y_pred_orig = np.argmax(probs_7, axis=1) + 1  # back to 1..7 space
    acc = accuracy_score(y_true_orig, y_pred_orig)
    logging.info("[%s] Validation Accuracy: %.6f", tag, acc)
    report = classification_report(y_true_orig, y_pred_orig, labels=[1,2,3,4,5,6,7], zero_division=0, digits=6)
    logging.info("[%s] Classification Report:\n%s", tag, report)
    cm = confusion_matrix(y_true_orig, y_pred_orig, labels=[1,2,3,4,5,6,7])
    logging.info("[%s] Confusion Matrix (rows=true, cols=pred in label order 1..7):\n%s", tag, cm)
    return acc, report


def main():
    setup_logging()
    logging.info("Pipeline configuration: %s", str(CONFIG))
    train, test, sample_sub = load_data()

    # Preprocess
    X, y, X_test, test_ids, pp_stats = preprocess(train, test)

    # Determine ordered labels after removing class 5
    present_classes = sorted([c for c in np.unique(y) if int(c) != CONFIG["removed_class"]])
    ordered_original_labels = present_classes  # e.g., [1,2,3,4,6,7]
    logging.info("Ordered original labels for modeling (class 5 removed): %s", ordered_original_labels)

    # Split into single fold (train/valid)
    tr_idx, va_idx = get_train_valid_indices(y)
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr_orig, y_va_orig = y.iloc[tr_idx].values, y.iloc[va_idx].values

    # Class weights and sample weights (based on original labels, not remapped)
    class_weights_orig = compute_class_weights(pd.Series(y_tr_orig))
    sample_weight_tr = prepare_sample_weights(y_tr_orig, class_weights_orig)

    # Map labels to compact indices 0..K-1 for modeling
    y_tr_map, orig_to_model, model_to_orig = map_labels(pd.Series(y_tr_orig), ordered_original_labels)
    y_va_map = np.array([orig_to_model[int(v)] for v in y_va_orig])

    # Train models
    models = {}
    val_probs = {}

    if CONFIG["enable_models"]["lightgbm"]:
        logging.info("Starting LightGBM training...")
        lgbm = train_lightgbm(X_tr, y_tr_map, X_va, y_va_map, sample_weight_tr, num_classes=len(ordered_original_labels))
        models["lightgbm"] = lgbm
        val_probs["lightgbm"] = lgbm.predict_proba(X_va, num_iteration=lgbm.best_iteration_)
        evaluate_and_log(y_va_orig, val_probs["lightgbm"], ordered_original_labels, tag="LightGBM")

    if CONFIG["enable_models"]["xgboost"]:
        logging.info("Starting XGBoost training...")
        xgb_clf = train_xgboost(X_tr, y_tr_map, X_va, y_va_map, sample_weight_tr, num_classes=len(ordered_original_labels))
        models["xgboost"] = xgb_clf
        val_probs["xgboost"] = xgb_clf.predict_proba(X_va)
        evaluate_and_log(y_va_orig, val_probs["xgboost"], ordered_original_labels, tag="XGBoost")

    if CONFIG["enable_models"]["catboost"]:
        logging.info("Starting CatBoost training...")
        cat = train_catboost(X_tr, y_tr_map, X_va, y_va_map, sample_weight_tr, num_classes=len(ordered_original_labels))
        models["catboost"] = cat
        val_probs["catboost"] = cat.predict_proba(X_va)
        evaluate_and_log(y_va_orig, val_probs["catboost"], ordered_original_labels, tag="CatBoost")

    # Ensemble via soft-voting (average probabilities)
    logging.info("Creating soft-voting ensemble of available models.")
    prob_list = [val_probs[k] for k in val_probs.keys()]
    ensemble_probs_6 = np.mean(prob_list, axis=0)
    ens_acc, ens_report = evaluate_and_log(y_va_orig, ensemble_probs_6, ordered_original_labels, tag="Ensemble")

    logging.info("Final validation results for Ensemble:")
    logging.info("Accuracy: %.6f", ens_acc)
    logging.info("Classification Report:\n%s", ens_report)
    logging.info("Logging final validation results completed.")

    # Fit models on full training data (optional in later iterations). For now, predict test with current models and average.
    logging.info("Generating test set predictions with trained models and writing submission.")
    test_probs_6_list = []
    if "lightgbm" in models:
        p = models["lightgbm"].predict_proba(X_test, num_iteration=models["lightgbm"].best_iteration_)
        test_probs_6_list.append(p)
    if "xgboost" in models:
        p = models["xgboost"].predict_proba(X_test)
        test_probs_6_list.append(p)
    if "catboost" in models:
        p = models["catboost"].predict_proba(X_test)
        test_probs_6_list.append(p)

    test_probs_6 = np.mean(test_probs_6_list, axis=0)
    test_probs_7 = probs_to_seven_classes(test_probs_6, ordered_original_labels)
    test_pred_labels = np.argmax(test_probs_7, axis=1) + 1  # back to 1..7

    # Build submission
    submission = pd.DataFrame({
        CONFIG["id_col"]: test_ids.values,
        CONFIG["target_col"]: test_pred_labels.astype(int)
    })
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    submission.to_csv(CONFIG["submission_path"], index=False)
    logging.info("Submission saved to %s with shape %s", CONFIG["submission_path"], str(submission.shape))
    logging.info("Done.")


if __name__ == "__main__":
    main()