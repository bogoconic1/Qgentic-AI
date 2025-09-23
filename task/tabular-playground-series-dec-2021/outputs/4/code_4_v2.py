import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import logging
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss
import xgboost as xgb
from xgboost.callback import EarlyStopping

# ---------------------------------------------------------------------------
# Directory and logging setup (basicConfig MUST be first before any logging)
# ---------------------------------------------------------------------------
BASE_DIR = "task/tabular-playground-series-dec-2021"
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs", "4")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
LOG_FILE = os.path.join(OUTPUTS_DIR, "code_4_v2.txt")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    filemode="w",
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    base_dir: str = BASE_DIR
    outputs_dir: str = OUTPUTS_DIR
    train_file: str = os.path.join(BASE_DIR, "train.csv")
    test_file: str = os.path.join(BASE_DIR, "test.csv")
    submission_file: str = os.path.join(OUTPUTS_DIR, "submission_2.csv")
    random_state: int = 42
    valid_size: float = 0.2  # Single split (early iteration requirement)
    target_col: str = "Cover_Type"
    id_col: str = "Id"
    drop_test_target_if_present: bool = True
    drop_soil_zero_variance: tuple = ("Soil_Type7", "Soil_Type15")
    hillshade_cols: tuple = ("Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm")
    horiz_distance_cols: tuple = (
        "Horizontal_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Horizontal_Distance_To_Fire_Points",
    )
    wilderness_area_cols: tuple = ("Wilderness_Area1", "Wilderness_Area2", "Wilderness_Area3", "Wilderness_Area4")
    soil_type_cols: tuple = tuple(f"Soil_Type{i}" for i in range(1, 41))
    # XGBoost params (GPU)
    xgb_n_estimators: int = 2000
    xgb_learning_rate: float = 0.05
    xgb_max_depth: int = 8
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_min_child_weight: float = 2.0
    xgb_reg_lambda: float = 1.0
    xgb_tree_method: str = "gpu_hist"
    xgb_predictor: str = "gpu_predictor"
    xgb_eval_metric: str = "mlogloss"
    xgb_early_stopping_rounds: int = 200
    # feature engineering toggles
    make_aspect_sin_cos: bool = True
    clip_hillshade: bool = True
    fix_negative_horizontal_distances: bool = True
    engineer_hydrology_features: bool = True
    engineer_hillshade_stats: bool = True
    engineer_onehot_sums: bool = True
    drop_original_aspect: bool = True
    remove_cover_type_5_from_train: bool = True  # per research plan
    # numeric stability
    dtype_float: str = "float32"

CFG = Config()
logging.info(f"Config: {asdict(CFG)}")

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def ks_statistic(x: np.ndarray, y: np.ndarray) -> float:
    logging.info("Computing KS statistic.")
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    # Evaluate CDF of y at points of x_sorted
    y_cdf_idx = np.searchsorted(y_sorted, x_sorted, side="right")
    x_cdf = np.linspace(1 / x_sorted.size, 1.0, num=x_sorted.size)
    y_cdf = y_cdf_idx / y_sorted.size
    d_stat = float(np.max(np.abs(x_cdf - y_cdf)))
    logging.info(f"KS statistic computed: {d_stat}")
    return d_stat

def load_data(cfg: Config):
    logging.info("Loading data.")
    train = pd.read_csv(cfg.train_file)
    test = pd.read_csv(cfg.test_file)
    logging.info(f"Loaded train shape: {train.shape}, test shape: {test.shape}")
    return train, test

def drop_invalid_test_target(test: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    logging.info("Checking for presence of target in test and dropping if present.")
    if cfg.drop_test_target_if_present and cfg.target_col in test.columns:
        logging.info(f"Found {cfg.target_col} in test; dropping it as per plan.")
        test = test.drop(columns=[cfg.target_col])
    return test

def drop_zero_variance_soil(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    for col in cfg.drop_soil_zero_variance:
        if col in df.columns:
            logging.info(f"Dropping zero-variance soil column: {col}")
            df = df.drop(columns=[col])
    return df

def clip_hillshade_values(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if cfg.clip_hillshade:
        logging.info("Clipping hillshade values to [0, 255].")
        for c in cfg.hillshade_cols:
            if c in df.columns:
                df[c] = df[c].clip(lower=0, upper=255)
    return df

def fix_negative_horizontal(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if cfg.fix_negative_horizontal_distances:
        logging.info("Fixing negative horizontal distances by setting negatives to 0.")
        for c in cfg.horiz_distance_cols:
            if c in df.columns:
                negatives = int((df[c] < 0).sum())
                if negatives > 0:
                    logging.info(f"{c}: {negatives} negative values found; setting them to 0.")
                df[c] = df[c].mask(df[c] < 0, 0)
    return df

def normalize_and_encode_aspect(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if "Aspect" in df.columns and cfg.make_aspect_sin_cos:
        logging.info("Normalizing Aspect to [0, 360) and creating circular encodings.")
        aspect = df["Aspect"]
        aspect_norm = ((aspect % 360) + 360) % 360
        df["Aspect_sin"] = np.sin(np.deg2rad(aspect_norm)).astype(cfg.dtype_float)
        df["Aspect_cos"] = np.cos(np.deg2rad(aspect_norm)).astype(cfg.dtype_float)
        if cfg.drop_original_aspect:
            logging.info("Dropping original Aspect column.")
            df = df.drop(columns=["Aspect"])
    return df

def engineer_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    logging.info("Starting feature engineering.")
    # Hydrology resultant distance + direction
    if cfg.engineer_hydrology_features:
        if "Horizontal_Distance_To_Hydrology" in df.columns and "Vertical_Distance_To_Hydrology" in df.columns:
            logging.info("Engineering hydrology features.")
            hx = df["Horizontal_Distance_To_Hydrology"].astype(cfg.dtype_float)
            vz = df["Vertical_Distance_To_Hydrology"].astype(cfg.dtype_float)
            df["Total_Distance_To_Hydrology"] = np.sqrt(hx * hx + vz * vz).astype(cfg.dtype_float)
            df["To_Hydrology_Direction"] = np.arctan2(vz, hx + 1e-6).astype(cfg.dtype_float)

    # Hillshade stats
    if cfg.engineer_hillshade_stats:
        missing = [c for c in cfg.hillshade_cols if c not in df.columns]
        if len(missing) == 0:
            logging.info("Engineering hillshade mean and range features.")
            h = df.loc[:, list(cfg.hillshade_cols)].astype(cfg.dtype_float)
            df["Hillshade_Mean"] = h.mean(axis=1).astype(cfg.dtype_float)
            df["Hillshade_Range"] = (h.max(axis=1) - h.min(axis=1)).astype(cfg.dtype_float)

    # One-hot sums for wilderness and soil
    if cfg.engineer_onehot_sums:
        if all(c in df.columns for c in cfg.wilderness_area_cols):
            logging.info("Engineering Wilderness_Area count feature.")
            df["Wilderness_Area_Count"] = df.loc[:, list(cfg.wilderness_area_cols)].sum(axis=1).astype(cfg.dtype_float)
        soil_cols_present = [c for c in cfg.soil_type_cols if c in df.columns]
        if len(soil_cols_present) > 0:
            logging.info("Engineering Soil_Type sum feature.")
            df["Soil_Type_Sum"] = df.loc[:, soil_cols_present].sum(axis=1).astype(cfg.dtype_float)

    logging.info("Finished feature engineering.")
    return df

def wilderness_and_soil_sum_diagnostics(train: pd.DataFrame, test: pd.DataFrame, cfg: Config):
    logging.info("Running diagnostics for Wilderness_Area and Soil_Type sum distributions.")
    if all(c in train.columns for c in cfg.wilderness_area_cols):
        train_sum = train.loc[:, list(cfg.wilderness_area_cols)].sum(axis=1)
        test_sum = test.loc[:, list(cfg.wilderness_area_cols)].sum(axis=1)
        def freq_table(s: pd.Series):
            vals, counts = np.unique(s.values, return_counts=True)
            return {int(v): int(c) for v, c in zip(vals, counts)}
        logging.info(f"Wilderness_Area sum freq (train): {freq_table(train_sum)}")
        logging.info(f"Wilderness_Area sum freq (test): {freq_table(test_sum)}")

    soil_present_train = [c for c in cfg.soil_type_cols if c in train.columns]
    soil_present_test = [c for c in cfg.soil_type_cols if c in test.columns]
    if len(soil_present_train) > 0:
        train_sum = train.loc[:, soil_present_train].sum(axis=1)
        vt, ct = np.unique(train_sum.values, return_counts=True)
        logging.info(f"Soil_Type sum freq (train): {{"
                     + ", ".join([f"{int(v)}: {int(c)}" for v, c in zip(vt, ct)]) + "}}")
    if len(soil_present_test) > 0:
        test_sum = test.loc[:, soil_present_test].sum(axis=1)
        vt, ct = np.unique(test_sum.values, return_counts=True)
        logging.info(f"Soil_Type sum freq (test): {{"
                     + ", ".join([f"{int(v)}: {int(c)}" for v, c in zip(vt, ct)]) + "}}")

def compute_class_weights(y: np.ndarray) -> dict:
    logging.info("Computing class weights (inverse frequency).")
    classes, counts = np.unique(y, return_counts=True)
    total = y.shape[0]
    n_classes = classes.size
    weights = {int(cls): float(total / (n_classes * cnt)) for cls, cnt in zip(classes, counts)}
    logging.info(f"Class distribution: { {int(k): int(v) for k, v in zip(classes, counts)} }")
    logging.info(f"Class weights: {weights}")
    return weights

def prepare_matrices(train: pd.DataFrame, test: pd.DataFrame, cfg: Config):
    logging.info("Preparing training and test matrices.")
    # Ensure identical feature columns order
    feature_cols = [c for c in train.columns if c not in (cfg.target_col,)]
    if cfg.id_col in feature_cols:
        feature_cols.remove(cfg.id_col)
    # Align columns with test (in case of dropped columns)
    test_feature_cols = [c for c in test.columns if c != cfg.id_col]
    common_cols = sorted(list(set(feature_cols).intersection(set(test_feature_cols))))
    logging.info(f"Number of common feature columns: {len(common_cols)}")
    X_train = train[common_cols].astype(cfg.dtype_float)
    X_test = test[common_cols].astype(cfg.dtype_float)
    return X_train, X_test, common_cols

def train_single_fold_xgb(X: pd.DataFrame, y: np.ndarray, X_test: pd.DataFrame, cfg: Config, label_encoder: LabelEncoder):
    logging.info("Starting single-fold stratified split for validation (early iteration).")
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=cfg.valid_size, random_state=cfg.random_state)
    tr_idx, val_idx = next(splitter.split(X, y))

    X_tr = X.iloc[tr_idx]
    y_tr = y[tr_idx]
    X_val = X.iloc[val_idx]
    y_val = y[val_idx]

    logging.info(f"Train fold shape: {X_tr.shape}, Validation fold shape: {X_val.shape}")

    # Compute sample weights for imbalanced classes (on training subset)
    weights_full = compute_class_weights(y_tr)
    sample_weight_tr = np.array([weights_full[int(cls)] for cls in y_tr], dtype=np.float32)

    num_class = len(label_encoder.classes_)
    logging.info(f"Number of classes after potential filtering: {num_class}")

    clf = xgb.XGBClassifier(
        n_estimators=cfg.xgb_n_estimators,
        learning_rate=cfg.xgb_learning_rate,
        max_depth=cfg.xgb_max_depth,
        subsample=cfg.xgb_subsample,
        colsample_bytree=cfg.xgb_colsample_bytree,
        min_child_weight=cfg.xgb_min_child_weight,
        reg_lambda=cfg.xgb_reg_lambda,
        objective="multi:softprob",
        num_class=num_class,
        tree_method=cfg.xgb_tree_method,
        predictor=cfg.xgb_predictor,
        eval_metric=cfg.xgb_eval_metric,
        random_state=cfg.random_state,
        n_jobs=-1,
        verbosity=1,
    )

    logging.info("Fitting XGBoost model on GPU with sample weights (using EarlyStopping callback).")
    clf.fit(
        X_tr,
        y_tr,
        sample_weight=sample_weight_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[EarlyStopping(rounds=cfg.xgb_early_stopping_rounds, save_best=True)]
    )

    logging.info(f"Best iteration: {getattr(clf, 'best_iteration', None)}")
    logging.info("Evaluating validation performance.")
    if getattr(clf, "best_iteration", None) is not None:
        val_proba = clf.predict_proba(X_val, iteration_range=(0, clf.best_iteration + 1))
    else:
        val_proba = clf.predict_proba(X_val)
    val_pred = np.argmax(val_proba, axis=1)
    val_acc = accuracy_score(y_val, val_pred)
    val_logloss = log_loss(y_val, val_proba, labels=np.arange(num_class))
    logging.info(f"Validation accuracy: {val_acc:.6f}")
    logging.info(f"Validation logloss: {val_logloss:.6f}")

    logging.info("Predicting on test data for submission.")
    if getattr(clf, "best_iteration", None) is not None:
        test_proba = clf.predict_proba(X_test, iteration_range=(0, clf.best_iteration + 1))
    else:
        test_proba = clf.predict_proba(X_test)
    test_pred = np.argmax(test_proba, axis=1)
    # Map back to original label space
    test_pred_labels = label_encoder.inverse_transform(test_pred)
    return test_pred_labels, val_acc, val_logloss

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
logging.info("==== Pipeline start ====")

# 1) Load
train_df, test_df = load_data(CFG)

# 2) Optional: Drop leaked/invalid target in test (as per research plan)
test_df = drop_invalid_test_target(test_df, CFG)

# 3) Drop zero-variance soil columns (Soil_Type7, Soil_Type15) in both
train_df = drop_zero_variance_soil(train_df, CFG)
test_df = drop_zero_variance_soil(test_df, CFG)

# 4) Clip hillshade values to physical bounds
train_df = clip_hillshade_values(train_df, CFG)
test_df = clip_hillshade_values(test_df, CFG)

# 5) Fix negative horizontal distances
train_df = fix_negative_horizontal(train_df, CFG)
test_df = fix_negative_horizontal(test_df, CFG)

# 6) Circular encoding for Aspect and drop original Aspect
train_df = normalize_and_encode_aspect(train_df, CFG)
test_df = normalize_and_encode_aspect(test_df, CFG)

# 7) Feature engineering
train_df = engineer_features(train_df, CFG)
test_df = engineer_features(test_df, CFG)

# 8) Wilderness/Soil diagnostics (no mutation, only logs)
wilderness_and_soil_sum_diagnostics(train_df, test_df, CFG)

# 9) Target handling: remove Cover_Type == 5 from train (rare class) per plan
if CFG.remove_cover_type_5_from_train and CFG.target_col in train_df.columns:
    logging.info("Removing rows with Cover_Type == 5 from training data as specified.")
    before = train_df.shape[0]
    train_df = train_df[train_df[CFG.target_col] != 5]
    after = train_df.shape[0]
    logging.info(f"Removed {before - after} rows from training (Cover_Type == 5).")

# 10) Prepare matrices and label encoding
logging.info("Preparing features and labels.")
assert CFG.target_col in train_df.columns, "Target column not found in training data."
y_raw = train_df[CFG.target_col].values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw)
logging.info(f"Encoded classes: {label_encoder.classes_.tolist()}")

# Feature matrices aligned between train and test
X_train, X_test, used_features = prepare_matrices(train_df, test_df, CFG)
logging.info(f"Using {len(used_features)} features.")

# 11) Optional distribution shift diagnostics via KS statistic (log only)
if "Elevation" in X_train.columns:
    logging.info("Running simple distribution shift diagnostics (KS) on Elevation.")
    ks_elev = ks_statistic(X_train["Elevation"].values.astype(np.float64),
                           X_test["Elevation"].values.astype(np.float64))
    logging.info(f"KS statistic Elevation (train vs test): {ks_elev:.6f}")
if "Hillshade_Mean" in X_train.columns:
    logging.info("Running simple distribution shift diagnostics (KS) on Hillshade_Mean.")
    ks_hmean = ks_statistic(X_train["Hillshade_Mean"].values.astype(np.float64),
                            X_test["Hillshade_Mean"].values.astype(np.float64))
    logging.info(f"KS statistic Hillshade_Mean (train vs test): {ks_hmean:.6f}")

# 12) Train GPU XGBoost on single validation split, evaluate, and predict
test_pred_labels, val_acc, val_logloss = train_single_fold_xgb(X_train, y_encoded, X_test, CFG, label_encoder)

# 13) Build submission
logging.info("Building submission file.")
assert CFG.id_col in test_df.columns, "Id column not found in test data."
submission = pd.DataFrame({
    CFG.id_col: test_df[CFG.id_col].values,
    CFG.target_col: test_pred_labels.astype(int)
})
submission = submission[[CFG.id_col, CFG.target_col]]
submission.to_csv(CFG.submission_file, index=False)
logging.info(f"Saved submission to: {CFG.submission_file}")

# 14) Log final validation results and pipeline end
logging.info(f"Final validation results -> Accuracy: {val_acc:.6f}, LogLoss: {val_logloss:.6f}")
logging.info("==== Pipeline end ====")