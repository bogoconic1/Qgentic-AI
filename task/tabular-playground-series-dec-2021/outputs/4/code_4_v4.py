import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import logging
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss
from sklearn.decomposition import TruncatedSVD

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool

# -----------------------------------------------------------------------------
# Logging must be configured before any logging statements
# -----------------------------------------------------------------------------
BASE_DIR = "task/tabular-playground-series-dec-2021"
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs", "4")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
LOG_FILE = os.path.join(OUTPUTS_DIR, "code_4_v4.txt")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    filemode="w",
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass
class Config:
    base_dir: str = BASE_DIR
    outputs_dir: str = OUTPUTS_DIR
    train_file: str = os.path.join(BASE_DIR, "train.csv")
    test_file: str = os.path.join(BASE_DIR, "test.csv")
    submission_file: str = os.path.join(OUTPUTS_DIR, "submission_4.csv")

    random_state_main: int = 42
    n_splits: int = 5
    seeds: tuple = (42, 1337)  # seed bagging
    target_col: str = "Cover_Type"
    id_col: str = "Id"

    # Preprocessing / feature toggles
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
    clip_hillshade: bool = True
    fix_negative_horizontal_distances: bool = True
    make_aspect_sin_cos: bool = True
    drop_original_aspect: bool = True

    # Added engineered features per advice
    engineer_base_hydrology_features: bool = True
    engineer_distance_interactions: bool = True
    engineer_distance_log1p: bool = True
    engineer_hillshade_stats: bool = True
    engineer_aspect_slope: bool = True
    engineer_onehot_sums: bool = True
    soil_svd_components: int = 12
    soil_svd_random_state: int = 42

    remove_cover_type_5_from_train: bool = True  # per plan

    # Models: XGBoost (GPU), LightGBM (GPU, dart), CatBoost (GPU)
    # XGBoost
    xgb_params: dict = None
    xgb_n_estimators: int = 2200
    xgb_early_stopping_rounds: int = 200
    # LightGBM
    lgbm_params: dict = None
    lgbm_n_estimators: int = 5000
    lgbm_early_stopping_rounds: int = 200
    # CatBoost
    cat_params: dict = None
    cat_iterations: int = 5000
    cat_early_stopping_rounds: int = 200

    # Blending weight search
    blend_weight_step: float = 0.1
    blend_sample_size: int = 500000
    blend_random_state: int = 123

    dtype_float: str = "float32"

    def __post_init__(self):
        if self.xgb_params is None:
            self.xgb_params = dict(
                learning_rate=0.05,
                max_depth=8,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=2.0,
                reg_lambda=1.0,
                objective="multi:softprob",
                tree_method="gpu_hist",
                predictor="gpu_predictor",
                eval_metric="mlogloss",
                n_estimators=self.xgb_n_estimators,
                early_stopping_rounds=self.xgb_early_stopping_rounds,
                n_jobs=-1,
                verbosity=1,
            )
        if self.lgbm_params is None:
            # Use GPU and dart boosting per advice
            self.lgbm_params = dict(
                boosting_type="dart",
                objective="multiclass",
                num_leaves=511,
                learning_rate=0.03,
                min_data_in_leaf=64,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=1,
                reg_lambda=1.0,
                n_estimators=self.lgbm_n_estimators,
                device="gpu"  # GPU acceleration
            )
        if self.cat_params is None:
            self.cat_params = dict(
                loss_function="MultiClass",
                eval_metric="MultiClass",
                task_type="GPU",
                devices="0",
                depth=8,
                learning_rate=0.03,
                l2_leaf_reg=6.0,
                iterations=self.cat_iterations,
                verbose=0
            )

CFG = Config()
logging.info(f"Config: {asdict(CFG)}")
logging.info("==== Pipeline start ====")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def load_data(cfg: Config):
    logging.info("Loading train/test CSV files.")
    train = pd.read_csv(cfg.train_file)
    test = pd.read_csv(cfg.test_file)
    logging.info(f"Loaded shapes -> train: {train.shape}, test: {test.shape}")
    return train, test

def drop_invalid_test_target(test: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    logging.info("Checking for target leakage in test.")
    if cfg.drop_test_target_if_present and cfg.target_col in test.columns:
        logging.info(f"Found target column '{cfg.target_col}' in test; dropping it.")
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
                df[c] = df[c].clip(0, 255)
    return df

def fix_negative_horizontal(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if cfg.fix_negative_horizontal_distances:
        logging.info("Fixing negative horizontal distances (set negatives to 0).")
        for c in cfg.horiz_distance_cols:
            if c in df.columns:
                neg_count = int((df[c] < 0).sum())
                logging.info(f"{c}: negatives={neg_count}")
                df[c] = df[c].mask(df[c] < 0, 0)
    return df

def circular_aspect(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if "Aspect" in df.columns and cfg.make_aspect_sin_cos:
        logging.info("Computing Aspect_sin and Aspect_cos from Aspect.")
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

    # Base hydrology features
    if cfg.engineer_base_hydrology_features:
        if "Horizontal_Distance_To_Hydrology" in df.columns and "Vertical_Distance_To_Hydrology" in df.columns:
            logging.info("Engineering Total_Distance_To_Hydrology and To_Hydrology_Direction.")
            hx = df["Horizontal_Distance_To_Hydrology"].astype(cfg.dtype_float)
            vz = df["Vertical_Distance_To_Hydrology"].astype(cfg.dtype_float)
            df["Total_Distance_To_Hydrology"] = np.sqrt(hx * hx + vz * vz).astype(cfg.dtype_float)
            df["To_Hydrology_Direction"] = np.arctan2(vz, hx + 1e-6).astype(cfg.dtype_float)

    # Distance interactions and aggregates
    if cfg.engineer_distance_interactions and all(c in df.columns for c in cfg.horiz_distance_cols):
        logging.info("Engineering distance interaction features.")
        hH, hR, hF = cfg.horiz_distance_cols
        df["HDiff_Road"] = (df[hH] - df[hR]).abs().astype(cfg.dtype_float)
        df["HDiff_Fire"] = (df[hH] - df[hF]).abs().astype(cfg.dtype_float)
        df["Road_Fire"] = (df[hR] - df[hF]).abs().astype(cfg.dtype_float)
        dists = df[[hH, hR, hF]].astype(cfg.dtype_float)
        df["Distance_Sum"] = dists.sum(axis=1).astype(cfg.dtype_float)
        df["Distance_Mean"] = dists.mean(axis=1).astype(cfg.dtype_float)
        df["Distance_Max"] = dists.max(axis=1).astype(cfg.dtype_float)
        df["Distance_Min"] = dists.min(axis=1).astype(cfg.dtype_float)
        df["Distance_Std"] = dists.std(axis=1).astype(cfg.dtype_float)
        argmin = dists.values.argmin(axis=1)
        df["Dist_Argmin_Hydrology"] = (argmin == 0).astype(cfg.dtype_float)
        df["Dist_Argmin_Roadways"] = (argmin == 1).astype(cfg.dtype_float)
        df["Dist_Argmin_Firepoints"] = (argmin == 2).astype(cfg.dtype_float)

    # log1p transforms of distances
    if cfg.engineer_distance_log1p and all(c in df.columns for c in cfg.horiz_distance_cols):
        logging.info("Engineering log1p transforms of horizontal distances.")
        for c in cfg.horiz_distance_cols:
            df[f"log1p_{c}"] = np.log1p(df[c].astype(cfg.dtype_float)).astype(cfg.dtype_float)

    # Hillshade aggregates, diffs, ratios
    if cfg.engineer_hillshade_stats and all(c in df.columns for c in cfg.hillshade_cols):
        logging.info("Engineering hillshade stats (mean, range, std, diffs, ratios).")
        h9, hn, h3 = cfg.hillshade_cols
        h = df[[h9, hn, h3]].astype(cfg.dtype_float)
        df["Hillshade_Mean"] = h.mean(axis=1).astype(cfg.dtype_float)
        df["Hillshade_Range"] = (h.max(axis=1) - h.min(axis=1)).astype(cfg.dtype_float)
        df["Hillshade_Std"] = h.std(axis=1).astype(cfg.dtype_float)
        # diffs
        df["H9_Noondiff"] = (df[h9] - df[hn]).astype(cfg.dtype_float)
        df["Noon_3pmdiff"] = (df[hn] - df[h3]).astype(cfg.dtype_float)
        df["H9_3pmdiff"] = (df[h9] - df[h3]).astype(cfg.dtype_float)
        # ratios (safe denom)
        eps = 1e-3
        df["H9_div_Noon"] = (df[h9] / (df[hn] + eps)).astype(cfg.dtype_float)
        df["Noon_div_3pm"] = (df[hn] / (df[h3] + eps)).astype(cfg.dtype_float)
        df["H9_div_3pm"] = (df[h9] / (df[h3] + eps)).astype(cfg.dtype_float)

    # Aspect-Slope interactions
    if cfg.engineer_aspect_slope and "Slope" in df.columns:
        if "Aspect_sin" in df.columns and "Aspect_cos" in df.columns:
            logging.info("Engineering slope-aware aspect features.")
            df["Eastness_slope"] = (df["Aspect_sin"] * df["Slope"]).astype(cfg.dtype_float)
            df["Northness_slope"] = (df["Aspect_cos"] * df["Slope"]).astype(cfg.dtype_float)

    # Wilderness/Soil sums
    if cfg.engineer_onehot_sums:
        if all(c in df.columns for c in cfg.wilderness_area_cols):
            logging.info("Engineering Wilderness_Area_Count.")
            df["Wilderness_Area_Count"] = df.loc[:, list(cfg.wilderness_area_cols)].sum(axis=1).astype(cfg.dtype_float)
        soil_present = [c for c in cfg.soil_type_cols if c in df.columns]
        if len(soil_present) > 0:
            logging.info("Engineering Soil_Type_Sum.")
            df["Soil_Type_Sum"] = df.loc[:, soil_present].sum(axis=1).astype(cfg.dtype_float)

    logging.info("Finished feature engineering.")
    return df

def add_soil_svd(train: pd.DataFrame, test: pd.DataFrame, cfg: Config):
    logging.info("Adding low-rank soil embeddings via TruncatedSVD.")
    soil_cols = [c for c in cfg.soil_type_cols if c in train.columns and c in test.columns]
    if len(soil_cols) == 0:
        logging.info("No soil one-hot columns found for SVD. Skipping.")
        return train, test
    # Fit on train, transform both
    svd = TruncatedSVD(n_components=cfg.soil_svd_components, random_state=cfg.soil_svd_random_state)
    train_soil = train[soil_cols].astype(cfg.dtype_float).values
    test_soil = test[soil_cols].astype(cfg.dtype_float).values
    logging.info(f"Fitting SVD on shape: {train_soil.shape}")
    train_svd = svd.fit_transform(train_soil).astype(cfg.dtype_float)
    test_svd = svd.transform(test_soil).astype(cfg.dtype_float)
    for i in range(cfg.soil_svd_components):
        col = f"Soil_SVD_{i}"
        train[col] = train_svd[:, i]
        test[col] = test_svd[:, i]
    logging.info("Completed SVD features.")
    return train, test

def compute_class_weights(y: np.ndarray) -> dict:
    logging.info("Computing class weights (inverse frequency).")
    classes, counts = np.unique(y, return_counts=True)
    total = y.shape[0]
    n_classes = classes.size
    weights = {int(cls): float(total / (n_classes * cnt)) for cls, cnt in zip(classes, counts)}
    logging.info(f"Class distribution: { {int(k): int(v) for k, v in zip(classes, counts)} }")
    logging.info(f"Class weights: {weights}")
    return weights

def prepare_features(train: pd.DataFrame, test: pd.DataFrame, cfg: Config):
    logging.info("Preparing aligned feature matrices.")
    feat_cols = [c for c in train.columns if c not in (cfg.target_col, cfg.id_col)]
    test_cols = [c for c in test.columns if c != cfg.id_col]
    common = sorted(list(set(feat_cols).intersection(test_cols)))
    logging.info(f"Common feature columns: {len(common)}")
    X_train = train[common].astype(cfg.dtype_float)
    X_test = test[common].astype(cfg.dtype_float)
    return X_train, X_test, common

def train_cv_xgb(X, y, X_test, num_class, seed, cfg: Config):
    logging.info(f"[XGB][seed={seed}] Starting {cfg.n_splits}-fold StratifiedKFold.")
    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=seed)
    oof = np.zeros((X.shape[0], num_class), dtype=np.float32)
    test_pred_accum = np.zeros((X_test.shape[0], num_class), dtype=np.float32)
    fold_accs = []
    classes = np.arange(num_class)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        logging.info(f"[XGB][seed={seed}] Fold {fold} -> train: {len(tr_idx)}, valid: {len(va_idx)}")
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X.iloc[va_idx], y[va_idx]
        class_w = compute_class_weights(y_tr)
        sw_tr = np.array([class_w[int(cls)] for cls in y_tr], dtype=np.float32)

        params = dict(CFG.xgb_params)
        params["num_class"] = num_class
        params["random_state"] = seed

        model = xgb.XGBClassifier(**params)
        logging.info(f"[XGB][seed={seed}] Fitting model with early stopping.")
        model.fit(
            X_tr, y_tr,
            sample_weight=sw_tr,
            eval_set=[(X_va, y_va)],
            verbose=True
        )

        if getattr(model, "best_iteration", None) is not None:
            proba_va = model.predict_proba(X_va, iteration_range=(0, model.best_iteration + 1))
            proba_te = model.predict_proba(X_test, iteration_range=(0, model.best_iteration + 1))
        else:
            proba_va = model.predict_proba(X_va)
            proba_te = model.predict_proba(X_test)

        oof[va_idx] = proba_va.astype(np.float32)
        preds = np.argmax(proba_va, axis=1)
        acc = accuracy_score(y_va, preds)
        ll = log_loss(y_va, proba_va, labels=classes)
        logging.info(f"[XGB][seed={seed}] Fold {fold} ACC={acc:.6f} LogLoss={ll:.6f}")
        fold_accs.append(acc)
        # weight test predictions by fold accuracy
        test_pred_accum += (acc * proba_te.astype(np.float32))

    fold_accs = np.array(fold_accs, dtype=np.float32)
    test_pred = test_pred_accum / fold_accs.sum()
    oof_acc = accuracy_score(y, np.argmax(oof, axis=1))
    oof_ll = log_loss(y, oof, labels=classes)
    logging.info(f"[XGB][seed={seed}] OOF ACC={oof_acc:.6f} LogLoss={oof_ll:.6f}")
    return oof, test_pred, oof_acc, oof_ll

def train_cv_lgbm(X, y, X_test, num_class, seed, cfg: Config):
    logging.info(f"[LGBM-DART][seed={seed}] Starting {cfg.n_splits}-fold StratifiedKFold.")
    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=seed)
    oof = np.zeros((X.shape[0], num_class), dtype=np.float32)
    test_pred_accum = np.zeros((X_test.shape[0], num_class), dtype=np.float32)
    fold_accs = []
    classes = np.arange(num_class)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        logging.info(f"[LGBM-DART][seed={seed}] Fold {fold} -> train: {len(tr_idx)}, valid: {len(va_idx)}")
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X.iloc[va_idx], y[va_idx]
        class_w = compute_class_weights(y_tr)
        sw_tr = np.array([class_w[int(cls)] for cls in y_tr], dtype=np.float32)

        params = dict(CFG.lgbm_params)
        params["num_class"] = num_class
        params["random_state"] = seed

        model = lgb.LGBMClassifier(**params)
        logging.info(f"[LGBM-DART][seed={seed}] Fitting model with early stopping.")
        model.fit(
            X_tr, y_tr,
            sample_weight=sw_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="multi_logloss",
            verbose=True,
            early_stopping_rounds=CFG.lgbm_early_stopping_rounds
        )
        proba_va = model.predict_proba(X_va)
        proba_te = model.predict_proba(X_test)
        oof[va_idx] = proba_va.astype(np.float32)
        preds = np.argmax(proba_va, axis=1)
        acc = accuracy_score(y_va, preds)
        ll = log_loss(y_va, proba_va, labels=classes)
        logging.info(f"[LGBM-DART][seed={seed}] Fold {fold} ACC={acc:.6f} LogLoss={ll:.6f}")
        fold_accs.append(acc)
        test_pred_accum += (acc * proba_te.astype(np.float32))

    fold_accs = np.array(fold_accs, dtype=np.float32)
    test_pred = test_pred_accum / fold_accs.sum()
    oof_acc = accuracy_score(y, np.argmax(oof, axis=1))
    oof_ll = log_loss(y, oof, labels=classes)
    logging.info(f"[LGBM-DART][seed={seed}] OOF ACC={oof_acc:.6f} LogLoss={oof_ll:.6f}")
    return oof, test_pred, oof_acc, oof_ll

def train_cv_catboost(X, y, X_test, num_class, seed, cfg: Config):
    logging.info(f"[CatBoost][seed={seed}] Starting {cfg.n_splits}-fold StratifiedKFold.")
    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=seed)
    oof = np.zeros((X.shape[0], num_class), dtype=np.float32)
    test_pred_accum = np.zeros((X_test.shape[0], num_class), dtype=np.float32)
    fold_accs = []
    classes = np.arange(num_class)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        logging.info(f"[CatBoost][seed={seed}] Fold {fold} -> train: {len(tr_idx)}, valid: {len(va_idx)}")
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X.iloc[va_idx], y[va_idx]
        class_w = compute_class_weights(y_tr)
        weights_list = [class_w[i] for i in range(num_class)]

        params = dict(CFG.cat_params)
        params["random_seed"] = seed
        params["iterations"] = CFG.cat_iterations

        model = CatBoostClassifier(**params)
        logging.info(f"[CatBoost][seed={seed}] Fitting model with early stopping.")
        train_pool = Pool(X_tr, y_tr, weight=None)
        valid_pool = Pool(X_va, y_va)
        model.set_params(class_weights=weights_list)
        model.fit(
            train_pool,
            eval_set=valid_pool,
            use_best_model=True,
            early_stopping_rounds=CFG.cat_early_stopping_rounds,
            verbose=False
        )

        proba_va = model.predict_proba(X_va)
        proba_te = model.predict_proba(X_test)
        oof[va_idx] = proba_va.astype(np.float32)
        preds = np.argmax(proba_va, axis=1)
        acc = accuracy_score(y_va, preds)
        ll = log_loss(y_va, proba_va, labels=classes)
        logging.info(f"[CatBoost][seed={seed}] Fold {fold} ACC={acc:.6f} LogLoss={ll:.6f}")
        fold_accs.append(acc)
        test_pred_accum += (acc * proba_te.astype(np.float32))

    fold_accs = np.array(fold_accs, dtype=np.float32)
    test_pred = test_pred_accum / fold_accs.sum()
    oof_acc = accuracy_score(y, np.argmax(oof, axis=1))
    oof_ll = log_loss(y, oof, labels=classes)
    logging.info(f"[CatBoost][seed={seed}] OOF ACC={oof_acc:.6f} LogLoss={oof_ll:.6f}")
    return oof, test_pred, oof_acc, oof_ll

def optimize_blend_weights(oof_list, y, step, sample_size, seed):
    logging.info("Optimizing blend weights on OOF probabilities via grid search.")
    n = y.shape[0]
    rng = np.random.RandomState(seed)
    if sample_size < n:
        logging.info(f"Subsampling OOF for blending: {sample_size} of {n} (seed={seed}).")
        idx = rng.choice(n, size=sample_size, replace=False)
    else:
        logging.info("Using full OOF for blending.")
        idx = np.arange(n)

    P = [p[idx] for p in oof_list]
    y_sub = y[idx]
    best = (0.0, (1.0, 0.0, 0.0))  # (acc, weights)
    grid = np.arange(0.0, 1.0 + 1e-9, step)
    for w1 in grid:
        for w2 in grid:
            w3 = 1.0 - w1 - w2
            if w3 < -1e-9 or w3 > 1.0 + 1e-9:
                continue
            if w3 < 0:
                continue
            blended = w1 * P[0] + w2 * P[1] + w3 * P[2]
            acc = accuracy_score(y_sub, blended.argmax(axis=1))
            if acc > best[0]:
                best = (acc, (float(w1), float(w2), float(w3)))
    logging.info(f"Best blend (sample) ACC={best[0]:.6f} Weights(XGB,LGBM,Cat)={best[1]}")
    return best[1]

# -----------------------------------------------------------------------------
# Load and preprocess
# -----------------------------------------------------------------------------
train_df, test_df = load_data(CFG)
test_df = drop_invalid_test_target(test_df, CFG)

# Drop zero-variance soil in both
train_df = drop_zero_variance_soil(train_df, CFG)
test_df = drop_zero_variance_soil(test_df, CFG)

# Clip hillshade
train_df = clip_hillshade_values(train_df, CFG)
test_df = clip_hillshade_values(test_df, CFG)

# Fix negative horizontal distances
train_df = fix_negative_horizontal(train_df, CFG)
test_df = fix_negative_horizontal(test_df, CFG)

# Aspect circular encoding
train_df = circular_aspect(train_df, CFG)
test_df = circular_aspect(test_df, CFG)

# Feature engineering
train_df = engineer_features(train_df, CFG)
test_df = engineer_features(test_df, CFG)

# Soil SVD embeddings
train_df, test_df = add_soil_svd(train_df, test_df, CFG)

# Remove extremely rare class 5
if CFG.remove_cover_type_5_from_train:
    if CFG.target_col in train_df.columns:
        before = train_df.shape[0]
        train_df = train_df[train_df[CFG.target_col] != 5]
        after = train_df.shape[0]
        logging.info(f"Removed {before - after} rows with Cover_Type==5 from training.")

# Labels
assert CFG.target_col in train_df.columns, "Target column not found in training data."
y_raw = train_df[CFG.target_col].values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
num_class = len(label_encoder.classes_)
logging.info(f"Encoded classes: {label_encoder.classes_.tolist()} (num_class={num_class})")

# Features
X_train, X_test, used_features = prepare_features(train_df, test_df, CFG)
logging.info(f"Using {len(used_features)} features.")

# -----------------------------------------------------------------------------
# CV + Seed bagging for XGB, LGBM, CatBoost
# -----------------------------------------------------------------------------
xgb_oof_sum = np.zeros((X_train.shape[0], num_class), dtype=np.float32)
xgb_test_sum = np.zeros((X_test.shape[0], num_class), dtype=np.float32)
lgb_oof_sum = np.zeros_like(xgb_oof_sum)
lgb_test_sum = np.zeros_like(xgb_test_sum)
cat_oof_sum = np.zeros_like(xgb_oof_sum)
cat_test_sum = np.zeros_like(xgb_test_sum)

xgb_accs, lgb_accs, cat_accs = [], []
xgb_lls, lgb_lls, cat_lls = [], []

for seed in CFG.seeds:
    logging.info(f"==== Seed {seed} start ====")
    # XGBoost
    oof_x, te_x, acc_x, ll_x = train_cv_xgb(X_train, y, X_test, num_class, seed, CFG)
    xgb_oof_sum += oof_x
    xgb_test_sum += te_x
    xgb_accs.append(acc_x)
    xgb_lls.append(ll_x)

    # LightGBM (dart)
    oof_l, te_l, acc_l, ll_l = train_cv_lgbm(X_train, y, X_test, num_class, seed, CFG)
    lgb_oof_sum += oof_l
    lgb_test_sum += te_l
    lgb_accs.append(acc_l)
    lgb_lls.append(ll_l)

    # CatBoost
    oof_c, te_c, acc_c, ll_c = train_cv_catboost(X_train, y, X_test, num_class, seed, CFG)
    cat_oof_sum += oof_c
    cat_test_sum += te_c
    cat_accs.append(acc_c)
    cat_lls.append(ll_c)
    logging.info(f"==== Seed {seed} end ====")

n_seeds = float(len(CFG.seeds))
xgb_oof = xgb_oof_sum / n_seeds
xgb_test = xgb_test_sum / n_seeds
lgb_oof = lgb_oof_sum / n_seeds
lgb_test = lgb_test_sum / n_seeds
cat_oof = cat_oof_sum / n_seeds
cat_test = cat_test_sum / n_seeds

# Per-model OOF metrics
classes_idx = np.arange(num_class)
xgb_oof_acc = accuracy_score(y, xgb_oof.argmax(axis=1))
xgb_oof_ll = log_loss(y, xgb_oof, labels=classes_idx)
lgb_oof_acc = accuracy_score(y, lgb_oof.argmax(axis=1))
lgb_oof_ll = log_loss(y, lgb_oof, labels=classes_idx)
cat_oof_acc = accuracy_score(y, cat_oof.argmax(axis=1))
cat_oof_ll = log_loss(y, cat_oof, labels=classes_idx)

logging.info(f"Model OOF metrics after seed bagging:")
logging.info(f"  XGB  -> ACC={xgb_oof_acc:.6f} LogLoss={xgb_oof_ll:.6f}, seeds ACCs={xgb_accs}, LLs={xgb_lls}")
logging.info(f"  LGBM -> ACC={lgb_oof_acc:.6f} LogLoss={lgb_oof_ll:.6f}, seeds ACCs={lgb_accs}, LLs={lgb_lls}")
logging.info(f"  CAT  -> ACC={cat_oof_acc:.6f} LogLoss={cat_oof_ll:.6f}, seeds ACCs={cat_accs}, LLs={cat_lls}")

# -----------------------------------------------------------------------------
# Blend weights optimization on OOF (sampled for speed), then full-OFF evaluation
# -----------------------------------------------------------------------------
best_w = optimize_blend_weights(
    [xgb_oof, lgb_oof, cat_oof],
    y,
    step=CFG.blend_weight_step,
    sample_size=CFG.blend_sample_size,
    seed=CFG.blend_random_state
)

wx, wl, wc = best_w
blend_oof = wx * xgb_oof + wl * lgb_oof + wc * cat_oof
blend_acc = accuracy_score(y, blend_oof.argmax(axis=1))
blend_ll = log_loss(y, blend_oof, labels=classes_idx)
logging.info(f"Blended OOF -> ACC={blend_acc:.6f} LogLoss={blend_ll:.6f} with weights (XGB,LGBM,Cat)=({wx:.3f},{wl:.3f},{wc:.3f})")

# -----------------------------------------------------------------------------
# Create final blended test predictions and submission
# -----------------------------------------------------------------------------
blend_test = wx * xgb_test + wl * lgb_test + wc * cat_test
test_pred_idx = blend_test.argmax(axis=1)
test_pred_labels = label_encoder.inverse_transform(test_pred_idx)

logging.info("Building submission file.")
assert CFG.id_col in test_df.columns, "Id column not found in test data."
submission = pd.DataFrame({
    CFG.id_col: test_df[CFG.id_col].values,
    CFG.target_col: test_pred_labels.astype(int)
})
submission = submission[[CFG.id_col, CFG.target_col]]
submission.to_csv(CFG.submission_file, index=False)
logging.info(f"Saved submission to: {CFG.submission_file}")

# Final validation results logging
logging.info("Final validation results (OOF):")
logging.info(f"  XGB  -> ACC={xgb_oof_acc:.6f} LogLoss={xgb_oof_ll:.6f}")
logging.info(f"  LGBM -> ACC={lgb_oof_acc:.6f} LogLoss={lgb_oof_ll:.6f}")
logging.info(f"  CAT  -> ACC={cat_oof_acc:.6f} LogLoss={cat_oof_ll:.6f}")
logging.info(f"  BLEND-> ACC={blend_acc:.6f} LogLoss={blend_ll:.6f} Weights=({wx:.3f},{wl:.3f},{wc:.3f})")
logging.info("==== Pipeline end ====")