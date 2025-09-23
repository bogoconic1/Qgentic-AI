import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import logging

# Initialize logging immediately
BASE_DIR = "task/tabular-playground-series-dec-2021"
OUT_DIR = os.path.join(BASE_DIR, "outputs", "3")
os.makedirs(OUT_DIR, exist_ok=True)
LOG_FILE = os.path.join(OUT_DIR, "code_3_v7.txt")
SUBMISSION_FILE = os.path.join(OUT_DIR, "submission_7.csv")

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

import math
import time
import random
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb
import catboost
from catboost import CatBoostClassifier, Pool

import lightgbm as lgb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Reproducibility
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed_all(GLOBAL_SEED)
torch.backends.cudnn.benchmark = True
logging.info(f"Global seed set to {GLOBAL_SEED}.")

# Log library versions
logging.info(f"XGBoost version: {getattr(xgb, '__version__', 'unknown')}")
logging.info(f"CatBoost version: {getattr(catboost, '__version__', 'unknown')}")
logging.info(f"LightGBM version: {getattr(lgb, '__version__', 'unknown')}")
logging.info(f"Torch version: {torch.__version__}")
logging.info(f"CUDA available: {torch.cuda.is_available()} | device count: {torch.cuda.device_count()} | current device: {torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'}")

# -----------------------------
# Configurable pipeline options
# -----------------------------
config = {
    "zero_var_cols": ["Soil_Type7", "Soil_Type15"],
    "clip_nonneg_cols": [
        "Horizontal_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Horizontal_Distance_To_Fire_Points",
    ],
    "hillshade_cols": ["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"],
    "aspect_col": "Aspect",
    "wilderness_cols": [f"Wilderness_Area{i}" for i in range(1, 5)],
    "soil_cols": [f"Soil_Type{i}" for i in range(1, 41)],
    "drop_id_from_features": True,
    "epsilon_div": 1e-3,

    # OOF stacking and blending
    "n_folds": 5,
    "shuffle": True,
    "oof_seed": 2025,
    "use_mixup": True,
    "mixup_alpha": 0.2,

    # Class weighting (mild): sqrt inverse frequency with cap
    "class_weight_power": 0.5,
    "max_class_weight": 20.0,

    # XGBoost (CUDA) variants
    "xgb_setups": [
        {
            "name": "xgb_v1",
            "params": {
                "tree_method": "gpu_hist",
                "predictor": "gpu_predictor",
                "objective": "multi:softprob",
                "eval_metric": "mlogloss",
                "learning_rate": 0.03,
                "max_depth": 12,
                "min_child_weight": 8.0,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 5.0,
                "reg_alpha": 0.5,
                "n_estimators": 8000,
                "random_state": 1337,
                "gpu_id": 0,
                "early_stopping_rounds": 300,
            },
        },
        {
            "name": "xgb_v2",
            "params": {
                "tree_method": "gpu_hist",
                "predictor": "gpu_predictor",
                "objective": "multi:softprob",
                "eval_metric": "mlogloss",
                "learning_rate": 0.025,
                "max_depth": 14,
                "min_child_weight": 12.0,
                "subsample": 0.75,
                "colsample_bytree": 0.7,
                "reg_lambda": 10.0,
                "reg_alpha": 1.0,
                "n_estimators": 10000,
                "random_state": 2718,
                "gpu_id": 0,
                "early_stopping_rounds": 300,
            },
        },
    ],

    # CatBoost (CUDA)
    "cat_setup": {
        "name": "cat_v1",
        "params": {
            "loss_function": "MultiClass",
            "eval_metric": "MultiClass",
            "task_type": "GPU",
            "devices": "0",
            "learning_rate": 0.03,
            "depth": 10,
            "l2_leaf_reg": 8.0,
            "iterations": 4000,
            "random_seed": 4242,
            "od_type": "Iter",
            "od_wait": 300,
            "verbose": 200,
        },
    },

    # LightGBM (CUDA)
    "lgb_setup": {
        "name": "lgb_v1",
        "params": {
            "objective": "multiclass",
            "num_class": 7,
            "metric": "multi_logloss",
            "learning_rate": 0.02,
            "num_leaves": 8192,         # 2^13
            "max_depth": 14,
            "min_data_in_leaf": 2000,
            "feature_fraction": 0.75,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l1": 5.0,
            "lambda_l2": 20.0,
            "verbosity": -1,
            "device_type": "gpu",       # CUDA acceleration
            "max_bin": 255,
            "seed": 31415,
            "n_estimators": 8000,
        },
        "early_stopping_rounds": 300,
    },

    # Tabular NN (CUDA, fp16 AMP)
    "nn": {
        "name": "mlp_v1",
        "hidden_sizes": [512, 512, 256, 256],
        "dropout": 0.2,
        "epochs": 2,
        "batch_size": 8192,
        "lr": 2e-3,
        "weight_decay": 1e-4,
        "label_smoothing": 0.05,
        "warmup_steps": 50,
    },
}

for k, v in config.items():
    logging.info(f"CONFIG {k}: {v}")

# -----------------------------
# Data loading and preprocessing
# -----------------------------
def load_data(base_dir: str):
    logging.info("Loading data files.")
    train_df = pd.read_csv(os.path.join(base_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(base_dir, "test.csv"))
    sample_sub = pd.read_csv(os.path.join(base_dir, "sample_submission.csv"))
    logging.info(f"Loaded train shape: {train_df.shape}, test shape: {test_df.shape}")
    return train_df, test_df, sample_sub

def preprocess(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    logging.info(f"Preprocessing start | is_train={is_train} | shape={df.shape}")
    df = df.copy()

    # Remove zero-variance columns
    for col in config["zero_var_cols"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
            logging.info(f"Dropped zero-variance col: {col}")

    # Clip physical columns
    for col in config["clip_nonneg_cols"]:
        if col in df.columns:
            mn = df[col].min()
            df[col] = df[col].clip(lower=0)
            logging.info(f"Clipped negatives in {col} | min_before={mn} -> min_after={df[col].min()}")

    for col in config["hillshade_cols"]:
        if col in df.columns:
            mn, mx = df[col].min(), df[col].max()
            df[col] = df[col].clip(0, 255)
            logging.info(f"Clipped hillshade {col} to [0,255] | before=({mn},{mx}) after=({df[col].min()},{df[col].max()})")

    # Aspect wrap + sin/cos
    a_col = config["aspect_col"]
    if a_col in df.columns:
        mn, mx = df[a_col].min(), df[a_col].max()
        df[a_col] = np.mod(df[a_col], 360)
        logging.info(f"Wrapped {a_col} to [0,360) | before=({mn},{mx}) -> after=({df[a_col].min()},{df[a_col].max()})")
        rad = np.deg2rad(df[a_col].values)
        df["Aspect_Sin"] = np.sin(rad)
        df["Aspect_Cos"] = np.cos(rad)
        logging.info("Added Aspect_Sin and Aspect_Cos.")

    # Wilderness consistency: ensure exactly one '1'
    w_cols = [c for c in config["wilderness_cols"] if c in df.columns]
    if len(w_cols) == 4:
        sums = df[w_cols].sum(axis=1)
        zero_idx = df.index[sums == 0]
        multi_idx = df.index[sums > 1]
        if len(zero_idx) > 0:
            col_means = df[w_cols].mean().values
            default_col = w_cols[int(np.argmax(col_means))]
            df.loc[zero_idx, w_cols] = 0
            df.loc[zero_idx, default_col] = 1
            logging.info(f"Wilderness: fixed sum==0 rows: {len(zero_idx)} -> assigned {default_col}")
        if len(multi_idx) > 0:
            sub = df.loc[multi_idx, w_cols].to_numpy()
            keep = np.argmax(sub, axis=1)
            new_sub = np.zeros_like(sub)
            new_sub[np.arange(len(multi_idx)), keep] = 1
            df.loc[multi_idx, w_cols] = new_sub
            logging.info(f"Wilderness: fixed sum>1 rows: {len(multi_idx)} (keep argmax).")

    # Feature engineering
    if "Vertical_Distance_To_Hydrology" in df.columns and "Elevation" in df.columns:
        df["VDistHydro_by_Elev"] = df["Vertical_Distance_To_Hydrology"] / (df["Elevation"] + config["epsilon_div"])
        logging.info("Feature: VDistHydro_by_Elev.")

    if "Horizontal_Distance_To_Hydrology" in df.columns and "Vertical_Distance_To_Hydrology" in df.columns:
        df["Straight_Dist_To_Hydrology"] = np.sqrt(
            df["Horizontal_Distance_To_Hydrology"] ** 2 + df["Vertical_Distance_To_Hydrology"] ** 2
        )
        logging.info("Feature: Straight_Dist_To_Hydrology.")

    if all(c in df.columns for c in config["hillshade_cols"]):
        df["Hillshade_Mean"] = df[config["hillshade_cols"]].mean(axis=1)
        logging.info("Feature: Hillshade_Mean.")

    soil_present = [c for c in config["soil_cols"] if c in df.columns]
    if len(soil_present) > 0:
        df["Soil_Type_Kurtosis"] = df[soil_present].sum(axis=1) / 40.0
        logging.info("Feature: Soil_Type_Kurtosis (normalized sum of soil one-hots).")

    # Extra distance interactions
    if all(c in df.columns for c in ["Horizontal_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways"]):
        df["Hydro_Road_Diff"] = np.abs(df["Horizontal_Distance_To_Hydrology"] - df["Horizontal_Distance_To_Roadways"])
        logging.info("Feature: Hydro_Road_Diff.")
    if all(c in df.columns for c in ["Horizontal_Distance_To_Hydrology", "Horizontal_Distance_To_Fire_Points"]):
        df["Hydro_Fire_Diff"] = np.abs(df["Horizontal_Distance_To_Hydrology"] - df["Horizontal_Distance_To_Fire_Points"])
        logging.info("Feature: Hydro_Fire_Diff.")
    if all(c in df.columns for c in ["Horizontal_Distance_To_Roadways", "Horizontal_Distance_To_Fire_Points"]):
        df["Road_Fire_Diff"] = np.abs(df["Horizontal_Distance_To_Roadways"] - df["Horizontal_Distance_To_Fire_Points"])
        logging.info("Feature: Road_Fire_Diff.")
    if all(c in df.columns for c in ["Horizontal_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways", "Horizontal_Distance_To_Fire_Points"]):
        df["Dist_Sum"] = df["Horizontal_Distance_To_Hydrology"] + df["Horizontal_Distance_To_Roadways"] + df["Horizontal_Distance_To_Fire_Points"]
        logging.info("Feature: Dist_Sum.")

    logging.info(f"Preprocessing done | final shape={df.shape}")
    return df

# -----------------------------
# Load and preprocess data
# -----------------------------
train_df, test_df, sample_sub = load_data(BASE_DIR)
target_col = "Cover_Type"
y_raw = train_df[target_col].values
X = train_df.drop(columns=[target_col])
if config["drop_id_from_features"] and "Id" in X.columns:
    logging.info("Dropping Id from training features.")
    X = X.drop(columns=["Id"])
test_ids = test_df["Id"].values if "Id" in test_df.columns else np.arange(len(test_df))
X_test = test_df.drop(columns=["Id"]) if ("Id" in test_df.columns and config["drop_id_from_features"]) else test_df.copy()

X = preprocess(X, is_train=True)
X_test = preprocess(X_test, is_train=False)

# Align columns
train_cols = set(X.columns)
test_cols = set(X_test.columns)
only_train = sorted(list(train_cols - test_cols))
only_test = sorted(list(test_cols - train_cols))
logging.info(f"Aligning columns. Only in train (drop): {only_train} | Only in test (drop): {only_test}")
X = X.drop(columns=only_train)
X_test = X_test.drop(columns=only_test)
common_cols = sorted(list(set(X.columns) & set(X_test.columns)))
X = X[common_cols]
X_test = X_test[common_cols]
logging.info(f"Aligned shapes | X: {X.shape}, X_test: {X_test.shape}")

# Encode labels to 0..6 for all models
le = LabelEncoder()
y = le.fit_transform(y_raw)  # maps original {1..7} -> {0..6}
classes_mapped = dict(zip(le.classes_.tolist(), range(len(le.classes_))))
logging.info(f"Label mapping (original -> encoded 0-based): {classes_mapped}")

X_np = X.values.astype(np.float32)
X_test_np = X_test.values.astype(np.float32)

# -----------------------------
# Class weights (sqrt-inv-freq, capped)
# -----------------------------
classes_unique, class_counts = np.unique(y, return_counts=True)
logging.info(f"Encoded class distribution: {dict(zip(classes_unique.tolist(), class_counts.tolist()))}")
mean_count = class_counts.mean()
weights = (mean_count / (class_counts + 1e-12)) ** config["class_weight_power"]
weights = np.clip(weights, 0.0, config["max_class_weight"])
weights = weights / (weights.mean() + 1e-12)
class_weights = np.zeros(7, dtype=np.float32)
for c, w in zip(classes_unique, weights):
    class_weights[c] = w
logging.info(f"Class weights (sqrt-inv-freq, capped, normalized): {class_weights.tolist()}")

# -----------------------------
# Stratified folds (keep the single class-5 sample in train for every fold)
# -----------------------------
idx_all = np.arange(len(y))
mask_not_c5 = y != le.transform([5])[0]  # encoded index for original class 5
X_strat = X_np[mask_not_c5]
y_strat = y[mask_not_c5]
logging.info(f"Stratified KFold on non-class5 samples: n={X_strat.shape[0]} | class5 held-out from validation.")

skf = StratifiedKFold(n_splits=config["n_folds"], shuffle=config["shuffle"], random_state=config["oof_seed"])
fold_splits = list(skf.split(X_strat, y_strat))
logging.info(f"Prepared {len(fold_splits)} stratified folds (class 5 excluded from validation sets).")

fold_indices = []
strat_indices = np.where(mask_not_c5)[0]
c5_indices = np.where(~mask_not_c5)[0].tolist()
for k, (tr_sub, va_sub) in enumerate(fold_splits):
    tr_idx = strat_indices[tr_sub].tolist() + c5_indices
    va_idx = strat_indices[va_sub].tolist()
    fold_indices.append((np.array(tr_idx, dtype=np.int64), np.array(va_idx, dtype=np.int64)))
    logging.info(f"Fold {k}: train size={len(tr_idx)} | valid size={len(va_idx)} | c5 in train: {len(c5_indices)}")

# -----------------------------
# Containers for OOF stacking
# -----------------------------
model_names = [cfg["name"] for cfg in config["xgb_setups"]] + [config["cat_setup"]["name"], config["lgb_setup"]["name"], config["nn"]["name"]]
num_models = len(model_names)
eps = 1e-15

oof_logits = np.zeros((X_np.shape[0], num_models * 7), dtype=np.float32)
test_logits_per_model = {name: [] for name in model_names}
oof_pred_tracker = {name: {} for name in model_names}

def make_sample_weights(y_indices: np.ndarray):
    w = np.ones_like(y_indices, dtype=np.float32)
    for c in range(7):
        w[y_indices == c] = class_weights[c]
    return w

# -----------------------------
# XGBoost models (CUDA)
# -----------------------------
for xi, xgb_setup in enumerate(config["xgb_setups"]):
    name = xgb_setup["name"]
    params = xgb_setup["params"].copy()
    params["num_class"] = 7
    logging.info(f"Training model: {name} with params: {params}")
    for fold_id, (tr_idx, va_idx) in enumerate(fold_indices):
        X_tr, y_tr = X_np[tr_idx], y[tr_idx]
        X_va, y_va = X_np[va_idx], y[va_idx]
        w_tr = make_sample_weights(y_tr)
        w_va = make_sample_weights(y_va)

        model = xgb.XGBClassifier(**params)
        logging.info(f"[{name}][fold {fold_id}] fit start.")
        t0 = time.time()
        model.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            eval_set=[(X_va, y_va)],
            sample_weight_eval_set=[w_va],
            verbose=True,
        )
        dt = time.time() - t0
        logging.info(f"[{name}][fold {fold_id}] fit done in {dt:.2f}s. best_iteration={model.best_iteration}")

        va_proba = model.predict_proba(X_va)
        va_logits = np.log(np.clip(va_proba, eps, 1.0)).astype(np.float32)
        oof_logits[va_idx, xi*7:(xi+1)*7] = va_logits

        va_pred = va_proba.argmax(axis=1)
        va_acc = accuracy_score(y_va, va_pred)
        oof_pred_tracker[name][fold_id] = va_acc
        logging.info(f"[{name}][fold {fold_id}] valid accuracy: {va_acc:.6f}")

        test_proba = model.predict_proba(X_test_np)
        test_logits = np.log(np.clip(test_proba, eps, 1.0)).astype(np.float32)
        test_logits_per_model[name].append(test_logits)
        logging.info(f"[{name}][fold {fold_id}] test logits collected.")

# -----------------------------
# CatBoost (CUDA) — no class weights (per advice)
# -----------------------------
cat_name = config["cat_setup"]["name"]
cat_params = config["cat_setup"]["params"].copy()
logging.info(f"Training model: {cat_name} with params: {cat_params}")
for fold_id, (tr_idx, va_idx) in enumerate(fold_indices):
    X_tr, y_tr = X_np[tr_idx], y[tr_idx]
    X_va, y_va = X_np[va_idx], y[va_idx]
    train_pool = Pool(X_tr, y_tr)
    valid_pool = Pool(X_va, y_va)

    model = CatBoostClassifier(**cat_params)
    logging.info(f"[{cat_name}][fold {fold_id}] fit start.")
    t0 = time.time()
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    dt = time.time() - t0
    logging.info(f"[{cat_name}][fold {fold_id}] fit done in {dt:.2f}s.")

    va_proba = model.predict_proba(valid_pool)
    va_logits = np.log(np.clip(va_proba, eps, 1.0)).astype(np.float32)
    oof_logits[va_idx, (len(config["xgb_setups"]))*7:(len(config["xgb_setups"])+1)*7] = va_logits

    va_pred = va_proba.argmax(axis=1)
    va_acc = accuracy_score(y_va, va_pred)
    oof_pred_tracker[cat_name][fold_id] = va_acc
    logging.info(f"[{cat_name}][fold {fold_id}] valid accuracy: {va_acc:.6f}")

    test_proba = model.predict_proba(X_test_np)
    test_logits = np.log(np.clip(test_proba, eps, 1.0)).astype(np.float32)
    test_logits_per_model[cat_name].append(test_logits)
    logging.info(f"[{cat_name}][fold {fold_id}] test logits collected.")

# -----------------------------
# LightGBM (CUDA)
# -----------------------------
lgb_name = config["lgb_setup"]["name"]
lgb_params = config["lgb_setup"]["params"].copy()
lgb_es = config["lgb_setup"]["early_stopping_rounds"]
logging.info(f"Training model: {lgb_name} with params: {lgb_params}")
for fold_id, (tr_idx, va_idx) in enumerate(fold_indices):
    X_tr, y_tr = X_np[tr_idx], y[tr_idx]
    X_va, y_va = X_np[va_idx], y[va_idx]

    model = lgb.LGBMClassifier(**lgb_params)
    logging.info(f"[{lgb_name}][fold {fold_id}] fit start.")
    t0 = time.time()
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="multi_logloss",
        early_stopping_rounds=lgb_es,
        verbose=True,
    )
    dt = time.time() - t0
    logging.info(f"[{lgb_name}][fold {fold_id}] fit done in {dt:.2f}s. best_iteration_={model.best_iteration_}")

    va_proba = model.predict_proba(X_va, num_iteration=model.best_iteration_)
    va_logits = np.log(np.clip(va_proba, eps, 1.0)).astype(np.float32)
    oof_logits[va_idx, (len(config["xgb_setups"])+1)*7:(len(config["xgb_setups"])+2)*7] = va_logits

    va_pred = va_proba.argmax(axis=1)
    va_acc = accuracy_score(y_va, va_pred)
    oof_pred_tracker[lgb_name][fold_id] = va_acc
    logging.info(f"[{lgb_name}][fold {fold_id}] valid accuracy: {va_acc:.6f}")

    test_proba = model.predict_proba(X_test_np, num_iteration=model.best_iteration_)
    test_logits = np.log(np.clip(test_proba, eps, 1.0)).astype(np.float32)
    test_logits_per_model[lgb_name].append(test_logits)
    logging.info(f"[{lgb_name}][fold {fold_id}] test logits collected.")

# -----------------------------
# Tabular NN (CUDA, AMP fp16)
# -----------------------------
device = torch.device("cuda:0")
logging.info(f"Tabular NN will train on device: {device}")

def compute_standardizer(X_fit: np.ndarray):
    mean = X_fit.mean(axis=0, dtype=np.float64)
    std = X_fit.std(axis=0, dtype=np.float64) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)

class NpDataset(Dataset):
    def __init__(self, X, y=None, mean=None, std=None):
        self.X = X
        self.y = y
        self.mean = mean
        self.std = std

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = (self.X[idx] - self.mean) / self.std
        if self.y is None:
            return x.astype(np.float32)
        return x.astype(np.float32), np.int64(self.y[idx])

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_sizes, out_dim, dropout=0.2):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden_sizes:
            layers += [
                nn.Linear(last, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def cosine_lr(optimizer, step, total_steps, base_lr, warmup):
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))

def train_nn_fold(X_tr, y_tr, X_va, y_va, X_te, mean, std, in_dim, out_dim, nn_cfg):
    model = MLP(in_dim, nn_cfg["hidden_sizes"], out_dim, dropout=nn_cfg["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=nn_cfg["lr"], weight_decay=nn_cfg["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    train_ds = NpDataset(X_tr, y_tr, mean, std)
    valid_ds = NpDataset(X_va, y_va, mean, std)
    test_ds  = NpDataset(X_te, None, mean, std)

    train_loader = DataLoader(train_ds, batch_size=nn_cfg["batch_size"], shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=nn_cfg["batch_size"], shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=nn_cfg["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

    total_steps = nn_cfg["epochs"] * len(train_loader)
    step = 0

    for epoch in range(nn_cfg["epochs"]):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            if config["use_mixup"]:
                lam = np.random.beta(config["mixup_alpha"], config["mixup_alpha"])
                perm = torch.randperm(xb.size(0), device=device)
                x1, x2 = xb, xb[perm]
                y1, y2 = yb, yb[perm]
                xb_mixed = lam * x1 + (1 - lam) * x2
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    logits = model(xb_mixed)
                    loss1 = F.cross_entropy(logits, y1, label_smoothing=nn_cfg["label_smoothing"])
                    loss2 = F.cross_entropy(logits, y2, label_smoothing=nn_cfg["label_smoothing"])
                    loss = lam * loss1 + (1 - lam) * loss2
            else:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    logits = model(xb)
                    loss = F.cross_entropy(logits, yb, label_smoothing=nn_cfg["label_smoothing"])

            lr = cosine_lr(optimizer, step, total_steps, nn_cfg["lr"], nn_cfg["warmup_steps"])
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
            step += 1

        avg_loss = running_loss / max(1, len(train_loader))
        logging.info(f"[NN] epoch {epoch+1}/{nn_cfg['epochs']} | train loss {avg_loss:.6f}")

    # Valid logits
    model.eval()
    va_logits_list = []
    with torch.no_grad():
        for xb, yb in valid_loader:
            xb = xb.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = model(xb)
            va_logits_list.append(logits.float().cpu().numpy())
    va_logits = np.concatenate(va_logits_list, axis=0)

    # Test logits
    te_logits_list = []
    with torch.no_grad():
        for xb in test_loader:
            xb = xb.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = model(xb)
            te_logits_list.append(logits.float().cpu().numpy())
    te_logits = np.concatenate(te_logits_list, axis=0)

    return va_logits.astype(np.float32), te_logits.astype(np.float32)

nn_name = config["nn"]["name"]
in_dim = X_np.shape[1]
out_dim = 7
for fold_id, (tr_idx, va_idx) in enumerate(fold_indices):
    X_tr, y_tr = X_np[tr_idx], y[tr_idx]
    X_va, y_va = X_np[va_idx], y[va_idx]
    mean, std = compute_standardizer(X_tr)
    logging.info(f"[{nn_name}][fold {fold_id}] mean/std computed. Training NN...")
    t0 = time.time()
    va_logits_nn, te_logits_nn = train_nn_fold(X_tr, y_tr, X_va, y_va, X_test_np, mean, std, in_dim, out_dim, config["nn"])
    dt = time.time() - t0
    logging.info(f"[{nn_name}][fold {fold_id}] training done in {dt:.2f}s.")

    # Store OOF logits for NN directly (logits)
    start_col = (len(config["xgb_setups"])+2) * 7
    oof_logits[va_idx, start_col:start_col+7] = va_logits_nn

    va_pred = va_logits_nn.argmax(axis=1)
    va_acc = accuracy_score(y[va_idx], va_pred)
    oof_pred_tracker[nn_name][fold_id] = va_acc
    logging.info(f"[{nn_name}][fold {fold_id}] valid accuracy: {va_acc:.6f}")

    test_logits_per_model[nn_name].append(te_logits_nn)
    logging.info(f"[{nn_name}][fold {fold_id}] test logits collected.")

# -----------------------------
# Aggregate per-model test logits across folds (mean)
# -----------------------------
model_test_logits = []
for name in model_names:
    arr = np.stack(test_logits_per_model[name], axis=0)  # [n_folds, n_test, 7]
    mean_logits = arr.mean(axis=0)                       # [n_test, 7]
    model_test_logits.append(mean_logits.astype(np.float32))
    logging.info(f"[{name}] test logits averaged across {arr.shape[0]} folds.")

# -----------------------------
# Build stacked features (OOF and Test)
# -----------------------------
oof_features = oof_logits.reshape(len(X_np), num_models, 7)   # [N, M, C]
test_features = np.stack(model_test_logits, axis=1)           # [n_test, M, C]
logging.info(f"Stacked shapes | OOF: {oof_features.shape} | Test: {test_features.shape}")

# Keep only rows that received OOF predictions (exclude the single class-5 row)
filled_mask = (np.abs(oof_features).sum(axis=(1, 2)) > 0)
num_filled = int(filled_mask.sum())
logging.info(f"OOF filled rows: {num_filled}/{len(oof_features)} (excluded rows will be ignored in blender training).")

oof_features_filled = oof_features[filled_mask]
y_filled = y[filled_mask]

# -----------------------------
# Blender: constrained logit blending (non-negative weights per model/class)
# z_c = sum_m softplus(w_{m,c}) * logit_{m,c}
# -----------------------------
class LogitBlender(nn.Module):
    def __init__(self, n_models, n_classes):
        super().__init__()
        self.raw_w = nn.Parameter(torch.zeros(n_models, n_classes))  # initialized to 0 -> softplus ~ 0.693

    def forward(self, logits_stack):  # logits_stack: [B, M, C]
        W = F.softplus(self.raw_w)    # [M, C] >= 0
        z = (logits_stack * W.unsqueeze(0)).sum(dim=1)  # [B, C]
        return z

device = torch.device("cuda:0")
blender = LogitBlender(n_models=num_models, n_classes=7).to(device)
optimizer = torch.optim.AdamW(blender.parameters(), lr=0.1, weight_decay=0.0)
scaler = torch.cuda.amp.GradScaler(enabled=True)

def build_blender_loaders(oof_feats, y_true, test_feats, batch=65536):
    ds_train = torch.utils.data.TensorDataset(
        torch.from_numpy(oof_feats.astype(np.float32)),
        torch.from_numpy(y_true.astype(np.int64)),
    )
    dl_train = DataLoader(ds_train, batch_size=batch, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    ds_all = torch.from_numpy(oof_feats.astype(np.float32))
    ds_test = torch.from_numpy(test_feats.astype(np.float32))

    dl_all = DataLoader(ds_all, batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)
    dl_test = DataLoader(ds_test, batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)
    return dl_train, dl_all, dl_test

dl_train, dl_all, dl_test = build_blender_loaders(oof_features_filled, y_filled, test_features, batch=65536)

blender_epochs = 5
logging.info(f"Training blender for {blender_epochs} epochs on GPU with AMP.")
for epoch in range(blender_epochs):
    blender.train()
    running = 0.0
    for xb, yb in dl_train:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            logits = blender(xb)
            loss = F.cross_entropy(logits, yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        running += loss.item()
    avg = running / max(1, len(dl_train))
    logging.info(f"[Blender] epoch {epoch+1}/{blender_epochs} | loss={avg:.6f}")

# OOF evaluation with blender (on filled rows only)
blender.eval()
oof_logits_blend = []
with torch.no_grad():
    for xb in dl_all:
        xb = xb.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            z = blender(xb)
        oof_logits_blend.append(z.float().cpu().numpy())
oof_logits_blend = np.concatenate(oof_logits_blend, axis=0)
oof_pred_blend = oof_logits_blend.argmax(axis=1)
oof_acc = accuracy_score(y_filled, oof_pred_blend)
logging.info(f"Blender OOF accuracy (filled rows only): {oof_acc:.6f}")

report = classification_report(y_filled, oof_pred_blend, digits=6)
logging.info("Final OOF classification report (blender):")
for line in report.splitlines():
    logging.info(line)

# -----------------------------
# Test inference with blender
# -----------------------------
test_logits_blend = []
with torch.no_grad():
    for xb in dl_test:
        xb = xb.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            z = blender(xb)
        test_logits_blend.append(z.float().cpu().numpy())
test_logits_blend = np.concatenate(test_logits_blend, axis=0)
test_pred_encoded = test_logits_blend.argmax(axis=1)

# Map back to original labels using LabelEncoder inverse_transform
pred_labels = le.inverse_transform(test_pred_encoded)

# -----------------------------
# Save submission
# -----------------------------
submission = pd.DataFrame({"Id": test_ids, "Cover_Type": pred_labels.astype(int)})
submission.sort_values("Id", inplace=True)
submission.to_csv(SUBMISSION_FILE, index=False)
logging.info(f"Submission saved to: {SUBMISSION_FILE}")

# -----------------------------
# Log per-model fold accuracies and final summary
# -----------------------------
for name in model_names:
    accs = [oof_pred_tracker[name][k] for k in sorted(oof_pred_tracker[name].keys())]
    logging.info(f"[{name}] per-fold valid accuracy: {', '.join([f'{a:.6f}' for a in accs])} | mean={np.mean(accs):.6f}")

logging.info(f"FINAL SUMMARY: num_models={num_models} | folds={config['n_folds']} | OOF blender accuracy={oof_acc:.6f}")
logging.info("Pipeline complete. All final validation results have been logged.")