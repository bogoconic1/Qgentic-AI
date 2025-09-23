import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import logging
import time
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import xgboost as xgb
import lightgbm as lgb
from lightgbm import early_stopping as lgb_early_stopping, log_evaluation as lgb_log_evaluation, record_evaluation as lgb_record_evaluation
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold


# ------------------------ Logging and Utils ------------------------ #

def setup_logging(log_path: str):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler()
        ],
    )
    logging.info(f"Logging initialized. Writing to {log_path}")


def seed_everything(seed: int):
    logging.info(f"Setting global random seed to {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------ Column Detection and Cleaning ------------------------ #

def detect_columns(df: pd.DataFrame):
    soil_cols = [c for c in df.columns if re.match(r"^Soil_Type\d+$", c)]
    wild_cols = [c for c in df.columns if re.match(r"^Wilderness_Area\d+$", c)]
    exclude = set(['Id', 'Cover_Type'] + soil_cols + wild_cols)
    base_num_cols = [c for c in df.columns if c not in exclude]
    logging.info(f"Detected {len(soil_cols)} soil type columns, {len(wild_cols)} wilderness columns, and {len(base_num_cols)} other numeric columns.")
    return base_num_cols, wild_cols, soil_cols


def clean_one_hot_group(df: pd.DataFrame, group_cols: list, clean_name: str, unknown_index: int = 0, prefer='first'):
    logging.info(f"Cleaning one-hot group {clean_name} from columns: {group_cols} with prefer='{prefer}'")
    arr = df[group_cols].values.astype(np.int16)
    sums = arr.sum(axis=1)
    if prefer == 'first':
        chosen = arr.argmax(axis=1) + 1  # 1..n
    elif prefer == 'last':
        chosen = (arr.shape[1] - np.flip(arr, axis=1).argmax(axis=1))  # 1..n
    else:
        chosen = arr.argmax(axis=1) + 1
    clean = np.where(sums == 0, unknown_index, chosen)  # 0 for unknown
    reliable = (sums == 1).astype(np.int8)
    df[f"{clean_name}_Clean"] = clean.astype(np.int16)
    df[f"{clean_name}_Reliable"] = reliable
    logging.info(f"{clean_name}_Clean summary: unknown (sum=0) count={int((sums==0).sum())}, multi-hot (sum>1) count={int((sums>1).sum())}, proper one-hot (sum=1) count={int((sums==1).sum())}")
    return df


# ------------------------ Feature Engineering ------------------------ #

def engineer_distance_features(df: pd.DataFrame):
    logging.info("Engineering distance features (Hydrology_Dist, Fire_Road_Dist)")
    df["Hydrology_Dist"] = np.sqrt(df["Horizontal_Distance_To_Hydrology"].values**2 +
                                   df["Vertical_Distance_To_Hydrology"].values**2)
    df["Fire_Road_Dist"] = df["Horizontal_Distance_To_Fire_Points"].values + df["Horizontal_Distance_To_Roadways"].values
    return df


def engineer_binned_features(df: pd.DataFrame, elevation_band_size: int = 100):
    logging.info(f"Engineering binned features (Elevation_Band) with band size {elevation_band_size}")
    df["Elevation_Band"] = (df["Elevation"].values // elevation_band_size).astype(np.int16)
    return df


def engineer_group_diagnostics(df: pd.DataFrame, soil_cols: list, wild_cols: list):
    logging.info("Engineering diagnostic features for Soil_Type and Wilderness groups")
    soil_arr = df[soil_cols].values.astype(np.int16)
    wild_arr = df[wild_cols].values.astype(np.int16)

    soil_sum = soil_arr.sum(axis=1)
    wild_sum = wild_arr.sum(axis=1)
    df["Soil_Sum"] = soil_sum
    df["Soil_IsUnknown"] = (soil_sum == 0).astype(np.uint8)
    df["Soil_IsMultihot"] = (soil_sum > 1).astype(np.uint8)
    df["Soil_Argmax"] = np.where(soil_sum == 0, 0, soil_arr.argmax(axis=1) + 1).astype(np.int16)

    df["Wilder_Sum"] = wild_sum
    df["Wilder_IsUnknown"] = (wild_sum == 0).astype(np.uint8)
    df["Wilder_IsMultihot"] = (wild_sum > 1).astype(np.uint8)
    df["Wilder_Argmax"] = np.where(wild_sum == 0, 0, wild_arr.argmax(axis=1) + 1).astype(np.int16)
    return df


def engineer_interactions_original_onehots(df_tr: pd.DataFrame, df_te: pd.DataFrame, soil_cols: list, wild_cols: list):
    logging.info("Engineering interaction features from original one-hot groups: Soil_Type_i x Wilderness_Area_j")
    inter_cols_tr = {}
    inter_cols_te = {}
    for s in soil_cols:
        s_tr = df_tr[s].astype(np.uint8)
        s_te = df_te[s].astype(np.uint8)
        for w in wild_cols:
            name = f"INT_{s}_x_{w}"
            inter_cols_tr[name] = (s_tr & df_tr[w].astype(np.uint8)).astype(np.uint8)
            inter_cols_te[name] = (s_te & df_te[w].astype(np.uint8)).astype(np.uint8)
    inter_tr = pd.DataFrame(inter_cols_tr, index=df_tr.index)
    inter_te = pd.DataFrame(inter_cols_te, index=df_te.index)
    logging.info(f"Created {inter_tr.shape[1]} interaction columns")
    return inter_tr, inter_te


def drop_duplicate_columns(df: pd.DataFrame):
    if not df.columns.is_unique:
        dup = df.columns[df.columns.duplicated()].tolist()
        logging.info(f"Dropping duplicate columns: first 20={dup[:20]} ... total={len(dup)}")
        df = df.loc[:, ~df.columns.duplicated()].copy()
    return df


# ------------------------ Shift and MI Checks (GPU) ------------------------ #

def ks_test_torch_gpu(a: np.ndarray, b: np.ndarray, max_samples: int = 500000, device: str = "cuda"):
    na = min(len(a), max_samples)
    nb = min(len(b), max_samples)
    logging.info(f"KS test on GPU with samples a={na}, b={nb}")
    rng = np.random.RandomState(42)
    a_s = a if len(a) <= max_samples else rng.choice(a, size=na, replace=False)
    b_s = b if len(b) <= max_samples else rng.choice(b, size=nb, replace=False)

    x = torch.from_numpy(np.concatenate([a_s, b_s]).astype(np.float32)).to(device)
    flags = torch.from_numpy(np.concatenate([np.ones(len(a_s), dtype=np.int8), np.zeros(len(b_s), dtype=np.int8)])).to(device)

    idx = torch.argsort(x)
    flags_sorted = flags[idx]

    ones_a = (flags_sorted == 1).to(torch.float32)
    ones_b = 1.0 - ones_a

    cdf_a = torch.cumsum(ones_a, dim=0) / float(len(a_s))
    cdf_b = torch.cumsum(ones_b, dim=0) / float(len(b_s))

    d_stat = torch.max(torch.abs(cdf_a - cdf_b)).item()

    n1 = float(len(a_s))
    n2 = float(len(b_s))
    ne = np.sqrt(n1 * n2 / (n1 + n2))
    lam = (ne + 0.12 + 0.11 / ne) * d_stat

    k = torch.arange(1, 1001, device=device, dtype=torch.float64)
    terms = 2.0 * torch.exp(-2.0 * (k ** 2) * (lam ** 2))
    alt = torch.ones_like(terms)
    alt[1::2] *= -1
    series = (alt * terms).cumsum(dim=0)
    pval = float(series[-1].clamp(min=0.0, max=1.0).item())
    return d_stat, pval


def chi2_pvalue_gpu(train_counts: np.ndarray, test_counts: np.ndarray, device: str = "cuda"):
    obs = np.stack([train_counts, test_counts], axis=0).astype(np.float64)
    obs_t = torch.from_numpy(obs).to(device)
    row_sum = obs_t.sum(dim=1, keepdim=True)
    col_sum = obs_t.sum(dim=0, keepdim=True)
    total = obs_t.sum()
    expected = row_sum @ (col_sum / total)
    chi2 = torch.sum((obs_t - expected) ** 2 / (expected + 1e-12))
    dof = obs_t.shape[1] - 1
    s = 0.5 * dof
    x = 0.5 * chi2
    P = torch.special.gammainc(torch.tensor(s, device=device, dtype=torch.float64),
                               x.to(torch.float64))
    pval = float((1.0 - P).clamp(min=0.0, max=1.0).item())
    return float(chi2.item()), pval


def distribution_shift_checks(train_df: pd.DataFrame, test_df: pd.DataFrame, cont_cols: list, cat_specs: dict, device: str = "cuda"):
    logging.info("Running distribution shift checks (KS for continuous, Chi-square for categorical)")
    ks_results = []
    for c in cont_cols:
        d, p = ks_test_torch_gpu(train_df[c].values, test_df[c].values, device=device)
        ks_results.append((c, d, p))
    ks_flagged = [(c, d, p) for c, d, p in ks_results if p < 0.01]
    logging.info(f"KS test flagged {len(ks_flagged)} continuous features with p<0.01")
    for c, d, p in sorted(ks_flagged, key=lambda x: x[1], reverse=True)[:20]:
        logging.info(f"KS shift: {c}: D={d:.6f}, p={p:.4e}")

    chi2_results = []
    for (name, vals_train, vals_test) in cat_specs["datasets"]:
        cats = np.unique(np.concatenate([vals_train, vals_test]))
        tr_cnt = np.array([(vals_train == k).sum() for k in cats], dtype=np.int64)
        te_cnt = np.array([(vals_test == k).sum() for k in cats], dtype=np.int64)
        chi2, pval = chi2_pvalue_gpu(tr_cnt, te_cnt, device=device)
        chi2_results.append((name, chi2, pval))
    chi2_flagged = [(n, s, p) for (n, s, p) in chi2_results if p < 0.01]
    logging.info(f"Chi-square test flagged {len(chi2_flagged)} categorical features with p<0.01")
    for n, s, p in sorted(chi2_flagged, key=lambda x: x[1], reverse=True)[:20]:
        logging.info(f"Chi2 shift: {n}: chi2={s:.3f}, p={p:.4e}")


def discretize_quantile_gpu(x_np: np.ndarray, n_bins: int = 20, device: str = "cuda"):
    logging.info(f"Discretizing to quantile bins on GPU: n={x_np.shape[0]}, bins={n_bins}")
    x = torch.from_numpy(x_np.astype(np.float32)).to(device)
    qs = torch.linspace(0.0, 1.0, n_bins + 1, device=device)
    edges = torch.quantile(x, qs)
    eps = 1e-6
    edges = torch.maximum(edges, (edges - eps).cummax(dim=0).values)
    bins = torch.bucketize(x, edges[1:-1], right=False)
    return bins.to(torch.int64)


def mutual_information_gpu(x_np: np.ndarray, y_np: np.ndarray, bins: int = 20, is_categorical: bool = False, device: str = "cuda"):
    logging.info(f"Computing MI on GPU (is_categorical={is_categorical}, bins={bins}) for {len(x_np)} samples")
    y = torch.from_numpy(y_np.astype(np.int64)).to(device)
    y = y - y.min()
    ny = int(y.max().item() + 1)

    if is_categorical:
        x_codes = torch.from_numpy(x_np.astype(np.int64)).to(device)
        x_codes = x_codes - x_codes.min()
        nx = int(x_codes.max().item() + 1)
    else:
        x_codes = discretize_quantile_gpu(x_np, n_bins=bins, device=device)
        nx = bins

    idx = x_codes + nx * y
    counts = torch.bincount(idx, minlength=nx * ny).to(torch.float64)
    pxy = counts.view(ny, nx)
    N = float(len(y_np))
    pxy = pxy / N
    py = pxy.sum(dim=1, keepdim=True)
    px = pxy.sum(dim=0, keepdim=True)
    den = (py @ px).to(torch.float64)
    mask = pxy > 0
    num = pxy.to(torch.float64)
    mi = torch.sum(num[mask] * (torch.log(num[mask]) - torch.log(den[mask] + 1e-16))).item()
    logging.info(f"Computed MI={mi:.6f}")
    return mi


def leakage_checks_mi(train_df: pd.DataFrame, target: np.ndarray, cont_cols: list, cat_specs: dict, device: str = "cuda"):
    logging.info("Computing mutual information for leakage checks")
    y0 = target.astype(np.int64) - 1  # 0..6
    mi_results = []
    for c in cont_cols:
        mi = mutual_information_gpu(train_df[c].values, y0, bins=20, is_categorical=False, device=device)
        mi_results.append((c, mi))
    for (name, vals_train, _) in cat_specs["datasets"]:
        if name in ("Soil_Type_Clean", "Wilderness_Clean"):
            mi = mutual_information_gpu(vals_train, y0, bins=50, is_categorical=True, device=device)
            mi_results.append((name, mi))
    mi_results_sorted = sorted(mi_results, key=lambda x: x[1], reverse=True)
    logging.info("Top 15 features by MI with target:")
    for n, v in mi_results_sorted[:15]:
        logging.info(f"MI: {n} -> {v:.6f}")
    high_mi = [(n, v) for n, v in mi_results_sorted if v > 0.95]
    if len(high_mi) > 0:
        logging.info("Potential leakage detected (MI > 0.95):")
        for n, v in high_mi:
            logging.info(f"  {n} -> {v:.6f}")
    else:
        logging.info("No features exceed MI > 0.95; leakage unlikely.")


# ------------------------ Feature Matrix Builder ------------------------ #

def build_feature_matrix(train_df: pd.DataFrame, test_df: pd.DataFrame,
                         use_interactions: bool = True,
                         include_original_soil: bool = True,
                         include_original_wild: bool = True,
                         elevation_band_size: int = 100):
    logging.info("Building feature matrices (X_train, X_test)")
    base_num_cols, wild_cols, soil_cols = detect_columns(train_df)

    # Engineering numeric features
    train_df = engineer_distance_features(train_df)
    test_df = engineer_distance_features(test_df)

    train_df = engineer_binned_features(train_df, elevation_band_size=elevation_band_size)
    test_df = engineer_binned_features(test_df, elevation_band_size=elevation_band_size)

    train_df = engineer_group_diagnostics(train_df, soil_cols, wild_cols)
    test_df = engineer_group_diagnostics(test_df, soil_cols, wild_cols)

    # Interaction features from original one-hots
    if use_interactions:
        inter_tr, inter_te = engineer_interactions_original_onehots(train_df, test_df, soil_cols, wild_cols)
    else:
        inter_tr = pd.DataFrame(index=train_df.index)
        inter_te = pd.DataFrame(index=test_df.index)

    engineered_numeric = ["Hydrology_Dist", "Fire_Road_Dist", "Elevation_Band",
                          "Soil_Sum", "Soil_IsUnknown", "Soil_IsMultihot", "Soil_Argmax",
                          "Wilder_Sum", "Wilder_IsUnknown", "Wilder_IsMultihot", "Wilder_Argmax"]

    hillshade_cols = ["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]
    numeric_cols = sorted(set(base_num_cols + engineered_numeric + hillshade_cols))

    X_train_list = [train_df[numeric_cols].astype(np.float32)]
    X_test_list = [test_df[numeric_cols].astype(np.float32)]

    if include_original_soil:
        X_train_list.append(train_df[soil_cols].astype(np.uint8))
        X_test_list.append(test_df[soil_cols].astype(np.uint8))

    if include_original_wild:
        X_train_list.append(train_df[wild_cols].astype(np.uint8))
        X_test_list.append(test_df[wild_cols].astype(np.uint8))

    if use_interactions:
        X_train_list.append(inter_tr.astype(np.uint8))
        X_test_list.append(inter_te.astype(np.uint8))

    X_train = pd.concat(X_train_list, axis=1)
    X_test = pd.concat(X_test_list, axis=1)

    X_train = drop_duplicate_columns(X_train)
    X_test = drop_duplicate_columns(X_test)

    all_cols = sorted(set(X_train.columns).union(set(X_test.columns)))
    X_train = X_train.reindex(columns=all_cols, fill_value=0)
    X_test = X_test.reindex(columns=all_cols, fill_value=0)

    logging.info(f"Final feature matrix shapes: X_train={X_train.shape}, X_test={X_test.shape}")
    return X_train, X_test


# ------------------------ Metrics ------------------------ #

def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray):
    classes = np.unique(y_true)
    stats = {}
    for c in classes:
        mask = (y_true == c)
        acc = float((y_pred[mask] == c).mean()) if mask.sum() > 0 else 0.0
        stats[int(c)] = acc
    return stats


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray):
    return float((y_true == y_pred).mean())


# ------------------------ Training and Ensembling ------------------------ #

def train_xgb_gpu(X_tr, y_tr, X_va, y_va, num_class: int, params: dict, num_boost_round: int, early_stopping_rounds: int):
    logging.info("Training XGBoost (GPU)")
    dtr = xgb.DMatrix(X_tr, label=y_tr)
    dva = xgb.DMatrix(X_va, label=y_va)
    evals = [(dtr, "train"), (dva, "valid")]
    booster = xgb.train(
        params=params,
        dtrain=dtr,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=100
    )
    logging.info(f"XGBoost best_iteration={booster.best_iteration}, best_score={booster.best_score}")
    return booster


def predict_xgb(booster, X):
    d = xgb.DMatrix(X)
    return booster.predict(d)


def train_lgb_gpu(X_tr, y_tr, X_va, y_va, num_class: int, params: dict, num_boost_round: int, early_stopping_rounds: int):
    logging.info("Training LightGBM (GPU)")
    lgb_tr = lgb.Dataset(X_tr, label=y_tr)
    lgb_va = lgb.Dataset(X_va, label=y_va, reference=lgb_tr)
    evals_result = {}
    model = lgb.train(
        params=params,
        train_set=lgb_tr,
        num_boost_round=num_boost_round,
        valid_sets=[lgb_tr, lgb_va],
        valid_names=["train", "valid"],
        callbacks=[
            lgb_early_stopping(stopping_rounds=early_stopping_rounds, first_metric_only=True),
            lgb_log_evaluation(period=100),
            lgb_record_evaluation(evals_result),
        ]
    )
    # best score logging
    if "valid" in model.best_score and "multi_logloss" in model.best_score["valid"]:
        best_score = model.best_score["valid"]["multi_logloss"]
    else:
        best_score = None
    logging.info(f"LightGBM best_iteration={model.best_iteration}, best_score={best_score}")
    return model


def predict_lgb(model, X):
    return model.predict(X, num_iteration=model.best_iteration)


def train_cat_gpu(X_tr, y_tr, X_va, y_va, params: dict):
    logging.info("Training CatBoost (GPU)")
    train_pool = Pool(X_tr, label=y_tr)
    valid_pool = Pool(X_va, label=y_va)
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=valid_pool, verbose=100)
    logging.info(f"CatBoost best_iteration={model.get_best_iteration()}, best_score={model.get_best_score()}")
    return model


def predict_cat(model, X):
    return model.predict_proba(X)


# ------------------------ Main Pipeline ------------------------ #

def main():
    BASE_DIR = "task/tabular-playground-series-dec-2021"
    OUT_DIR = f"{BASE_DIR}/outputs/1"
    LOG_FILE = f"{OUT_DIR}/code_1_v6.txt"
    SUBMISSION_FILE = f"{OUT_DIR}/submission_6.csv"

    setup_logging(LOG_FILE)
    seed_everything(42)

    cfg = {
        "device": "cuda",
        "use_interactions": True,
        "include_original_soil": True,
        "include_original_wild": True,
        "elevation_band_size": 100,

        # CV
        "n_splits": 3,  # reduced to keep runtime reasonable while following K-fold advice
        "shuffle": True,
        "random_state": 42,

        # XGBoost params (updated for xgboost>=2.0 GPU API)
        "xgb_params": {
            "objective": "multi:softprob",
            "num_class": 7,
            "tree_method": "hist",
            "device": "cuda",
            "eval_metric": "mlogloss",
            "max_depth": 7,
            "eta": 0.06,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 16.0,
            "max_delta_step": 1,
            "lambda": 1.0,
            "alpha": 0.0,
            "verbosity": 1,
            "seed": 42,
        },
        "xgb_num_boost_round": 2500,
        "xgb_es_rounds": 200,

        # LightGBM params (GPU + callbacks for early stopping)
        "lgb_params": {
            "objective": "multiclass",
            "num_class": 7,
            "metric": "multi_logloss",
            "learning_rate": 0.05,
            "num_leaves": 256,
            "min_data_in_leaf": 1500,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "max_bin": 255,
            "device": "gpu",
            "gpu_platform_id": 0,
            "gpu_device_id": 0,
            "verbosity": -1,
            "seed": 42,
        },
        "lgb_num_boost_round": 6000,
        "lgb_es_rounds": 200,

        # CatBoost params (GPU)
        "cat_params": {
            "loss_function": "MultiClass",
            "iterations": 6000,
            "learning_rate": 0.05,
            "depth": 8,
            "l2_leaf_reg": 6.0,
            "bootstrap_type": "Bernoulli",
            "subsample": 0.8,
            "task_type": "GPU",
            "devices": "0",
            "allow_writing_files": False,
            "random_seed": 42,
            "use_best_model": True,
            "od_type": "Iter",
            "od_wait": 200,
        },
    }

    # Load data
    train_path = f"{BASE_DIR}/train.csv"
    test_path = f"{BASE_DIR}/test.csv"
    sample_sub_path = f"{BASE_DIR}/sample_submission.csv"

    logging.info(f"Loading training data from {train_path}")
    train_df = pd.read_csv(train_path)
    logging.info(f"Loading test data from {test_path}")
    test_df = pd.read_csv(test_path)
    sample_sub = pd.read_csv(sample_sub_path)
    logging.info(f"Loaded train shape={train_df.shape}, test shape={test_df.shape}")

    # Detect groups
    base_num_cols, wild_cols, soil_cols = detect_columns(train_df)

    # Clean (for diagnostics; we will keep the original one-hots as features)
    train_df = clean_one_hot_group(train_df, soil_cols, "Soil_Type", unknown_index=0, prefer='first')
    test_df = clean_one_hot_group(test_df, soil_cols, "Soil_Type", unknown_index=0, prefer='first')
    train_df = clean_one_hot_group(train_df, wild_cols, "Wilderness", unknown_index=0, prefer='first')
    test_df = clean_one_hot_group(test_df, wild_cols, "Wilderness", unknown_index=0, prefer='first')

    # Distribution shift and leakage diagnostics
    cont_cols = [c for c in base_num_cols if c not in ["Id", "Cover_Type"]] + ["Elevation"]
    cat_specs = {
        "datasets": [
            ("Wilderness_Clean", train_df["Wilderness_Clean"].values.astype(np.int16), test_df["Wilderness_Clean"].values.astype(np.int16)),
            ("Soil_Type_Clean", train_df["Soil_Type_Clean"].values.astype(np.int16), test_df["Soil_Type_Clean"].values.astype(np.int16)),
        ]
    }
    distribution_shift_checks(train_df, test_df, cont_cols, cat_specs, device=cfg["device"])
    leakage_checks_mi(train_df, train_df["Cover_Type"].values, cont_cols, cat_specs, device=cfg["device"])

    # Build features (includes engineered diagnostics, distances, elevation band, original one-hots, and interactions)
    X_train, X_test = build_feature_matrix(
        train_df, test_df,
        use_interactions=cfg["use_interactions"],
        include_original_soil=cfg["include_original_soil"],
        include_original_wild=cfg["include_original_wild"],
        elevation_band_size=cfg["elevation_band_size"]
    )

    # Prepare labels (0..6)
    y = train_df["Cover_Type"].astype(np.int64).values
    y0 = y - 1

    # CV setup
    skf = StratifiedKFold(n_splits=cfg["n_splits"], shuffle=cfg["shuffle"], random_state=cfg["random_state"])
    num_class = 7

    # Storage for OOF predictions and test preds
    oof_proba_xgb = np.zeros((len(X_train), num_class), dtype=np.float32)
    oof_proba_lgb = np.zeros((len(X_train), num_class), dtype=np.float32)
    oof_proba_cat = np.zeros((len(X_train), num_class), dtype=np.float32)

    test_proba_xgb = np.zeros((len(X_test), num_class), dtype=np.float32)
    test_proba_lgb = np.zeros((len(X_test), num_class), dtype=np.float32)
    test_proba_cat = np.zeros((len(X_test), num_class), dtype=np.float32)

    fold_logs = []

    # Train per-fold models
    fold_idx = 0
    for tr_idx, va_idx in skf.split(X_train, y0):
        fold_idx += 1
        logging.info(f"Starting fold {fold_idx}/{cfg['n_splits']}")

        X_tr = X_train.iloc[tr_idx]
        y_tr = y0[tr_idx]
        X_va = X_train.iloc[va_idx]
        y_va = y0[va_idx]

        # XGBoost
        start = time.time()
        xgb_model = train_xgb_gpu(
            X_tr, y_tr, X_va, y_va,
            num_class=num_class,
            params=cfg["xgb_params"],
            num_boost_round=cfg["xgb_num_boost_round"],
            early_stopping_rounds=cfg["xgb_es_rounds"]
        )
        dur_xgb = time.time() - start
        logging.info(f"Fold {fold_idx} XGBoost training time: {dur_xgb:.2f}s")

        va_proba_xgb = predict_xgb(xgb_model, X_va)
        oof_proba_xgb[va_idx] = va_proba_xgb
        test_proba_xgb += predict_xgb(xgb_model, X_test).astype(np.float32) / cfg["n_splits"]

        # LightGBM
        start = time.time()
        lgb_model = train_lgb_gpu(
            X_tr, y_tr, X_va, y_va,
            num_class=num_class,
            params=cfg["lgb_params"],
            num_boost_round=cfg["lgb_num_boost_round"],
            early_stopping_rounds=cfg["lgb_es_rounds"]
        )
        dur_lgb = time.time() - start
        logging.info(f"Fold {fold_idx} LightGBM training time: {dur_lgb:.2f}s")

        va_proba_lgb = predict_lgb(lgb_model, X_va)
        oof_proba_lgb[va_idx] = va_proba_lgb
        test_proba_lgb += predict_lgb(lgb_model, X_test).astype(np.float32) / cfg["n_splits"]

        # CatBoost
        start = time.time()
        cat_model = train_cat_gpu(
            X_tr, y_tr, X_va, y_va,
            params=cfg["cat_params"]
        )
        dur_cat = time.time() - start
        logging.info(f"Fold {fold_idx} CatBoost training time: {dur_cat:.2f}s")

        va_proba_cat = predict_cat(cat_model, X_va)
        oof_proba_cat[va_idx] = va_proba_cat
        test_proba_cat += predict_cat(cat_model, X_test).astype(np.float32) / cfg["n_splits"]

        # Metrics per model and ensemble on this fold
        va_pred_xgb = va_proba_xgb.argmax(axis=1)
        va_pred_lgb = va_proba_lgb.argmax(axis=1)
        va_pred_cat = va_proba_cat.argmax(axis=1)
        va_pred_ens = (va_proba_xgb + va_proba_lgb + va_proba_cat).argmax(axis=1)

        acc_xgb = accuracy_score(y_va, va_pred_xgb)
        acc_lgb = accuracy_score(y_va, va_pred_lgb)
        acc_cat = accuracy_score(y_va, va_pred_cat)
        acc_ens = accuracy_score(y_va, va_pred_ens)

        logging.info(f"Fold {fold_idx} accuracy: XGB={acc_xgb:.6f}, LGB={acc_lgb:.6f}, CAT={acc_cat:.6f}, ENS={acc_ens:.6f}")

        per_class = per_class_accuracy(y_va, va_pred_ens)
        logging.info(f"Fold {fold_idx} ensemble per-class accuracy:")
        for k in sorted(per_class.keys()):
            logging.info(f"  Class {k+1}: {per_class[k]:.6f}")

        fold_logs.append({
            "fold": fold_idx,
            "acc_xgb": acc_xgb,
            "acc_lgb": acc_lgb,
            "acc_cat": acc_cat,
            "acc_ens": acc_ens
        })

    # OOF metrics overall
    oof_pred_xgb = oof_proba_xgb.argmax(axis=1)
    oof_pred_lgb = oof_proba_lgb.argmax(axis=1)
    oof_pred_cat = oof_proba_cat.argmax(axis=1)
    oof_pred_ens = (oof_proba_xgb + oof_proba_lgb + oof_proba_cat).argmax(axis=1)

    acc_xgb_oof = accuracy_score(y0, oof_pred_xgb)
    acc_lgb_oof = accuracy_score(y0, oof_pred_lgb)
    acc_cat_oof = accuracy_score(y0, oof_pred_cat)
    acc_ens_oof = accuracy_score(y0, oof_pred_ens)

    logging.info("Cross-validated (OOF) accuracy:")
    logging.info(f"  XGBoost OOF accuracy: {acc_xgb_oof:.6f}")
    logging.info(f"  LightGBM OOF accuracy: {acc_lgb_oof:.6f}")
    logging.info(f"  CatBoost OOF accuracy: {acc_cat_oof:.6f}")
    logging.info(f"  Ensemble OOF accuracy: {acc_ens_oof:.6f}")

    per_class_oof = per_class_accuracy(y0, oof_pred_ens)
    logging.info("Ensemble OOF per-class accuracy:")
    for k in sorted(per_class_oof.keys()):
        logging.info(f"  Class {k+1}: {per_class_oof[k]::0.6f}")

    # Final test prediction by averaging model probabilities
    test_proba_ens = (test_proba_xgb + test_proba_lgb + test_proba_cat) / 3.0
    test_pred = test_proba_ens.argmax(axis=1) + 1  # back to 1..7

    # Save submission
    sub = sample_sub.copy()
    sub["Cover_Type"] = test_pred.astype(np.int64)
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    sub.to_csv(SUBMISSION_FILE, index=False)
    logging.info(f"Submission saved to {SUBMISSION_FILE}")
    logging.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()