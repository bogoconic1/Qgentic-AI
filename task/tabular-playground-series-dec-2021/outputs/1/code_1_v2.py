import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import logging
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb


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


def detect_columns(df: pd.DataFrame):
    soil_cols = [c for c in df.columns if c.startswith("Soil_Type")]
    wild_cols = [c for c in df.columns if c.startswith("Wilderness_Area")]
    base_num_cols = [c for c in df.columns if c not in (['Id', 'Cover_Type'] + soil_cols + wild_cols)]
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


def pca_fit_torch_gpu(X_np: np.ndarray, n_components: int = 2, device: str = "cuda"):
    logging.info(f"Fitting PCA on GPU with torch for {X_np.shape[0]} samples and {X_np.shape[1]} features, n_components={n_components}")
    X = torch.from_numpy(X_np.astype(np.float32)).to(device)
    mean = X.mean(dim=0, keepdim=True)
    Xc = X - mean
    cov = (Xc.T @ Xc) / (Xc.shape[0] - 1)
    evals, evecs = torch.linalg.eigh(cov)  # ascending order
    idx = torch.argsort(evals, descending=True)
    evecs = evecs[:, idx]
    comps = evecs[:, :n_components].contiguous()
    logging.info(f"PCA eigenvalues (descending): {evals[idx].detach().cpu().numpy().tolist()}")
    return mean.detach(), comps.detach()


def pca_transform_torch_gpu(X_np: np.ndarray, mean: torch.Tensor, comps: torch.Tensor, device: str = "cuda"):
    X = torch.from_numpy(X_np.astype(np.float32)).to(device)
    Xc = X - mean
    Z = Xc @ comps
    return Z.detach().cpu().numpy()


def engineer_hillshade_pca(train_df: pd.DataFrame, test_df: pd.DataFrame, n_components: int = 2, device: str = "cuda"):
    logging.info("Engineering PCA over Hillshade columns")
    hs_cols = ["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]
    X_train = train_df[hs_cols].values
    X_test = test_df[hs_cols].values
    mean, comps = pca_fit_torch_gpu(X_train, n_components=n_components, device=device)
    Z_train = pca_transform_torch_gpu(X_train, mean, comps, device=device)
    Z_test = pca_transform_torch_gpu(X_test, mean, comps, device=device)
    for i in range(n_components):
        train_df[f"Hillshade_PCA{i+1}"] = Z_train[:, i]
        test_df[f"Hillshade_PCA{i+1}"] = Z_test[:, i]
    logging.info("PCA features added: " + ", ".join([f"Hillshade_PCA{i+1}" for i in range(n_components)]))
    return train_df, test_df


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

    # Build joint counts as (ny, nx): rows=y class, cols=x bin
    idx = x_codes + nx * y
    counts = torch.bincount(idx, minlength=nx * ny).to(torch.float64)
    pxy = counts.view(ny, nx)
    N = float(len(y_np))
    pxy = pxy / N

    # Marginals: py (ny,1), px (1,nx)
    py = pxy.sum(dim=1, keepdim=True)
    px = pxy.sum(dim=0, keepdim=True)

    den = (py @ px).to(torch.float64)  # (ny, nx)
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


def stratified_single_fold_split(y: np.ndarray, test_size: float = 0.2, seed: int = 42):
    logging.info(f"Creating a single stratified split with test_size={test_size}, seed={seed}")
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    for tr_idx, va_idx in sss.split(np.zeros_like(y), y):
        return tr_idx, va_idx


def build_feature_matrix(train_df: pd.DataFrame, test_df: pd.DataFrame,
                         use_interactions: bool = True,
                         drop_original_soil: bool = True,
                         drop_original_wild: bool = False):
    logging.info("Building feature matrices (X_train, X_test)")
    base_num_cols, wild_cols, soil_cols = detect_columns(train_df)

    engineered_cols = [c for c in train_df.columns if c.startswith("Hillshade_PCA")] + ["Hydrology_Dist", "Fire_Road_Dist", "Elevation_Band"]

    soil_clean_dum = pd.get_dummies(train_df["Soil_Type_Clean"], prefix="Soil_Clean", dtype=np.uint8)
    soil_clean_dum_te = pd.get_dummies(test_df["Soil_Type_Clean"], prefix="Soil_Clean", dtype=np.uint8)

    wild_clean_dum = pd.get_dummies(train_df["Wilderness_Clean"], prefix="Wild_Clean", dtype=np.uint8)
    wild_clean_dum_te = pd.get_dummies(test_df["Wilderness_Clean"], prefix="Wild_Clean", dtype=np.uint8)

    soil_cols_all = sorted(set(soil_clean_dum.columns).union(set(soil_clean_dum_te.columns)))
    wild_cols_all = sorted(set(wild_clean_dum.columns).union(set(wild_clean_dum_te.columns)))
    soil_clean_dum = soil_clean_dum.reindex(columns=soil_cols_all, fill_value=0)
    soil_clean_dum_te = soil_clean_dum_te.reindex(columns=soil_cols_all, fill_value=0)
    wild_clean_dum = wild_clean_dum.reindex(columns=wild_cols_all, fill_value=0)
    wild_clean_dum_te = wild_clean_dum_te.reindex(columns=wild_cols_all, fill_value=0)

    if use_interactions:
        logging.info("Engineering interaction one-hot features Soil_Type_Clean x Wilderness_Clean")
        sw_train = train_df["Soil_Type_Clean"].astype(np.int16) * 10 + train_df["Wilderness_Clean"].astype(np.int16)
        sw_test = test_df["Soil_Type_Clean"].astype(np.int16) * 10 + test_df["Wilderness_Clean"].astype(np.int16)
        sw_dum_tr = pd.get_dummies(sw_train, prefix="SW", dtype=np.uint8)
        sw_dum_te = pd.get_dummies(sw_test, prefix="SW", dtype=np.uint8)
        sw_cols = sorted(set(sw_dum_tr.columns).union(set(sw_dum_te.columns)))
        sw_dum_tr = sw_dum_tr.reindex(columns=sw_cols, fill_value=0)
        sw_dum_te = sw_dum_te.reindex(columns=sw_cols, fill_value=0)
    else:
        sw_dum_tr = pd.DataFrame(index=train_df.index)
        sw_dum_te = pd.DataFrame(index=test_df.index)

    X_train_list = []
    X_test_list = []

    X_train_list.append(train_df[base_num_cols + engineered_cols])
    X_test_list.append(test_df[base_num_cols + engineered_cols])

    X_train_list.append(soil_clean_dum)
    X_test_list.append(soil_clean_dum_te)

    X_train_list.append(wild_clean_dum)
    X_test_list.append(wild_clean_dum_te)

    if use_interactions:
        X_train_list.append(sw_dum_tr)
        X_test_list.append(sw_dum_te)

    if not drop_original_soil:
        X_train_list.append(train_df[[c for c in train_df.columns if c.startswith("Soil_Type")]])
        X_test_list.append(test_df[[c for c in test_df.columns if c.startswith("Soil_Type")]])
    if not drop_original_wild:
        X_train_list.append(train_df[[c for c in train_df.columns if c.startswith("Wilderness_Area")]])
        X_test_list.append(test_df[[c for c in test_df.columns if c.startswith("Wilderness_Area")]])

    X_train = pd.concat(X_train_list, axis=1)
    X_test = pd.concat(X_test_list, axis=1)

    all_cols = sorted(set(X_train.columns).union(set(X_test.columns)))
    X_train = X_train.reindex(columns=all_cols, fill_value=0)
    X_test = X_test.reindex(columns=all_cols, fill_value=0)

    logging.info(f"Final feature matrix shapes: X_train={X_train.shape}, X_test={X_test.shape}")
    return X_train, X_test


def compute_class_weights(y: np.ndarray):
    logging.info("Computing inverse-frequency class weights")
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    freq = counts / total
    inv = 1.0 / freq
    inv = inv / inv.mean()
    w = {cls: inv[i] for i, cls in enumerate(classes)}
    logging.info(f"Class distribution: " + json.dumps({int(c): int(n) for c, n in zip(classes, counts)}))
    logging.info(f"Class weights (normalized): " + json.dumps({int(k): float(v) for k, v in w.items()}))
    return w


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray):
    classes = np.unique(y_true)
    stats = {}
    for c in classes:
        mask = (y_true == c)
        acc = (y_pred[mask] == c).mean() if mask.sum() > 0 else 0.0
        stats[int(c)] = float(acc)
    return stats


def main():
    BASE_DIR = "task/tabular-playground-series-dec-2021"
    OUT_DIR = f"{BASE_DIR}/outputs/1"
    LOG_FILE = f"{OUT_DIR}/code_1_v2.txt"
    SUBMISSION_FILE = f"{OUT_DIR}/submission_2.csv"

    cfg = {
        "seed": 42,
        "device": "cuda",
        "soil_prefer": "first",
        "elevation_band_size": 100,
        "hillshade_pca_components": 2,
        "use_interactions": True,
        "drop_original_soil": True,
        "drop_original_wild": False,
        "validation_size": 0.2,
        "soil_reliability_weight": 0.9,
        "ks_max_samples": 500000,
        "xgb_params": {
            "objective": "multi:softprob",
            "num_class": 7,
            "tree_method": "gpu_hist",
            "predictor": "gpu_predictor",
            "eval_metric": "mlogloss",
            "max_depth": 9,
            "eta": 0.08,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 4.0,
            "gamma": 0.0,
            "lambda": 1.0,
            "verbosity": 1,
            "gpu_id": 0,
            "seed": 42,
        },
        "num_boost_round": 2000,
        "early_stopping_rounds": 100,
    }

    setup_logging(LOG_FILE)
    seed_everything(cfg["seed"])

    train_path = f"{BASE_DIR}/train.csv"
    test_path = f"{BASE_DIR}/test.csv"
    sample_sub_path = f"{BASE_DIR}/sample_submission.csv"

    logging.info(f"Loading training data from {train_path}")
    train_df = pd.read_csv(train_path)
    logging.info(f"Loading test data from {test_path}")
    test_df = pd.read_csv(test_path)
    sample_sub = pd.read_csv(sample_sub_path)
    logging.info(f"Loaded train shape={train_df.shape}, test shape={test_df.shape}")

    base_num_cols, wild_cols, soil_cols = detect_columns(train_df)

    train_df = clean_one_hot_group(train_df, soil_cols, "Soil_Type", unknown_index=0, prefer=cfg["soil_prefer"])
    test_df = clean_one_hot_group(test_df, soil_cols, "Soil_Type", unknown_index=0, prefer=cfg["soil_prefer"])

    train_df = clean_one_hot_group(train_df, wild_cols, "Wilderness", unknown_index=0, prefer="first")
    test_df = clean_one_hot_group(test_df, wild_cols, "Wilderness", unknown_index=0, prefer="first")

    train_df = engineer_distance_features(train_df)
    test_df = engineer_distance_features(test_df)

    train_df = engineer_binned_features(train_df, elevation_band_size=cfg["elevation_band_size"])
    test_df = engineer_binned_features(test_df, elevation_band_size=cfg["elevation_band_size"])

    train_df, test_df = engineer_hillshade_pca(train_df, test_df, n_components=cfg["hillshade_pca_components"], device=cfg["device"])

    cont_cols = [c for c in base_num_cols] + [f"Hillshade_PCA{i+1}" for i in range(cfg["hillshade_pca_components"])] + ["Hydrology_Dist", "Fire_Road_Dist", "Elevation_Band"]
    cat_specs = {
        "datasets": [
            ("Wilderness_Clean", train_df["Wilderness_Clean"].values.astype(np.int16), test_df["Wilderness_Clean"].values.astype(np.int16)),
            ("Soil_Type_Clean", train_df["Soil_Type_Clean"].values.astype(np.int16), test_df["Soil_Type_Clean"].values.astype(np.int16)),
        ]
    }
    distribution_shift_checks(train_df, test_df, cont_cols, cat_specs, device=cfg["device"])

    leakage_checks_mi(train_df, train_df["Cover_Type"].values, cont_cols, cat_specs, device=cfg["device"])

    X_train, X_test = build_feature_matrix(train_df, test_df,
                                           use_interactions=cfg["use_interactions"],
                                           drop_original_soil=cfg["drop_original_soil"],
                                           drop_original_wild=cfg["drop_original_wild"])

    y = train_df["Cover_Type"].astype(np.int64).values  # 1..7
    y0 = y - 1  # 0..6

    tr_idx, va_idx = stratified_single_fold_split(y0, test_size=cfg["validation_size"], seed=cfg["seed"])

    class_w = compute_class_weights(y0)
    weights = np.array([class_w[c] for c in y0], dtype=np.float32)
    reliability = train_df["Soil_Type_Reliable"].values.astype(np.float32)
    weights = weights * (cfg["soil_reliability_weight"] + (1.0 - cfg["soil_reliability_weight"]) * reliability)

    logging.info("Creating XGBoost DMatrices (GPU)")
    dtrain = xgb.DMatrix(X_train.iloc[tr_idx], label=y0[tr_idx], weight=weights[tr_idx])
    dvalid = xgb.DMatrix(X_train.iloc[va_idx], label=y0[va_idx], weight=weights[va_idx])

    params = cfg["xgb_params"]
    evals = [(dtrain, "train"), (dvalid, "valid")]
    logging.info("Starting XGBoost training with GPU")
    start_time = time.time()
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=cfg["num_boost_round"],
        evals=evals,
        early_stopping_rounds=cfg["early_stopping_rounds"],
        verbose_eval=50
    )
    elapsed = time.time() - start_time
    logging.info(f"Training completed in {elapsed:.2f} seconds. Best iteration: {booster.best_iteration}, best score: {booster.best_score}")

    logging.info("Evaluating validation performance")
    pred_valid_proba = booster.predict(dvalid, iteration_range=(0, booster.best_iteration + 1))
    pred_valid = pred_valid_proba.argmax(axis=1)  # 0..6
    acc = (pred_valid == y0[va_idx]).mean()
    per_class_acc = per_class_accuracy(y0[va_idx], pred_valid)
    logging.info(f"Validation Accuracy: {acc:.6f}")
    logging.info("Per-class validation accuracy:")
    for k in sorted(per_class_acc.keys()):
        logging.info(f"  Class {k+1}: {per_class_acc[k]:.6f}")

    logging.info("Refitting model on full training data with best_iteration")
    dfull = xgb.DMatrix(X_train, label=y0, weight=weights)
    booster_full = xgb.train(
        params=params,
        dtrain=dfull,
        num_boost_round=booster.best_iteration + 1,
        evals=[(dfull, "full")],
        verbose_eval=50
    )

    logging.info("Predicting on test set and generating submission CSV")
    dtest = xgb.DMatrix(X_test)
    pred_test_proba = booster_full.predict(dtest)
    pred_test = pred_test_proba.argmax(axis=1) + 1  # back to 1..7

    sub = sample_sub.copy()
    sub["Cover_Type"] = pred_test.astype(np.int64)
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    sub.to_csv(SUBMISSION_FILE, index=False)
    logging.info(f"Submission saved to {SUBMISSION_FILE}")
    logging.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()