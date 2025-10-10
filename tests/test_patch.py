import ast
import difflib
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from agents.developer import DeveloperAgent

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logging.getLogger("agents.developer").setLevel(logging.DEBUG)



WRONG_PATCH = """--- code_10_v1.py
+++ code_10_v1.py
@@ -25,7 +25,7 @@
 import logging
 logging.basicConfig(
-    filename=str(OUT_DIR / "code_10_v1.txt"),
+    filename=str(OUT_DIR / "code_10_v2.txt"),
     level=logging.INFO,
     format="%(asctime)s %(levelname)s %(message)s"
 )
@@ -59,7 +59,7 @@
     cfg["base_dir"] = BASE_DIR
     cfg["train_path"] = BASE_DIR / "train.json"
     cfg["test_path"] = BASE_DIR / "test.json"
-    cfg["sub_path"] = OUT_DIR / "submission_1.csv"
+    cfg["sub_path"] = OUT_DIR / "submission_2.csv"
     cfg["device"] = "cuda" if torch.cuda.is_available() else "cuda"  # enforce CUDA usage
     cfg["round_angle_decimals"] = 4
     cfg["xgb_params"] = {
@@ -446,14 +446,18 @@
     folds = list(gkf.split(np.zeros(len(y)), y, groups))
 
     # Storage
     oof_xgb = np.zeros(len(y), dtype=np.float32)
     oof_cnn = np.zeros(len(y), dtype=np.float32)
+    oof_ens_cal = np.zeros(len(y), dtype=np.float32)
+    oof_ens_pp = np.zeros(len(y), dtype=np.float32)
     test_preds_xgb_folds = []
     test_preds_cnn_folds = []
+    test_preds_ens_cal_folds = []
 
     # Models list to free later if needed
     cnn_models = []
     xgb_models = []
+    calibrators = []
 
     # CV Loop
     for fold_idx, (tr_idx, va_idx) in enumerate(folds):
         # Build per-fold pure angle map from training part (leak-aware)
         angle_map_fold = get_pure_angle_map(
@@ -498,19 +502,31 @@
         test_preds_cnn_folds.append(te_pred_cnn)
         xgb_models.append(xgb_model)
 
         # Per-fold validation logging (raw)
         va_true = y[va_idx]
         fold_ll_xgb = log_loss(va_true, np.clip(va_pred_xgb, cfg["clip_min"], cfg["clip_max"]))
         logging.info(f"[{mode_str}] XGB Fold {fold_idx} log_loss: {fold_ll_xgb:.6f}")
         fold_ll_cnn = log_loss(va_true, np.clip(va_pred_cnn, cfg["clip_min"], cfg["clip_max"]))
         logging.info(f"[{mode_str}] CNN Fold {fold_idx} log_loss: {fold_ll_cnn:.6f}")
 
         # Ensemble raw
         va_pred_ens = (va_pred_xgb + va_pred_cnn) / 2.0
         fold_ll_ens = log_loss(va_true, np.clip(va_pred_ens, cfg["clip_min"], cfg["clip_max"]))
         logging.info(f"[{mode_str}] Ensemble (raw) Fold {fold_idx} log_loss: {fold_ll_ens:.6f}")
 
-        # Post-processed ensemble (using fold-specific angle map)
-        va_angle_keys = angle_keys_train[va_idx]
-        va_na_flags = na_train[va_idx]
-        va_pred_ens_pp = apply_postprocess(cfg, va_pred_ens, va_angle_keys, va_na_flags, angle_map_fold, keep_if_conf=cfg["pp_keep_if_confident"])
-        fold_ll_ens_pp = log_loss(va_true, va_pred_ens_pp)
-        logging.info(f"[{mode_str}] Ensemble (postproc) Fold {fold_idx} log_loss: {fold_ll_ens_pp:.6f}")
+        # Fold-wise calibration (fit calibrator on this fold's logits and labels)
+        va_logits = sigmoid_to_logit(va_pred_ens)
+        calib_fold = LogisticRegression(max_iter=1000, solver="lbfgs")
+        calib_fold.fit(va_logits.reshape(-1, 1), va_true)
+        calibrators.append(calib_fold)
+        va_pred_ens_cal = calib_fold.predict_proba(va_logits.reshape(-1, 1))[:, 1]
+        oof_ens_cal[va_idx] = va_pred_ens_cal
+        fold_ll_ens_cal = log_loss(va_true, np.clip(va_pred_ens_cal, cfg["clip_min"], cfg["clip_max"]))
+        logging.info(f"[{mode_str}] Ensemble (calibrated) Fold {fold_idx} log_loss: {fold_ll_ens_cal:.6f}")
+
+        # Post-processed calibrated ensemble (using fold-specific angle map)
+        va_angle_keys = angle_keys_train[va_idx]
+        va_na_flags = na_train[va_idx]
+        va_pred_ens_cal_pp = apply_postprocess(cfg, va_pred_ens_cal, va_angle_keys, va_na_flags, angle_map_fold, keep_if_conf=cfg["pp_keep_if_confident"])
+        oof_ens_pp[va_idx] = va_pred_ens_cal_pp
+        fold_ll_ens_cal_pp = log_loss(va_true, va_pred_ens_cal_pp)
+        logging.info(f"[{mode_str}] Ensemble (calibrated + postproc) Fold {fold_idx} log_loss: {fold_ll_ens_cal_pp:.6f}")
+
+        # Calibrated test predictions for this fold
+        te_pred_ens = (te_pred_xgb + te_pred_cnn) / 2.0
+        te_logits = sigmoid_to_logit(te_pred_ens)
+        te_pred_cal = calib_fold.predict_proba(te_logits.reshape(-1, 1))[:, 1]
+        test_preds_ens_cal_folds.append(te_pred_cal)
 
     # OOF metrics
     oof_ens_raw = (oof_xgb + oof_cnn) / 2.0
     oof_ll_xgb = log_loss(y, np.clip(oof_xgb, cfg["clip_min"], cfg["clip_max"]))
     oof_ll_cnn = log_loss(y, np.clip(oof_cnn, cfg["clip_min"], cfg["clip_max"]))
     oof_ll_ens_raw = log_loss(y, np.clip(oof_ens_raw, cfg["clip_min"], cfg["clip_max"]))
     logging.info(f"[{mode_str}] OOF XGB log_loss: {oof_ll_xgb:.6f}")
     logging.info(f"[{mode_str}] OOF CNN log_loss: {oof_ll_cnn:.6f}")
     logging.info(f"[{mode_str}] OOF Ensemble (raw) log_loss: {oof_ll_ens_raw:.6f}")
 
-    # Calibrate ensemble using OOF (Platt scaling via LogisticRegression on logits)
-    oof_logits = sigmoid_to_logit(oof_ens_raw)
-    calib = LogisticRegression(max_iter=1000, solver="lbfgs")
-    calib.fit(oof_logits.reshape(-1, 1), y)
-    oof_calibrated = calib.predict_proba(oof_logits.reshape(-1, 1))[:, 1]
-    oof_ll_ens_cal = log_loss(y, np.clip(oof_calibrated, cfg["clip_min"], cfg["clip_max"]))
-    logging.info(f"[{mode_str}] OOF Ensemble (calibrated) log_loss: {oof_ll_ens_cal:.6f}")
-
-    # Post-processing on OOF using full-train pure angle map
-    full_angle_map = get_pure_angle_map(train_df, angle_keys_train, y, min_count=2)
-    oof_ens_pp = apply_postprocess(cfg, oof_calibrated, angle_keys_train, na_train, full_angle_map, keep_if_conf=cfg["pp_keep_if_confident"])
-    oof_ll_ens_pp = log_loss(y, oof_ens_pp)
-    logging.info(f"[{mode_str}] OOF Ensemble (calibrated + postproc) log_loss: {oof_ll_ens_pp:.6f}")
+    # OOF calibrated and post-processed metrics (built fold-wise to avoid leakage)
+    oof_ll_ens_cal = log_loss(y, np.clip(oof_ens_cal, cfg["clip_min"], cfg["clip_max"]))
+    logging.info(f"[{mode_str}] OOF Ensemble (calibrated) log_loss: {oof_ll_ens_cal:.6f}")
+    oof_ll_ens_pp_val = log_loss(y, oof_ens_pp)
+    logging.info(f"[{mode_str}] OOF Ensemble (calibrated + postproc) log_loss: {oof_ll_ens_pp_val:.6f}")
 
     # Final validation results summary
-    logging.info(f"[{mode_str}] FINAL OOF RESULTS -> XGB: {oof_ll_xgb:.6f} | CNN: {oof_ll_cnn:.6f} | Ensemble Raw: {oof_ll_ens_raw:.6f} | Ensemble Calibrated: {oof_ll_ens_cal:.6f} | Ensemble Calibrated+Postproc: {oof_ll_ens_pp:.6f}")
+    logging.info(f"[{mode_str}] FINAL OOF RESULTS -> XGB: {oof_ll_xgb:.6f} | CNN: {oof_ll_cnn:.6f} | Ensemble Raw: {oof_ll_ens_raw:.6f} | Ensemble Calibrated: {oof_ll_ens_cal:.6f} | Ensemble Calibrated+Postproc: {oof_ll_ens_pp_val:.6f}")
 
-    # Test predictions
-    test_pred_xgb = np.mean(np.stack(test_preds_xgb_folds, axis=0), axis=0)
-    test_pred_cnn = np.mean(np.stack(test_preds_cnn_folds, axis=0), axis=0)
-    test_pred_ens_raw = (test_pred_xgb + test_pred_cnn) / 2.0
-    test_logits = sigmoid_to_logit(test_pred_ens_raw)
-    test_pred_cal = calib.predict_proba(test_logits.reshape(-1, 1))[:, 1]
-    test_pred_final = apply_postprocess(cfg, test_pred_cal, angle_keys_test, na_test, full_angle_map, keep_if_conf=cfg["pp_keep_if_confident"])
+    # Build full-train pure angle map for TEST post-processing only (no OOF use)
+    full_angle_map = get_pure_angle_map(train_df, angle_keys_train, y, min_count=2)
+
+    # Test predictions (per-fold calibrated ensemble averaged)
+    test_pred_ens_cal = np.mean(np.stack(test_preds_ens_cal_folds, axis=0), axis=0)
+    test_pred_final = apply_postprocess(cfg, test_pred_ens_cal, angle_keys_test, na_test, full_angle_map, keep_if_conf=cfg["pp_keep_if_confident"])
     # Clip
     test_pred_final = np.clip(test_pred_final, cfg["clip_min"], cfg["clip_max"])
 
     # Save submission only for FULL run
     if not debug:
         sub = pd.DataFrame({"id": test_df["id"], "is_iceberg": test_pred_final})
         sub.to_csv(cfg["sub_path"], index=False)
"""

@pytest.fixture
def developer_agent(monkeypatch):
    monkeypatch.setattr(DeveloperAgent, "_load_benchmark_info", lambda self: None)
    agent = DeveloperAgent("statoil-iceberg-classifier-challenge", iteration=10)
    agent.outputs_dir = Path("/workspace/gstar-project")
    agent.developer_log_path = agent.outputs_dir / "developer_patch.log"
    agent.plan_path = agent.outputs_dir / "developer_patch_plan.md"
    return agent


def test_apply_wrong_then_correct_patch(developer_agent, tmp_path):
    base_source = Path("task/statoil-iceberg-classifier-challenge/outputs/10/code_10_v1.py")
    base_source_content = base_source.read_text()

    work_dir = tmp_path / "outputs"
    work_dir.mkdir()

    copied_base = work_dir / base_source.name
    copied_base.write_text(base_source_content)

    developer_agent.outputs_dir = work_dir
    developer_agent.developer_log_path = work_dir / "developer_patch.log"
    developer_agent.plan_path = work_dir / "developer_patch_plan.md"

    normalized = DeveloperAgent._normalize_diff_payload(copied_base, WRONG_PATCH)
    assert normalized is not None

    new_file = developer_agent.outputs_dir / developer_agent._code_filename(2)
    if new_file.exists():
        new_file.unlink()

    result = developer_agent._apply_patch(base_version=1, diff_payload=WRONG_PATCH, target_version=2)
    assert result is not None

    expected_lines = DeveloperAgent._apply_unified_diff(base_source_content.splitlines(), normalized.splitlines())
    assert expected_lines is not None
    assert result.splitlines() == expected_lines

    assert copied_base.read_text() == base_source_content

    try:
        assert new_file.exists()
    finally:
        if new_file.exists():
            new_file.unlink()
