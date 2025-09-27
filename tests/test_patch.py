import difflib
import shutil
import tempfile
from pathlib import Path

from agents.developer import DeveloperAgent


BASE_TEMPLATE = """import os
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

BASE_DIR = "task/statoil-iceberg-classifier-challenge"
OUT_DIR = os.path.join(BASE_DIR, "outputs", "10")
os.makedirs(OUT_DIR, exist_ok=True)
LOG_PATH = os.path.join(OUT_DIR, "code_10_v1.txt")

logging.basicConfig(
    filename=LOG_PATH,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

def predict_cnn(model: nn.Module, ds: Dataset, device: str, batch_size: int) -> np.ndarray:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)
    model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 2:
                xb, auxb = batch
            else:
                xb, auxb, _ = batch
            xb = xb.to(device)
            auxb = auxb.to(device)
            with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=torch.bfloat16):
                logits = model(xb, auxb)
                probs = torch.sigmoid(logits).float()
            preds.append(probs.cpu().numpy())
    return np.concatenate(preds)

def train_cnn_folds(model, te_dataset, device, batch_size, y):
    test_pred_cnn_folds = []
    fold_id = 0
    val_probs = np.array([])
    val_ll = 0.0
    va_idx = slice(None)
    oof_cnn = np.zeros_like(y, dtype=float)
    oof_cnn[va_idx] = val_probs
    logging.info(f"[CNN] Fold {fold_id} - val_logloss={val_ll:.6f}")

    # Test predictions for this fold
    test_probs = predict_cnn(model, te_dataset, device=device, batch_size=batch_size)
    test_pred_cnn_folds.append(test_probs)
    fold_id += 1
    cnn_oof_logloss = log_loss(y, np.clip(oof_cnn, 1e-6, 1-1e-6))
    logging.info(f"[CNN] OOF logloss={cnn_oof_logloss:.6f}")
    return test_pred_cnn_folds

def run_pipeline(DEBUG: bool):
    sub = pd.DataFrame({"target": [0.0]})
    if DEBUG:
        sub_debug_path = os.path.join(OUT_DIR, "submission_debug.csv")
        sub.to_csv(sub_debug_path, index=False)
    else:
        sub_path = os.path.join(OUT_DIR, "submission_1.csv")
        sub.to_csv(sub_path, index=False)
    logging.info("Done")
"""


PATCHED_TEMPLATE = """import os
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

BASE_DIR = "task/statoil-iceberg-classifier-challenge"
OUT_DIR = os.path.join(BASE_DIR, "outputs", "10")
os.makedirs(OUT_DIR, exist_ok=True)
LOG_PATH = os.path.join(OUT_DIR, "code_10_v2.txt")

logging.basicConfig(
    filename=LOG_PATH,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

def predict_cnn(model: nn.Module, ds: Dataset, device: str, batch_size: int) -> np.ndarray:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)
    model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 2:
                xb, auxb = batch
            else:
                xb, auxb, _ = batch
            xb = xb.to(device)
            auxb = auxb.to(device)
            with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=torch.bfloat16):
                logits = model(xb, auxb)
                probs = torch.sigmoid(logits).float()
            preds.append(probs.cpu().numpy())
    return np.concatenate(preds)

def predict_cnn_tta(model: nn.Module, ds: Dataset, device: str, batch_size: int, use_tta: bool = True) -> np.ndarray:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)
    model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 2:
                xb, auxb = batch
            else:
                xb, auxb, _ = batch
            xb = xb.to(device)  # (B,3,75,75)
            auxb = auxb.to(device)

            if not use_tta:
                with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=torch.bfloat16):
                    logits = model(xb, auxb)
                    probs = torch.sigmoid(logits).float()
                preds.append(probs.cpu().numpy())
                continue

            # D8 dihedral TTA: rotations 0/90/180/270 and their horizontal flips
            tta_probs_sum = torch.zeros(xb.size(0), device=device, dtype=torch.float32)
            for k in range(4):
                xr = torch.rot90(xb, k=k, dims=(2, 3))
                with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=torch.bfloat16):
                    pr = torch.sigmoid(model(xr, auxb)).float()
                tta_probs_sum += pr
                xrf = torch.flip(xr, dims=(3,))  # horizontal flip
                with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=torch.bfloat16):
                    prf = torch.sigmoid(model(xrf, auxb)).float()
                tta_probs_sum += prf

            probs = tta_probs_sum / 8.0
            preds.append(probs.cpu().numpy())
    return np.concatenate(preds)

def train_cnn_folds(model, te_dataset, device, batch_size, y):
    test_pred_cnn_folds = []
    fold_id = 0
    val_probs = np.array([])
    val_ll = 0.0
    va_idx = slice(None)
    oof_cnn = np.zeros_like(y, dtype=float)
    oof_cnn[va_idx] = val_probs
    logging.info(f"[CNN] Fold {fold_id} - val_logloss={val_ll:.6f}")

    # Test predictions for this fold
    test_probs = predict_cnn_tta(model, te_dataset, device=device, batch_size=batch_size, use_tta=True)
    test_pred_cnn_folds.append(test_probs)
    fold_id += 1
    cnn_oof_logloss = log_loss(y, np.clip(oof_cnn, 1e-6, 1-1e-6))
    logging.info(f"[CNN] OOF logloss={cnn_oof_logloss:.6f}")
    return test_pred_cnn_folds

def run_pipeline(DEBUG: bool):
    sub = pd.DataFrame({"target": [0.0]})
    if DEBUG:
        sub_debug_path = os.path.join(OUT_DIR, "submission_debug.csv")
        sub.to_csv(sub_debug_path, index=False)
    else:
        sub_path = os.path.join(OUT_DIR, "submission_2.csv")
        sub.to_csv(sub_path, index=False)
    logging.info("Done")
"""


diff_body = "".join(
    difflib.unified_diff(
        BASE_TEMPLATE.splitlines(keepends=True),
        PATCHED_TEMPLATE.splitlines(keepends=True),
        fromfile="code_10_v1.py",
        tofile="code_10_v1.py",
    )
)

PATCH_PAYLOAD = f"*** Begin Patch\n*** Update File: code_10_v1.py\n{diff_body}*** End Patch"

PATCH_PAYLOAD = """*** Begin Patch
--- code_10_v1.py
+++ code_10_v1.py
@@ -11,7 +11,7 @@
 BASE_DIR = "task/statoil-iceberg-classifier-challenge"
 OUT_DIR = os.path.join(BASE_DIR, "outputs", "10")
 os.makedirs(OUT_DIR, exist_ok=True)
-LOG_PATH = os.path.join(OUT_DIR, "code_10_v1.txt")
+LOG_PATH = os.path.join(OUT_DIR, "code_10_v2.txt")
 
 logging.basicConfig(
     filename=LOG_PATH,
@@ -39,6 +39,43 @@
             preds.append(probs.cpu().numpy())
     return np.concatenate(preds)
 
+def predict_cnn_tta(model: nn.Module, ds: Dataset, device: str, batch_size: int, use_tta: bool = True) -> np.ndarray:
+    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)
+    model.to(device)
+    model.eval()
+    preds = []
+    with torch.no_grad():
+        for batch in loader:
+            if len(batch) == 2:
+                xb, auxb = batch
+            else:
+                xb, auxb, _ = batch
+            xb = xb.to(device)  # (B,3,75,75)
+            auxb = auxb.to(device)
+
+            if not use_tta:
+                with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=torch.bfloat16):
+                    logits = model(xb, auxb)
+                    probs = torch.sigmoid(logits).float()
+                preds.append(probs.cpu().numpy())
+                continue
+
+            # D8 dihedral TTA: rotations 0/90/180/270 and their horizontal flips
+            tta_probs_sum = torch.zeros(xb.size(0), device=device, dtype=torch.float32)
+            for k in range(4):
+                xr = torch.rot90(xb, k=k, dims=(2, 3))
+                with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=torch.bfloat16):
+                    pr = torch.sigmoid(model(xr, auxb)).float()
+                tta_probs_sum += pr
+                xrf = torch.flip(xr, dims=(3,))  # horizontal flip
+                with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=torch.bfloat16):
+                    prf = torch.sigmoid(model(xrf, auxb)).float()
+                tta_probs_sum += prf
+
+            probs = tta_probs_sum / 8.0
+            preds.append(probs.cpu().numpy())
+    return np.concatenate(preds)
+
 def train_cnn_folds(model, te_dataset, device, batch_size, y):
     test_pred_cnn_folds = []
     fold_id = 0
@@ -50,7 +87,7 @@
     logging.info(f"[CNN] Fold {fold_id} - val_logloss={val_ll:.6f}")
 
     # Test predictions for this fold
-    test_probs = predict_cnn(model, te_dataset, device=device, batch_size=batch_size)
+    test_probs = predict_cnn_tta(model, te_dataset, device=device, batch_size=batch_size, use_tta=True)
     test_pred_cnn_folds.append(test_probs)
     fold_id += 1
     cnn_oof_logloss = log_loss(y, np.clip(oof_cnn, 1e-6, 1-1e-6))
@@ -63,6 +100,6 @@
         sub_debug_path = os.path.join(OUT_DIR, "submission_debug.csv")
         sub.to_csv(sub_debug_path, index=False)
     else:
-        sub_path = os.path.join(OUT_DIR, "submission_1.csv")
+        sub_path = os.path.join(OUT_DIR, "submission_2.csv")
         sub.to_csv(sub_path, index=False)
     logging.info("Done")
*** End Patch
"""


def main() -> None:
    slug = "statoil-iceberg-classifier-challenge"
    iteration = 10
    original_loader = DeveloperAgent._load_benchmark_info
    DeveloperAgent._load_benchmark_info = lambda self: None
    try:
        agent = DeveloperAgent(slug, iteration)
    finally:
        DeveloperAgent._load_benchmark_info = original_loader

    source_path = Path(
        "/workspace/gstar-project/task/statoil-iceberg-classifier-challenge/outputs/10/code_10_v1.py"
    )
    created_source = False
    if not source_path.exists():
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(BASE_TEMPLATE)
        created_source = True

    temp_dir = Path(tempfile.mkdtemp(prefix="patch-agent-demo-"))
    try:
        agent.outputs_dir = temp_dir
        base_path = agent.outputs_dir / agent._code_filename(1)
        agent.outputs_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(source_path, base_path)

        updated_code = agent._apply_patch(base_version=1, diff_payload=PATCH_PAYLOAD, target_version=2)
        if updated_code is None:
            raise RuntimeError("DeveloperAgent._apply_patch returned None")

        patched_path = agent.outputs_dir / agent._code_filename(2)
        patched_contents = patched_path.read_text()

        print("Scratch outputs dir:", agent.outputs_dir)
        print("Updated file written to:", patched_path)
        print("Contains new LOG_PATH:", "code_10_v2.txt" in patched_contents)
        print("Has predict_cnn_tta:", "predict_cnn_tta" in patched_contents)
        print("Uses submission_2:", "submission_2.csv" in patched_contents)
    finally:
        shutil.rmtree(agent.outputs_dir, ignore_errors=True)
        if created_source:
            try:
                source_path.unlink()
                source_path.parent.rmdir()
            except OSError:
                pass


if __name__ == "__main__":
    main()
