import os
import sys
import math
import time
import random
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# ============================================================
# Configuration and Logging
# ============================================================
BASE_DIR = "task/us-patent-phrase-to-phrase-matching"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "2")
LOG_PATH = os.path.join(OUTPUT_DIR, "code_2_v1.txt")
SUB_PATH = os.path.join(OUTPUT_DIR, "submission_1.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logging.info("Initialized logging.")
logging.info(f"Base directory: {BASE_DIR}")
logging.info(f"Output directory: {OUTPUT_DIR}")
logging.info(f"Log path: {LOG_PATH}")
logging.info(f"Submission path: {SUB_PATH}")

# ============================================================
# Reproducibility and CUDA setup
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

assert torch.cuda.is_available(), "CUDA device is required and was not found."
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

logging.info(f"Using device: {device}")
logging.info(f"CUDA device count: {torch.cuda.device_count()}")
logging.info(f"Current CUDA device: {torch.cuda.current_device()}")
logging.info(f"GPU name: {torch.cuda.get_device_name(0)}")
logging.info(f"PyTorch version: {torch.__version__}")

# ============================================================
# Data Loading
# ============================================================
train_path = os.path.join(BASE_DIR, "train.csv")
test_path = os.path.join(BASE_DIR, "test.csv")
titles_path = os.path.join(BASE_DIR, "titles.csv")

logging.info(f"Loading train data from {train_path}")
train_df = pd.read_csv(train_path)
logging.info(f"Train shape: {train_df.shape}")

logging.info(f"Loading test data from {test_path}")
test_df = pd.read_csv(test_path)
logging.info(f"Test shape: {test_df.shape}")

logging.info(f"Loading CPC titles from {titles_path}")
titles_df = pd.read_csv(titles_path)
logging.info(f"Titles shape: {titles_df.shape}")
logging.info(f"Titles columns: {list(titles_df.columns)}")

# ============================================================
# Merge CPC Titles and Build Text
# ============================================================
titles_df = titles_df.rename(columns={"code": "context"})
logging.info("Merging CPC titles into train and test on 'context'")
train_df = train_df.merge(titles_df[["context", "title"]], on="context", how="left")
test_df = test_df.merge(titles_df[["context", "title"]], on="context", how="left")

missing_train_titles = train_df["title"].isna().sum()
missing_test_titles = test_df["title"].isna().sum()
logging.info(f"Missing titles in train after merge: {missing_train_titles}")
logging.info(f"Missing titles in test after merge: {missing_test_titles}")
assert missing_train_titles == 0, "There are missing titles in train after merge."
assert missing_test_titles == 0, "There are missing titles in test after merge."

# Lowercase and concatenate: "anchor [SEP] target [SEP] context_title"
logging.info("Lowercasing text fields and concatenating with [SEP] placeholders (tokenizer sep_token will be used later).")
train_df["anchor"] = train_df["anchor"].astype(str).str.lower()
train_df["target"] = train_df["target"].astype(str).str.lower()
train_df["title"] = train_df["title"].astype(str).str.lower()

test_df["anchor"] = test_df["anchor"].astype(str).str.lower()
test_df["target"] = test_df["target"].astype(str).str.lower()
test_df["title"] = test_df["title"].astype(str).str.lower()

# We will replace [SEP] with tokenizer.sep_token after tokenizer is loaded.
train_df["text"] = train_df["anchor"] + " [SEP] " + train_df["target"] + " [SEP] " + train_df["title"]
test_df["text"] = test_df["anchor"] + " [SEP] " + test_df["target"] + " [SEP] " + test_df["title"]

unique_contexts_train = train_df["context"].nunique()
unique_contexts_test = test_df["context"].nunique()
logging.info(f"Unique contexts in train: {unique_contexts_train}")
logging.info(f"Unique contexts in test: {unique_contexts_test}")

# ============================================================
# Tokenizer and Model Setup
# ============================================================
MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LEN = 150
NUM_EPOCHS = 3
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
BATCH_SIZE = 8  # Effective BS can be increased via gradient_accumulation_steps
GRAD_ACCUM_STEPS = 2  # Effective global batch size = BATCH_SIZE * GRAD_ACCUM_STEPS
N_SPLITS = 4

logging.info(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
sep_token = tokenizer.sep_token
logging.info(f"Tokenizer SEP token: {sep_token}")

# Replace literal [SEP] placeholders with tokenizer.sep_token for proper tokenization
train_df["text"] = train_df["text"].str.replace("[SEP]", sep_token)
test_df["text"] = test_df["text"].str.replace("[SEP]", sep_token)

# ============================================================
# Binning labels and Computing Context-Specific Weights
# ============================================================
def score_to_bin_index(score_float: float) -> int:
    # Scores are in {0.0, 0.25, 0.5, 0.75, 1.0}
    return int(round(score_float * 4))

logging.info("Creating binned labels for stratification.")
train_df["bin"] = train_df["score"].apply(score_to_bin_index).astype(int)

# Sanity check distribution
bin_counts = train_df["bin"].value_counts().sort_index()
for bi, cnt in bin_counts.items():
    logging.info(f"Bin {bi} (score {bi/4.0:.2f}) count: {cnt}")

# Compute per-context, per-bin weights inversely proportional to frequency within context
logging.info("Computing per-context, per-bin sample weights to handle imbalance.")
group = train_df.groupby(["context", "bin"]).size().reset_index(name="count")
ctx_counts = train_df.groupby("context").size().reset_index(name="ctx_total")

weights_map = {}
for ctx in train_df["context"].unique():
    ctx_df = group[group["context"] == ctx]
    if ctx_df.empty:
        continue
    max_count = ctx_df["count"].max()
    for _, row in ctx_df.iterrows():
        b = int(row["bin"])
        c = int(row["count"])
        # weight = max_count / count -> rarer bins get higher weight; normalize slightly
        weight = (max_count / c)
        weights_map[(ctx, b)] = weight

weights_list = []
for i, row in train_df.iterrows():
    ctx = row["context"]
    b = int(row["bin"])
    w = weights_map[(ctx, b)]
    weights_list.append(w)

train_df["weight_raw"] = weights_list
# Normalize weights to have mean ~1 across dataset
mean_w = train_df["weight_raw"].mean()
train_df["weight"] = train_df["weight_raw"] / mean_w
logging.info(f"Mean raw weight before normalization: {mean_w:.6f}")
logging.info(f"Mean normalized weight: {train_df['weight'].mean():.6f}, std: {train_df['weight'].std():.6f}")
logging.info("Sample weights (context,bin) mapping snapshot (first 10):")
for k in list(weights_map.keys())[:10]:
    logging.info(f"  {k}: {weights_map[k]:.4f}")

# ============================================================
# Dataset class
# ============================================================
class PatentDataset(Dataset):
    def __init__(self, texts, labels=None, weights=None, tokenizer=None, max_length=128):
        logging.info(f"Initializing PatentDataset with {len(texts)} samples; encode now (labels present: {labels is not None}).")
        enc = tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
        )
        self.input_ids = np.array(enc["input_ids"], dtype=np.int64)
        self.attention_mask = np.array(enc["attention_mask"], dtype=np.int64)
        self.labels = None if labels is None else np.array(labels, dtype=np.float32)
        self.weights = None if weights is None else np.array(weights, dtype=np.float32)

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        if self.weights is not None:
            item["weights"] = torch.tensor(self.weights[idx], dtype=torch.float32)
        return item

# ============================================================
# Metrics
# ============================================================
def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))

def pearson_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xm = x.mean()
    ym = y.mean()
    xv = x - xm
    yv = y - ym
    num = np.sum(xv * yv)
    den = math.sqrt(np.sum(xv ** 2) * np.sum(yv ** 2))
    if den == 0.0:
        return 0.0
    return float(num / den)

def compute_metrics(eval_pred):
    preds = eval_pred.predictions
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = preds.reshape(-1)
    preds_sig = sigmoid_np(preds)
    preds_sig = np.clip(preds_sig, 0.0, 1.0)
    labels = eval_pred.label_ids.reshape(-1)
    r_cont = pearson_corr(preds_sig, labels)
    preds_rounded = np.round(preds_sig * 4.0) / 4.0
    r_round = pearson_corr(preds_rounded, labels)
    metrics = {
        "pearson_continuous": r_cont,
        "pearson_rounded025": r_round,
    }
    logging.info(f"Eval metrics -> Pearson continuous: {r_cont:.6f}, Pearson rounded(0.25): {r_round:.6f}")
    return metrics

# ============================================================
# Custom Trainer for Weighted MSE
# ============================================================
class WeightedMSETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs["labels"]
        weights = inputs["weights"]
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.view(-1)
        labels = labels.to(logits.dtype).view(-1)
        weights = weights.to(logits.dtype).view(-1)

        # MSE with per-sample weights
        loss = torch.mean(weights * (logits - labels) ** 2)
        if return_outputs:
            return loss, outputs
        return loss

# ============================================================
# Cross-validation Training
# ============================================================
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
oof_preds = np.zeros(len(train_df), dtype=np.float32)
test_preds_folds = []

fold_indices = list(skf.split(train_df, train_df["bin"]))
logging.info(f"Prepared StratifiedKFold with {N_SPLITS} splits. Fold sizes:")
for fi, (tr_idx, va_idx) in enumerate(fold_indices):
    logging.info(f"Fold {fi}: train={len(tr_idx)}, valid={len(va_idx)}")

for fold, (train_idx, valid_idx) in enumerate(fold_indices):
    logging.info(f"========== Fold {fold + 1}/{N_SPLITS} ==========")
    fold_dir = os.path.join(OUTPUT_DIR, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    df_tr = train_df.iloc[train_idx].reset_index(drop=True)
    df_va = train_df.iloc[valid_idx].reset_index(drop=True)

    logging.info(f"Tokenizing train fold {fold}...")
    dtrain = PatentDataset(
        texts=df_tr["text"].tolist(),
        labels=df_tr["score"].astype(np.float32).tolist(),
        weights=df_tr["weight"].astype(np.float32).tolist(),
        tokenizer=tokenizer,
        max_length=MAX_LEN,
    )

    logging.info(f"Tokenizing valid fold {fold}...")
    dvalid = PatentDataset(
        texts=df_va["text"].tolist(),
        labels=df_va["score"].astype(np.float32).tolist(),
        weights=df_va["weight"].astype(np.float32).tolist(),
        tokenizer=tokenizer,
        max_length=MAX_LEN,
    )

    logging.info(f"Loading pretrained model: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=1,
        problem_type="regression",
        torch_dtype=torch.float16
    )
    model.to(device)
    model.gradient_checkpointing_enable()
    logging.info("Model loaded and moved to CUDA with fp16; enabled gradient checkpointing.")

    total_steps_estimate = math.ceil(len(dtrain) / (BATCH_SIZE * GRAD_ACCUM_STEPS)) * NUM_EPOCHS
    logging.info(f"Training configuration: epochs={NUM_EPOCHS}, lr={LR}, batch_size={BATCH_SIZE}, grad_accum={GRAD_ACCUM_STEPS}, total_steps_estimate={total_steps_estimate}")

    args = TrainingArguments(
        output_dir=fold_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="no",
        report_to=[],
        fp16=True,
        dataloader_pin_memory=True,
        load_best_model_at_end=False,
        disable_tqdm=True,
    )

    trainer = WeightedMSETrainer(
        model=model,
        args=args,
        train_dataset=dtrain,
        eval_dataset=dvalid,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    logging.info("Starting training...")
    train_start = time.time()
    trainer.train()
    train_end = time.time()
    logging.info(f"Training finished for fold {fold} in {(train_end - train_start):.2f} seconds.")

    logging.info("Running validation predictions...")
    preds_output = trainer.predict(dvalid)
    val_logits = preds_output.predictions.reshape(-1)
    val_probs = sigmoid_np(val_logits)
    val_probs = np.clip(val_probs, 0.0, 1.0)
    oof_preds[valid_idx] = val_probs

    # Compute and log validation Pearson correlations
    y_true = df_va["score"].values.astype(np.float32)
    pearson_cont = pearson_corr(val_probs, y_true)
    val_round025 = np.round(val_probs * 4.0) / 4.0
    pearson_round = pearson_corr(val_round025, y_true)
    logging.info(f"Fold {fold} Pearson (continuous): {pearson_cont:.6f}")
    logging.info(f"Fold {fold} Pearson (rounded 0.25): {pearson_round:.6f}")

    logging.info("Tokenizing test data for inference...")
    dtest = PatentDataset(
        texts=test_df["text"].tolist(),
        labels=None,
        weights=None,
        tokenizer=tokenizer,
        max_length=MAX_LEN,
    )

    logging.info("Running test predictions for current fold...")
    test_preds_logits = trainer.predict(dtest).predictions.reshape(-1)
    test_probs = sigmoid_np(test_preds_logits)
    test_probs = np.clip(test_probs, 0.0, 1.0)
    test_preds_folds.append(test_probs.astype(np.float32))

    # Cleanup to free memory between folds
    del trainer
    del model
    torch.cuda.empty_cache()
    logging.info(f"Completed fold {fold} and freed CUDA cache.")

# ============================================================
# OOF Validation Results
# ============================================================
logging.info("Computing out-of-fold validation results...")
y_full = train_df["score"].values.astype(np.float32)
oof_cont_pearson = pearson_corr(oof_preds, y_full)
oof_round025 = np.round(oof_preds * 4.0) / 4.0
oof_round_pearson = pearson_corr(oof_round025, y_full)
logging.info(f"FINAL OOF Pearson (continuous): {oof_cont_pearson:.6f}")
logging.info(f"FINAL OOF Pearson (rounded 0.25): {oof_round_pearson:.6f}")

# ============================================================
# Ensembling and Submission
# ============================================================
logging.info("Ensembling test predictions across folds (mean).")
test_preds_mean = np.mean(np.stack(test_preds_folds, axis=0), axis=0)
test_preds_mean = np.clip(test_preds_mean, 0.0, 1.0)

logging.info("Applying rounding to nearest 0.25 for submission.")
test_preds_round025 = np.round(test_preds_mean * 4.0) / 4.0
submission = pd.DataFrame({
    "id": test_df["id"].values,
    "score": test_preds_round025
})

logging.info(f"Saving submission to {SUB_PATH}")
submission.to_csv(SUB_PATH, index=False)
logging.info("Submission saved successfully.")
logging.info("Script completed.")