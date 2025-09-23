import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import math
import time
import random
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import AdamW
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    DataCollatorWithPadding,
    get_cosine_schedule_with_warmup,
)
from sklearn.model_selection import StratifiedKFold

# =========================
# Paths and Logging Setup
# =========================
BASE_DIR = "task/us-patent-phrase-to-phrase-matching"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "3")
LOG_PATH = os.path.join(OUTPUT_DIR, "code_3_v1.txt")
SUB_PATH = os.path.join(OUTPUT_DIR, "submission_1.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    filename=LOG_PATH,
    filemode="w",
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logging.info("Initialized logging. All logs will be written to %s", LOG_PATH)

# =========================
# Reproducibility and CUDA
# =========================
def set_seed(seed: int):
    logging.info("Setting random seeds: %d", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.benchmark = True
device = torch.device("cuda")
logging.info("Using device: %s", device)
logging.info("CUDA available: %s", torch.cuda.is_available())
logging.info("CUDA device name: %s", torch.cuda.get_device_name(0))

# =========================
# Configuration
# =========================
@dataclass
class Config:
    model_name: str = "microsoft/deberta-v3-xsmall"  # small/fast for early iteration
    max_len: int = 64  # dynamic padding will be used; cap max length
    train_bs: int = 64
    valid_bs: int = 128
    lr: float = 5e-5
    weight_decay: float = 0.01
    epochs: int = 1  # early iteration, keep short; configurable
    n_folds: int = 5
    seed: int = 42
    warmup_ratio: float = 0.10
    grad_accum_steps: int = 1
    scheduler: str = "cosine"
    use_fp16: bool = True  # Always use fp16 as per requirement
    pooling: str = "mean"  # mean pooling by default
    max_grad_norm: float = 1.0
    pad_to_multiple_of: int = 8
    text_prefix_anchor: str = "anchor:"
    text_prefix_target: str = "target:"
    text_prefix_context: str = "context:"
    text_prefix_title: str = "title:"
    text_prefix_section: str = "section:"
    text_prefix_subclass: str = "subclass:"
    use_context_title: bool = True
    use_context_section: bool = True
    use_context_subclass: bool = True
    # Logging & Evaluation
    log_interval: int = 100

CFG = Config()
logging.info("Config: %s", asdict(CFG))

set_seed(CFG.seed)

# =========================
# Data Loading
# =========================
train_path = os.path.join(BASE_DIR, "train.csv")
test_path = os.path.join(BASE_DIR, "test.csv")
titles_path = os.path.join(BASE_DIR, "titles.csv")
sample_sub_path = os.path.join(BASE_DIR, "sample_submission.csv")

logging.info("Loading train.csv from %s", train_path)
train_df = pd.read_csv(train_path)
logging.info("Loading test.csv from %s", test_path)
test_df = pd.read_csv(test_path)
logging.info("Loading titles.csv from %s", titles_path)
titles_df = pd.read_csv(titles_path)
logging.info("Loading sample_submission.csv from %s", sample_sub_path)
sample_sub = pd.read_csv(sample_sub_path)

logging.info("train_df shape: %s", train_df.shape)
logging.info("test_df shape: %s", test_df.shape)
logging.info("titles_df shape: %s", titles_df.shape)
logging.info("sample_submission shape: %s", sample_sub.shape)
logging.info("train_df columns: %s", list(train_df.columns))
logging.info("test_df columns: %s", list(test_df.columns))
logging.info("titles_df columns: %s", list(titles_df.columns))
logging.info("Head of titles_df:\n%s", titles_df.head(3).to_string(index=False))

# =========================
# Merge titles.csv with train/test on context=code
# =========================
title_cols_avail = ["code"]
for c in ["title", "section", "subclass", "class"]:
    if c in titles_df.columns:
        title_cols_avail.append(c)
logging.info("Using title columns: %s", title_cols_avail)

train_merged = train_df.merge(titles_df[title_cols_avail], left_on="context", right_on="code", how="left")
test_merged = test_df.merge(titles_df[title_cols_avail], left_on="context", right_on="code", how="left")
logging.info("train_merged shape: %s", train_merged.shape)
logging.info("test_merged shape: %s", test_merged.shape)

# Coverage checks
contexts_train = set(train_df["context"].unique())
contexts_test = set(test_df["context"].unique())
codes_titles = set(titles_df["code"].unique())
cov_train = len(contexts_train - codes_titles)
cov_test = len(contexts_test - codes_titles)
logging.info("Unique contexts in train: %d", len(contexts_train))
logging.info("Unique contexts in test: %d", len(contexts_test))
logging.info("Unique codes in titles: %d", len(codes_titles))
logging.info("Contexts in train not covered by titles codes: %d", cov_train)
logging.info("Contexts in test not covered by titles codes: %d", cov_test)

if "title" in train_merged.columns:
    missing_titles_train = train_merged["title"].isna().sum()
    logging.info("Missing title count after merge (train): %d", missing_titles_train)
if "title" in test_merged.columns:
    missing_titles_test = test_merged["title"].isna().sum()
    logging.info("Missing title count after merge (test): %d", missing_titles_test)

# =========================
# EDA Queries as requested
# =========================
# 1) Leakage: How many (anchor, target, context) triplets exist in both train and test?
trip_cols = ["anchor", "target", "context"]
train_trip = train_df[trip_cols].copy()
test_trip = test_df[trip_cols].copy()
overlap = train_trip.merge(test_trip, on=trip_cols, how="inner")
logging.info("EDA: Overlapping (anchor, target, context) triplets between train and test: %d", overlap.shape[0])

# 2) Test context distribution: count and percentage top 10
test_context_counts = test_df["context"].value_counts()
test_context_pct = test_context_counts / len(test_df) * 100.0
top10_contexts = test_context_counts.head(10).index.tolist()
logging.info("EDA: Top 10 test context codes by count:")
for code in top10_contexts:
    logging.info("  %s: count=%d, pct=%.2f%%", code, test_context_counts[code], test_context_pct[code])

# 3) For H04, G01, A61: score distribution in train
codes_to_check = ["H04", "G01", "A61"]
score_bins = [0.0, 0.25, 0.5, 0.75, 1.0]
for c in codes_to_check:
    sub = train_df[train_df["context"] == c]
    dist = sub["score"].value_counts().reindex(score_bins, fill_value=0).astype(int)
    logging.info("EDA: Score distribution for context %s:", c)
    for s in score_bins:
        logging.info("  score=%.2f -> count=%d", s, dist[s])

# 4) Max target length in test
max_target_chars_test = test_df["target"].astype(str).str.len().max()
logging.info("EDA: Maximum target character count in test.csv: %d", max_target_chars_test)

# Extra: Length check on train
max_target_chars_train = train_df["target"].astype(str).str.len().max()
logging.info("EDA: Maximum target character count in train.csv: %d", max_target_chars_train)
avg_anchor_chars_train = train_df["anchor"].astype(str).str.len().mean()
avg_target_chars_train = train_df["target"].astype(str).str.len().mean()
logging.info("EDA: Average anchor char len (train): %.2f", avg_anchor_chars_train)
logging.info("EDA: Average target char len (train): %.2f", avg_target_chars_train)

# =========================
# Text Assembly
# =========================
def build_input_texts(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    # Build pair texts (text_a, text_b)
    logging.info("Building text inputs with context enrichment...")
    use_cols = []
    if CFG.use_context_title and "title" in df.columns:
        use_cols.append("title")
    if CFG.use_context_section and "section" in df.columns:
        use_cols.append("section")
    if CFG.use_context_subclass and "subclass" in df.columns:
        use_cols.append("subclass")
    logging.info("Context enrichment columns used: %s", use_cols)

    text_a_list = []
    text_b_list = []
    for i in range(len(df)):
        row = df.iloc[i]
        ctx_parts = [f"{CFG.text_prefix_context} {row['context']}"]
        if "title" in use_cols:
            ctx_parts.append(f"{CFG.text_prefix_title} {str(row['title'])}")
        if "section" in use_cols:
            ctx_parts.append(f"{CFG.text_prefix_section} {str(row['section'])}")
        if "subclass" in use_cols:
            ctx_parts.append(f"{CFG.text_prefix_subclass} {str(row['subclass'])}")

        ctx_str = " | ".join(ctx_parts)
        text_a = f"{CFG.text_prefix_anchor} {row['anchor']} | {ctx_str}"
        text_b = f"{CFG.text_prefix_target} {row['target']}"
        text_a_list.append(text_a)
        text_b_list.append(text_b)
        if i < 2:
            logging.info("Sample built text_a[%d]: %s", i, text_a)
            logging.info("Sample built text_b[%d]: %s", i, text_b)
    logging.info("Completed building %d input text pairs.", len(text_a_list))
    return text_a_list, text_b_list

train_text_a, train_text_b = build_input_texts(train_merged)
test_text_a, test_text_b = build_input_texts(test_merged)

# =========================
# Tokenizer
# =========================
logging.info("Loading tokenizer from %s", CFG.model_name)
tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
logging.info("Tokenizer loaded: %s", tokenizer.__class__.__name__)

# Pre-tokenize for speed, dynamic padding in collator
logging.info("Tokenizing train texts with max_length=%d", CFG.max_len)
train_encodings = tokenizer(
    train_text_a,
    train_text_b,
    truncation=True,
    max_length=CFG.max_len,
    padding=False,
)
logging.info("Tokenizing test texts with max_length=%d", CFG.max_len)
test_encodings = tokenizer(
    test_text_a,
    test_text_b,
    truncation=True,
    max_length=CFG.max_len,
    padding=False,
)

# =========================
# Dataset
# =========================
class PatentPairsDataset(Dataset):
    def __init__(self, encodings: Dict[str, List[int]], labels: np.ndarray = None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

train_labels = train_df["score"].values.astype(np.float32)
train_dataset = PatentPairsDataset(train_encodings, train_labels)
test_dataset = PatentPairsDataset(test_encodings, None)

# =========================
# Model
# =========================
class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        mean_pooled = summed / counts
        return mean_pooled

class PatentRegressor(nn.Module):
    def __init__(self, model_name: str, pooling: str = "mean"):
        super().__init__()
        logging.info("Initializing PatentRegressor with backbone: %s", model_name)
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=False)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        self.pooling_type = pooling
        if pooling == "mean":
            self.pool = MeanPooling()
        else:
            self.pool = None
        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden_size, 1)
        logging.info("Backbone hidden size: %d", hidden_size)
        logging.info("Pooling: %s", pooling)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_state = outputs.last_hidden_state
        if self.pooling_type == "mean":
            pooled = self.pool(last_hidden_state, attention_mask)
        else:
            pooled = last_hidden_state[:, 0]  # fallback if other pooling selected
        x = self.dropout(pooled)
        logits = self.fc(x)
        return logits.squeeze(-1)

# =========================
# Utils
# =========================
def prepare_dataloaders(train_idx: np.ndarray, valid_idx: np.ndarray) -> Tuple[DataLoader, DataLoader]:
    logging.info("Preparing DataLoaders: %d train samples, %d valid samples", len(train_idx), len(valid_idx))
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=CFG.pad_to_multiple_of)
    train_loader = DataLoader(
        Subset(train_dataset, train_idx),
        batch_size=CFG.train_bs,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collator,
        drop_last=False,
    )
    valid_loader = DataLoader(
        Subset(train_dataset, valid_idx),
        batch_size=CFG.valid_bs,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collator,
        drop_last=False,
    )
    return train_loader, valid_loader

def get_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    logging.info("Preparing optimizer (AdamW) with weight decay=%f and lr=%f", CFG.weight_decay, CFG.lr)
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "LayerNorm.bias", "layer_norm.bias"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": CFG.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=CFG.lr)
    return optimizer

def get_scheduler(optimizer: torch.optim.Optimizer, num_train_steps: int):
    warmup_steps = int(CFG.warmup_ratio * num_train_steps)
    logging.info("Preparing cosine scheduler: total steps=%d, warmup steps=%d", num_train_steps, warmup_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=num_train_steps
    )
    return scheduler

def compute_pearsonr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    y_true_mean = y_true.mean()
    y_pred_mean = y_pred.mean()
    num = np.sum((y_true - y_true_mean) * (y_pred - y_pred_mean))
    den = math.sqrt(np.sum((y_true - y_true_mean) ** 2) * np.sum((y_pred - y_pred_mean) ** 2))
    if den == 0:
        return 0.0
    return float(num / den)

# =========================
# Stratified K-Fold setup
# =========================
y = train_df["score"].values
y_bins = np.floor(y * 4).astype(int)  # bins: 0->0.0, 1->0.25, 2->0.5, 3->0.75, 4->1.0
skf = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)
logging.info("Prepared StratifiedKFold with %d folds.", CFG.n_folds)

# =========================
# Training and Validation per-fold
# =========================
oof_pred = np.zeros(len(train_df), dtype=np.float32)
test_pred_folds = []

fold_idx = 0
for tr_idx, va_idx in skf.split(np.zeros(len(y_bins)), y_bins):
    fold_idx += 1
    logging.info("=" * 80)
    logging.info("Starting Fold %d/%d", fold_idx, CFG.n_folds)
    logging.info("Fold train size: %d | Fold valid size: %d", len(tr_idx), len(va_idx))

    train_loader, valid_loader = prepare_dataloaders(tr_idx, va_idx)

    model = PatentRegressor(CFG.model_name, pooling=CFG.pooling)
    model.to(device)

    optimizer = get_optimizer(model)
    num_train_steps = len(train_loader) * CFG.epochs // CFG.grad_accum_steps
    scheduler = get_scheduler(optimizer, num_train_steps=num_train_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=CFG.use_fp16)
    mse_loss = nn.MSELoss()

    global_step = 0
    for epoch in range(CFG.epochs):
        logging.info("Fold %d | Epoch %d/%d - Training start", fold_idx, epoch + 1, CFG.epochs)
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            token_type_ids = batch.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=CFG.use_fp16, dtype=torch.float16):
                preds = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                loss = mse_loss(preds, labels)

            if CFG.grad_accum_steps > 1:
                loss = loss / CFG.grad_accum_steps

            scaler.scale(loss).backward()
            epoch_loss += loss.item()

            if (step + 1) % CFG.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            if (step + 1) % CFG.log_interval == 0:
                logging.info(
                    "Fold %d | Epoch %d | Step %d/%d | Loss: %.6f | LR: %.8f",
                    fold_idx,
                    epoch + 1,
                    step + 1,
                    len(train_loader),
                    loss.item(),
                    scheduler.get_last_lr()[0],
                )

        elapsed = time.time() - start_time
        avg_train_loss = epoch_loss / len(train_loader)
        logging.info("Fold %d | Epoch %d completed in %.2fs | Avg Train Loss: %.6f",
                     fold_idx, epoch + 1, elapsed, avg_train_loss)

    # Validation
    logging.info("Fold %d | Validation start", fold_idx)
    model.eval()
    valid_preds = []
    valid_labels = []
    with torch.no_grad():
        for step, batch in enumerate(valid_loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            token_type_ids = batch.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device, non_blocking=True)
            labels = batch["labels"].cpu().numpy()
            with torch.cuda.amp.autocast(enabled=CFG.use_fp16, dtype=torch.float16):
                preds = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            valid_preds.append(preds.detach().cpu().numpy())
            valid_labels.append(labels)
    valid_preds = np.concatenate(valid_preds).astype(np.float32)
    valid_labels = np.concatenate(valid_labels).astype(np.float32)
    fold_pearson = compute_pearsonr(valid_labels, valid_preds)
    logging.info("Fold %d | Validation Pearson: %.6f", fold_idx, fold_pearson)

    oof_pred[va_idx] = valid_preds

    # Test inference for this fold
    logging.info("Fold %d | Test inference start", fold_idx)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=CFG.pad_to_multiple_of)
    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.valid_bs,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collator,
        drop_last=False,
    )

    model.eval()
    fold_test_preds = []
    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            token_type_ids = batch.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=CFG.use_fp16, dtype=torch.float16):
                preds = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            fold_test_preds.append(preds.detach().cpu().numpy())
    fold_test_preds = np.concatenate(fold_test_preds).astype(np.float32)
    test_pred_folds.append(fold_test_preds)
    logging.info("Fold %d | Test inference completed", fold_idx)

# Overall OOF Pearson
oof_score = compute_pearsonr(train_labels, oof_pred)
logging.info("FINAL OOF PEARSON: %.6f", oof_score)
logging.info("Logging final validation results complete.")

# =========================
# Submission
# =========================
logging.info("Preparing submission by averaging %d folds", len(test_pred_folds))
test_pred_mean = np.mean(np.vstack(test_pred_folds), axis=0)
test_pred_mean = np.clip(test_pred_mean, 0.0, 1.0)
submission = pd.DataFrame({"id": test_df["id"].values, "score": test_pred_mean})
submission.to_csv(SUB_PATH, index=False)
logging.info("Saved submission to %s with shape %s", SUB_PATH, submission.shape)

# =========================
# Additional Diagnostics
# =========================
# Show a few prediction samples
for i in range(3):
    logging.info("Sample submission row %d: id=%s, score=%.5f", i, submission.iloc[i]["id"], submission.iloc[i]["score"])

logging.info("Script completed successfully.")