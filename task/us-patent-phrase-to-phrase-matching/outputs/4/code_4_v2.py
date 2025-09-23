import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import math
import time
import random
import logging
from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding,
)
from sklearn.model_selection import StratifiedKFold

# -----------------------------------------------------------------------------
# Paths, Config, Logging
# -----------------------------------------------------------------------------
BASE_DIR = "task/us-patent-phrase-to-phrase-matching"
OUT_DIR = os.path.join(BASE_DIR, "outputs", "4")
os.makedirs(OUT_DIR, exist_ok=True)
LOG_FILE = os.path.join(OUT_DIR, "code_4_v2.txt")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),
        logging.StreamHandler()
    ]
)
logging.info("Initialized logging system.")
logging.info(f"Logs will be written to: {LOG_FILE}")

@dataclass
class Config:
    seed: int = 42
    model_name: str = "microsoft/deberta-v3-base"
    max_length: int = 64
    train_bs: int = 32
    valid_bs: int = 64
    epochs: int = 3
    lr: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    num_workers: int = 4
    fold_id: int = 0
    n_splits: int = 5
    gradient_accumulation_steps: int = 1
    fp16: bool = True
    optimizer_eps: float = 1e-8
    max_grad_norm: float = 1.0
    dropout: float = 0.1
    gradient_checkpointing: bool = False  # disabled to avoid backward graph reuse with checkpointing
    train_file: str = os.path.join(BASE_DIR, "train.csv")
    test_file: str = os.path.join(BASE_DIR, "test.csv")
    titles_file: str = os.path.join(BASE_DIR, "titles.csv")
    submission_file: str = os.path.join(OUT_DIR, "submission_2.csv")
    log_interval: int = 100

CFG = Config()
logging.info(f"Config: {asdict(CFG)}")

# -----------------------------------------------------------------------------
# Reproducibility and CUDA
# -----------------------------------------------------------------------------
def seed_all(seed: int):
    logging.info(f"Seeding all RNGs with seed={seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_all(CFG.seed)
assert torch.cuda.is_available(), "CUDA must be available."
device = torch.device("cuda")
logging.info(f"Using device: {device}, CUDA device count: {torch.cuda.device_count()}")
logging.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")

# -----------------------------------------------------------------------------
# Load data and enrich with CPC titles
# -----------------------------------------------------------------------------
logging.info("Loading CSV files.")
df_train = pd.read_csv(CFG.train_file)
df_test = pd.read_csv(CFG.test_file)
df_titles = pd.read_csv(CFG.titles_file)

logging.info(f"Train shape: {df_train.shape}, columns: {list(df_train.columns)}")
logging.info(f"Test shape: {df_test.shape}, columns: {list(df_test.columns)}")
logging.info(f"Titles shape: {df_titles.shape}, columns: {list(df_titles.columns)}")

expected_title_cols = ["code", "title", "section", "class", "subclass", "group", "subgroup"]
for col in expected_title_cols:
    if col not in df_titles.columns:
        logging.info(f"Column '{col}' not found in titles; creating empty column.")
        df_titles[col] = ""

logging.info("Merging CPC titles into train and test on context->code.")
df_train = df_train.merge(df_titles[expected_title_cols], left_on="context", right_on="code", how="left")
df_test = df_test.merge(df_titles[expected_title_cols], left_on="context", right_on="code", how="left")

for col in expected_title_cols:
    if col != "code":
        df_train[col] = df_train[col].fillna("")
        df_test[col] = df_test[col].fillna("")

logging.info(f"Post-merge train shape: {df_train.shape}, columns now: {list(df_train.columns)}")
logging.info(f"Post-merge test shape: {df_test.shape}, columns now: {list(df_test.columns)}")

# -----------------------------------------------------------------------------
# Research plan EDA checks
# -----------------------------------------------------------------------------
logging.info("Checking for train-test leakage (overlap of (anchor, target, context)).")
train_keys = df_train[["anchor", "target", "context"]].copy()
test_keys = df_test[["anchor", "target", "context"]].copy()
train_keys["triple"] = train_keys["anchor"].astype(str) + " | " + train_keys["target"].astype(str) + " | " + train_keys["context"].astype(str)
test_keys["triple"] = test_keys["anchor"].astype(str) + " | " + test_keys["target"].astype(str) + " | " + test_keys["context"].astype(str)
overlap = pd.Series(np.intersect1d(train_keys["triple"].values, test_keys["triple"].values))
logging.info(f"Number of overlapping (anchor, target, context) triplets: {len(overlap)}")

logging.info("Computing test context distribution (top 10).")
test_context_counts = df_test["context"].value_counts()
test_total = len(df_test)
for ctx, cnt in test_context_counts.head(10).items():
    pct = 100.0 * cnt / test_total
    logging.info(f"Test context {ctx}: count={cnt}, pct={pct:.2f}%")

logging.info("Score distribution per context for H04, G01, A61.")
for ctx in ["H04", "G01", "A61"]:
    sub = df_train[df_train["context"] == ctx]
    if len(sub) == 0:
        logging.info(f"Context {ctx}: no rows in train.")
    else:
        counts = sub["score"].value_counts().sort_index()
        logging.info(f"Context {ctx}: total={len(sub)}")
        for score_val, cnt in counts.items():
            logging.info(f"  score={score_val}: count={cnt}")

max_target_len_test = df_test["target"].astype(str).str.len().max()
logging.info(f"Maximum target character length in test.csv: {max_target_len_test}")

logging.info("Checking mapping coverage of contexts to titles codes.")
train_contexts = set(df_train["context"].unique().tolist())
test_contexts = set(df_test["context"].unique().tolist())
title_codes = set(df_titles["code"].astype(str).unique().tolist())
train_covered = len(train_contexts - title_codes) == 0
test_covered = len(test_contexts - title_codes) == 0
logging.info(f"All train contexts covered by titles: {train_covered}")
logging.info(f"All test contexts covered by titles: {test_covered}")

# -----------------------------------------------------------------------------
# Input text builder
# -----------------------------------------------------------------------------
def build_input_text(row: pd.Series) -> str:
    cpc_parts = [
        f"code: {str(row['context'])}",
        f"title: {str(row['title'])}" if 'title' in row and pd.notna(row['title']) else "title: ",
        f"section: {str(row['section'])}" if 'section' in row and pd.notna(row['section']) else "section: ",
        f"class: {str(row['class'])}" if 'class' in row and pd.notna(row['class']) else "class: ",
        f"subclass: {str(row['subclass'])}" if 'subclass' in row and pd.notna(row['subclass']) else "subclass: ",
    ]
    cpc_str = " | ".join(cpc_parts)
    return f"anchor: {str(row['anchor'])} [SEP] target: {str(row['target'])} [SEP] context: {cpc_str}"

logging.info("Building input_text fields for train and test.")
df_train["input_text"] = df_train.apply(build_input_text, axis=1)
df_test["input_text"] = df_test.apply(build_input_text, axis=1)

logging.info("Sample train input_text:")
for i in range(min(3, len(df_train))):
    logging.info(df_train["input_text"].iloc[i])

# -----------------------------------------------------------------------------
# Stratified split (single fold for early iteration)
# -----------------------------------------------------------------------------
def make_bins(values: np.ndarray, n_bins: int = 5) -> np.ndarray:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    inds = np.digitize(values, bins, right=True)
    inds = np.clip(inds, 1, n_bins) - 1
    return inds

logging.info("Preparing stratified split (single fold).")
y_all = df_train["score"].astype(float).values
bins = make_bins(y_all, n_bins=5)
skf = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)
folds = list(skf.split(np.zeros(len(df_train)), bins))
train_idx, valid_idx = folds[CFG.fold_id]
logging.info(f"Fold {CFG.fold_id} sizes -> train: {len(train_idx)}, valid: {len(valid_idx)}")

train_df = df_train.iloc[train_idx].reset_index(drop=True)
valid_df = df_train.iloc[valid_idx].reset_index(drop=True)

# -----------------------------------------------------------------------------
# Tokenizer, Dataset, DataLoaders
# -----------------------------------------------------------------------------
logging.info("Loading tokenizer.")
tokenizer = AutoTokenizer.from_pretrained(CFG.model_name, use_fast=True)
tokenizer.model_max_length = CFG.max_length

class PatentPhraseDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: AutoTokenizer, max_length: int, with_labels: bool = True):
        self.texts = df["input_text"].astype(str).tolist()
        self.with_labels = with_labels
        self.labels = df["score"].astype(np.float32).to_numpy() if with_labels and "score" in df.columns else None
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True
        )
        if self.with_labels:
            enc["labels"] = float(self.labels[idx])
        return enc

logging.info("Creating datasets and dataloaders.")
train_dataset = PatentPhraseDataset(train_df, tokenizer, CFG.max_length, with_labels=True)
valid_dataset = PatentPhraseDataset(valid_df, tokenizer, CFG.max_length, with_labels=True)
test_dataset = PatentPhraseDataset(df_test, tokenizer, CFG.max_length, with_labels=False)

collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

train_loader = DataLoader(
    train_dataset,
    batch_size=CFG.train_bs,
    shuffle=True,
    num_workers=CFG.num_workers,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=collator
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=CFG.valid_bs,
    shuffle=False,
    num_workers=CFG.num_workers,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=collator
)
test_loader = DataLoader(
    test_dataset,
    batch_size=CFG.valid_bs,
    shuffle=False,
    num_workers=CFG.num_workers,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=collator
)
logging.info(f"Train batches: {len(train_loader)}, Valid batches: {len(valid_loader)}, Test batches: {len(test_loader)}")

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
logging.info("Loading pretrained model.")
config = AutoConfig.from_pretrained(CFG.model_name)
config.num_labels = 1
config.problem_type = "regression"
if hasattr(config, "hidden_dropout_prob"):
    config.hidden_dropout_prob = CFG.dropout
if hasattr(config, "attention_probs_dropout_prob"):
    config.attention_probs_dropout_prob = CFG.dropout
if hasattr(config, "use_cache"):
    config.use_cache = False

model = AutoModelForSequenceClassification.from_pretrained(CFG.model_name, config=config)
if CFG.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()
else:
    logging.info("Gradient checkpointing is disabled in this run to ensure a single-backward-per-forward behavior.")

model.to(device)
logging.info(f"Model loaded: {CFG.model_name}")
logging.info(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
logging.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# -----------------------------------------------------------------------------
# Optimizer, Scheduler, Scaler
# -----------------------------------------------------------------------------
def get_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    logging.info("Building optimizer (AdamW) with decoupled weight decay.")
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": CFG.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(optimizer_grouped_parameters, lr=CFG.lr, eps=CFG.optimizer_eps)

num_update_steps_per_epoch = math.ceil(len(train_loader) / CFG.gradient_accumulation_steps)
num_training_steps = CFG.epochs * num_update_steps_per_epoch
num_warmup_steps = int(CFG.warmup_ratio * num_training_steps)
logging.info(f"Training steps: {num_training_steps}, Warmup steps: {num_warmup_steps}")

optimizer = get_optimizer(model)
scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
)
scaler = torch.amp.GradScaler('cuda', enabled=CFG.fp16)
criterion = nn.MSELoss(reduction="mean")

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def to_device(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out

def compute_pearsonr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
logging.info("Starting training with CUDA fp16.")
best_val_pearson = -1.0
best_epoch = -1
history = []
global_step = 0

for epoch in range(CFG.epochs):
    model.train()
    epoch_loss = 0.0
    t0 = time.time()

    for step, batch in enumerate(train_loader):
        batch = to_device(batch)
        labels = batch.pop("labels").float()

        with torch.amp.autocast('cuda', enabled=CFG.fp16):
            outputs = model(**batch)
            logits = outputs.logits.view(-1)
            loss = criterion(logits, labels)
            loss = loss / CFG.gradient_accumulation_steps

        scaler.scale(loss).backward()

        epoch_loss += (loss.item() * CFG.gradient_accumulation_steps)

        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1

        if (step + 1) % CFG.log_interval == 0:
            logging.info(
                f"Epoch {epoch+1}/{CFG.epochs} | Step {step+1}/{len(train_loader)} | "
                f"Loss: {loss.item()*CFG.gradient_accumulation_steps:.6f} | "
                f"LR: {scheduler.get_last_lr()[0]:.8f}"
            )

    avg_train_loss = epoch_loss / len(train_loader)
    elapsed = time.time() - t0
    logging.info(f"Epoch {epoch+1} done. Avg train loss: {avg_train_loss:.6f} | Time: {elapsed:.2f}s")

    # Validation
    model.eval()
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for batch in valid_loader:
            batch = to_device(batch)
            labels = batch.pop("labels").float()
            with torch.amp.autocast('cuda', enabled=CFG.fp16):
                outputs = model(**batch)
                logits = outputs.logits.view(-1)
            val_preds.append(logits.detach().float().cpu().numpy())
            val_targets.append(labels.detach().float().cpu().numpy())
    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)
    val_mse = float(np.mean((val_preds - val_targets) ** 2))
    val_pearson = compute_pearsonr(val_targets, val_preds)
    history.append({"epoch": epoch+1, "train_loss": avg_train_loss, "val_mse": val_mse, "val_pearson": val_pearson})
    logging.info(f"Validation epoch {epoch+1}: MSE={val_mse:.6f}, Pearson={val_pearson:.6f}")

    if val_pearson > best_val_pearson:
        best_val_pearson = val_pearson
        best_epoch = epoch + 1
        logging.info(f"New best validation Pearson: {best_val_pearson:.6f} at epoch {best_epoch}")

logging.info(f"Final best validation Pearson correlation: {best_val_pearson:.6f} (epoch {best_epoch})")
logging.info("Validation history:")
for rec in history:
    logging.info(f"  Epoch {rec['epoch']}: train_loss={rec['train_loss']:.6f}, val_mse={rec['val_mse']:.6f}, val_pearson={rec['val_pearson']:.6f}")

# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------
logging.info("Running final validation inference for logging.")
model.eval()
val_preds = []
val_targets = []
with torch.no_grad():
    for batch in valid_loader:
        batch = to_device(batch)
        labels = batch.pop("labels").float()
        with torch.amp.autocast('cuda', enabled=CFG.fp16):
            outputs = model(**batch)
            logits = outputs.logits.view(-1)
        val_preds.append(logits.detach().float().cpu().numpy())
        val_targets.append(labels.detach().float().cpu().numpy())
val_preds = np.concatenate(val_preds)
val_targets = np.concatenate(val_targets)
val_mse_final = float(np.mean((val_preds - val_targets) ** 2))
val_pearson_final = compute_pearsonr(val_targets, val_preds)
logging.info(f"Final Validation Results -> MSE: {val_mse_final:.6f}, Pearson: {val_pearson_final:.6f}")

logging.info("Running inference on test set.")
test_preds = []
with torch.no_grad():
    for batch in test_loader:
        batch = to_device(batch)
        with torch.amp.autocast('cuda', enabled=CFG.fp16):
            outputs = model(**batch)
            logits = outputs.logits.view(-1)
        test_preds.append(logits.detach().float().cpu().numpy())
test_preds = np.concatenate(test_preds)
test_preds = np.clip(test_preds, 0.0, 1.0)

# -----------------------------------------------------------------------------
# Save submission
# -----------------------------------------------------------------------------
logging.info("Preparing submission file.")
submission = pd.DataFrame({
    "id": df_test["id"],
    "score": test_preds
})
submission.to_csv(CFG.submission_file, index=False)
logging.info(f"Submission saved to: {CFG.submission_file}")

# -----------------------------------------------------------------------------
# Extra analytics
# -----------------------------------------------------------------------------
logging.info("Extra analytics and diagnostics.")
train_score_counts = df_train["score"].value_counts().sort_index()
for s, c in train_score_counts.items():
    logging.info(f"Train score={s}: count={c}, pct={100.0*c/len(df_train):.2f}%")

logging.info("Token length diagnostics (first 100 train samples).")
lens = []
for i in range(min(100, len(df_train))):
    enc = tokenizer(df_train["input_text"].iloc[i], truncation=True, max_length=CFG.max_length, add_special_tokens=True)
    lens.append(len(enc["input_ids"]))
lens = np.array(lens)
logging.info(f"Token length stats -> mean={lens.mean():.2f}, max={lens.max()}, min={lens.min()}")

logging.info("Pipeline completed successfully with CUDA fp16. Submission ready.")