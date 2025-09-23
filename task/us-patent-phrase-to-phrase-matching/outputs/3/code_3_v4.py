import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import math
import time
import random
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
LOG_PATH = os.path.join(OUTPUT_DIR, "code_3_v4.txt")
SUB_PATH = os.path.join(OUTPUT_DIR, "submission_4.csv")

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
    # Model and tokenizer
    model_name: str = "microsoft/deberta-v3-xsmall"  # small and fast
    max_len: int = 96  # allow longer CPC titles to reduce truncation
    use_fast_tokenizer: bool = True
    # Optimization
    epochs: int = 2
    train_bs: int = 64
    valid_bs: int = 128
    base_lr: float = 2e-5           # base LR for backbone bottom layers
    head_lr: float = 1e-3           # higher LR for regression head
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    scheduler: str = "cosine"
    # Mixed Precision (always fp16)
    use_fp16: bool = True
    # Pooling and head
    pooling: str = "mean_last_n"    # mean pooling over tokens of last N hidden layers
    last_n_layers: int = 4
    mlp_hidden: int = 256
    dropout: float = 0.2
    # LLRD
    llrd: bool = True
    llrd_decay: float = 0.9
    # Regularization / Loss
    use_smooth_l1: bool = True
    smooth_l1_beta: float = 0.1
    use_class_weights: bool = True  # handle score imbalance via weighted loss
    # Data Augmentation
    augment_swap: bool = True       # augment training by swapping anchor/target
    # Test-time augmentation
    tta_swap: bool = True
    # Tokenization / Collation
    pad_to_multiple_of: int = 8
    # Input formatting
    text_prefix_anchor: str = "anchor:"
    text_prefix_target: str = "target:"
    text_prefix_context: str = "context:"
    text_prefix_title: str = "title:"
    text_prefix_section: str = "section:"
    text_prefix_class: str = "class:"
    text_prefix_subclass: str = "subclass:"
    use_context_title: bool = True
    use_context_section: bool = True
    use_context_class: bool = True
    use_context_subclass: bool = True
    # Gradient checkpointing toggle (disable to avoid double-backward issues)
    grad_checkpointing: bool = False
    # CV
    n_folds: int = 5
    seed: int = 42
    # Logging
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
# EDA Queries (from plan)
# =========================
trip_cols = ["anchor", "target", "context"]
train_trip = train_df[trip_cols].copy()
test_trip = test_df[trip_cols].copy()
overlap = train_trip.merge(test_trip, on=trip_cols, how="inner")
logging.info("EDA: Overlapping (anchor, target, context) triplets between train and test: %d", overlap.shape[0])

test_context_counts = test_df["context"].value_counts()
test_context_pct = test_context_counts / len(test_df) * 100.0
top10_contexts = test_context_counts.head(10).index.tolist()
logging.info("EDA: Top 10 test context codes by count:")
for code in top10_contexts:
    logging.info("  %s: count=%d, pct=%.2f%%", code, test_context_counts[code], test_context_pct[code])

codes_to_check = ["H04", "G01", "A61"]
score_bins_list = [0.0, 0.25, 0.5, 0.75, 1.0]
for c in codes_to_check:
    sub = train_df[train_df["context"] == c]
    dist = sub["score"].value_counts().reindex(score_bins_list, fill_value=0).astype(int)
    logging.info("EDA: Score distribution for context %s:", c)
    for s in score_bins_list:
        logging.info("  score=%.2f -> count=%d", s, dist[s])

max_target_chars_test = test_df["target"].astype(str).str.len().max()
logging.info("EDA: Maximum target character count in test.csv: %d", max_target_chars_test)
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
    logging.info("Building text inputs with context enrichment...")
    use_cols = []
    if CFG.use_context_title and "title" in df.columns:
        use_cols.append("title")
    if CFG.use_context_section and "section" in df.columns:
        use_cols.append("section")
    if CFG.use_context_class and "class" in df.columns:
        use_cols.append("class")
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
        if "class" in use_cols:
            ctx_parts.append(f"{CFG.text_prefix_class} {str(row['class'])}")
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

def build_swapped_input_texts(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    logging.info("Building SWAPPED text inputs for augmentation/inference...")
    df_sw = df.rename(columns={"anchor": "target", "target": "anchor"})
    return build_input_texts(df_sw)

# =========================
# Tokenizer
# =========================
logging.info("Loading tokenizer from %s (use_fast=%s)", CFG.model_name, str(CFG.use_fast_tokenizer))
tokenizer = AutoTokenizer.from_pretrained(CFG.model_name, use_fast=CFG.use_fast_tokenizer)
logging.info("Tokenizer loaded: %s", tokenizer.__class__.__name__)

# =========================
# Dataset
# =========================
class PatentPairsDataset(Dataset):
    def __init__(self, encodings: Dict[str, List[int]], labels: Optional[np.ndarray] = None, weights: Optional[np.ndarray] = None):
        self.encodings = encodings
        self.labels = labels
        self.weights = weights

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        if self.weights is not None:
            item["weights"] = torch.tensor(self.weights[idx], dtype=torch.float)
        return item

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
        return summed / counts

class PatentRegressor(nn.Module):
    def __init__(self, model_name: str, pooling: str = "mean_last_n", last_n_layers: int = 4, mlp_hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        logging.info("Initializing PatentRegressor with backbone: %s", model_name)
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        if hasattr(self.config, "use_cache"):
            self.config.use_cache = False
            logging.info("Set config.use_cache=False for compatibility.")
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        if CFG.grad_checkpointing:
            self.backbone.gradient_checkpointing_enable()
            logging.info("Enabled gradient checkpointing.")
        else:
            logging.info("Gradient checkpointing disabled by config.")
        self.pooling_type = pooling
        self.last_n_layers = last_n_layers
        self.token_pool = MeanPooling()
        hidden_size = self.config.hidden_size
        self.hidden_size = hidden_size

        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

        logging.info("Backbone hidden size: %d", hidden_size)
        logging.info("Pooling: %s over last %d layers", pooling, last_n_layers)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        if self.pooling_type == "mean_last_n":
            hidden_states = outputs.hidden_states  # tuple: layer0..last
            selected = hidden_states[-self.last_n_layers:]
            stacked = torch.stack(selected, dim=0)  # [L, B, T, H]
            avg_last = stacked.mean(dim=0)         # [B, T, H]
            pooled = self.token_pool(avg_last, attention_mask)
        elif self.pooling_type == "cls":
            pooled = outputs.last_hidden_state[:, 0]
        else:
            pooled = self.token_pool(outputs.last_hidden_state, attention_mask)

        logits = self.regressor(pooled).squeeze(-1)
        return logits

# =========================
# Utils
# =========================
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

def compute_sample_weights(labels: np.ndarray) -> np.ndarray:
    logging.info("Computing sample weights for training loss balancing.")
    bins = np.floor(labels * 4 + 1e-6).astype(int)  # 0..4
    counts = np.bincount(bins, minlength=5).astype(np.float32)
    inv = 1.0 / np.clip(counts, 1.0, None)
    weights = inv[bins]
    weights = weights / weights.mean()
    logging.info("Class counts per bin [0,0.25,0.5,0.75,1.0]: %s", counts.tolist())
    logging.info("Weights per bin: %s", inv.tolist())
    return weights.astype(np.float32)

def get_optimizer_llrd(model: nn.Module) -> torch.optim.Optimizer:
    logging.info("Preparing LLRD optimizer (AdamW) with base_lr=%.6f, head_lr=%.6f, decay=%.3f, weight_decay=%.6f",
                 CFG.base_lr, CFG.head_lr, CFG.llrd_decay, CFG.weight_decay)
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "LayerNorm.bias", "layer_norm.bias"]

    param_groups = []

    backbone = model.backbone
    num_layers = backbone.config.num_hidden_layers
    logging.info("Backbone num_hidden_layers: %d", num_layers)

    # Embeddings
    emb_params = []
    for n, p in backbone.named_parameters():
        if "embeddings" in n:
            emb_params.append((n, p))
    emb_decay = [p for n, p in emb_params if not any(nd in n for nd in no_decay)]
    emb_nodecay = [p for n, p in emb_params if any(nd in n for nd in no_decay)]
    lr = CFG.base_lr * (CFG.llrd_decay ** (num_layers + 1))
    if len(emb_decay) > 0:
        param_groups.append({"params": emb_decay, "lr": lr, "weight_decay": CFG.weight_decay})
    if len(emb_nodecay) > 0:
        param_groups.append({"params": emb_nodecay, "lr": lr, "weight_decay": 0.0})
    logging.info("Embeddings lr: %.8f", lr)

    # Encoder layers
    for layer_idx in range(num_layers):
        layer_params = []
        for n, p in backbone.named_parameters():
            key = f"encoder.layer.{layer_idx}."
            if key in n:
                layer_params.append((n, p))
        lr = CFG.base_lr * (CFG.llrd_decay ** (num_layers - 1 - layer_idx))
        layer_decay = [p for n, p in layer_params if not any(nd in n for nd in no_decay)]
        layer_nodecay = [p for n, p in layer_params if any(nd in n for nd in no_decay)]
        if len(layer_decay) > 0:
            param_groups.append({"params": layer_decay, "lr": lr, "weight_decay": CFG.weight_decay})
        if len(layer_nodecay) > 0:
            param_groups.append({"params": layer_nodecay, "lr": lr, "weight_decay": 0.0})
        logging.info("Layer %d lr: %.8f (params=%d)", layer_idx, lr, len(layer_params))

    # Other backbone params not captured
    captured = set()
    for g in param_groups:
        for p in g["params"]:
            captured.add(id(p))
    other_backbone_decay = []
    other_backbone_nodecay = []
    for n, p in backbone.named_parameters():
        if id(p) in captured:
            continue
        if any(nd in n for nd in no_decay):
            other_backbone_nodecay.append(p)
        else:
            other_backbone_decay.append(p)
    if len(other_backbone_decay) > 0:
        param_groups.append({"params": other_backbone_decay, "lr": CFG.base_lr, "weight_decay": CFG.weight_decay})
        logging.info("Other backbone decay params: %d, lr: %.8f", len(other_backbone_decay), CFG.base_lr)
    if len(other_backbone_nodecay) > 0:
        param_groups.append({"params": other_backbone_nodecay, "lr": CFG.base_lr, "weight_decay": 0.0})
        logging.info("Other backbone no-decay params: %d, lr: %.8f", len(other_backbone_nodecay), CFG.base_lr)

    # Head params (regressor)
    head_params = []
    for n, p in model.named_parameters():
        if not n.startswith("backbone."):
            head_params.append((n, p))
    head_decay = [p for n, p in head_params if not any(nd in n for nd in no_decay)]
    head_nodecay = [p for n, p in head_params if any(nd in n for nd in no_decay)]
    if len(head_decay) > 0:
        param_groups.append({"params": head_decay, "lr": CFG.head_lr, "weight_decay": CFG.weight_decay})
    if len(head_nodecay) > 0:
        param_groups.append({"params": head_nodecay, "lr": CFG.head_lr, "weight_decay": 0.0})
    logging.info("Head params: %d | head_lr: %.8f", len(head_params), CFG.head_lr)

    optimizer = torch.optim.AdamW(param_groups)
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

# =========================
# Fold-specific data building (with augmentation)
# =========================
def build_fold_texts_and_labels(train_idx: np.ndarray, valid_idx: np.ndarray):
    logging.info("Building fold texts and labels with augmentation=%s", str(CFG.augment_swap))
    tr_df = train_merged.iloc[train_idx].reset_index(drop=True)
    va_df = train_merged.iloc[valid_idx].reset_index(drop=True)

    tr_a_1, tr_b_1 = build_input_texts(tr_df)
    tr_labels = tr_df["score"].values.astype(np.float32)

    if CFG.augment_swap:
        tr_a_2, tr_b_2 = build_swapped_input_texts(tr_df)
        tr_text_a = tr_a_1 + tr_a_2
        tr_text_b = tr_b_1 + tr_b_2
        tr_labels = np.concatenate([tr_labels, tr_labels], axis=0)
        logging.info("Train augmentation doubled samples: %d -> %d", len(tr_a_1), len(tr_text_a))
    else:
        tr_text_a, tr_text_b = tr_a_1, tr_b_1

    va_text_a, va_text_b = build_input_texts(va_df)
    va_labels = va_df["score"].values.astype(np.float32)

    return (tr_text_a, tr_text_b, tr_labels), (va_text_a, va_text_b, va_labels)

def tokenize_text_pairs(text_a: List[str], text_b: List[str]) -> Dict[str, List[int]]:
    enc = tokenizer(
        text_a,
        text_b,
        truncation=True,
        max_length=CFG.max_len,
        padding=False,
    )
    return enc

# =========================
# Training and Validation per-fold
# =========================
def train_one_fold(fold_idx: int, train_idx: np.ndarray, valid_idx: np.ndarray, test_text_a: List[str], test_text_b: List[str], test_text_a_sw: Optional[List[str]], test_text_b_sw: Optional[List[str]]) -> Tuple[np.ndarray, np.ndarray]:
    logging.info("=" * 80)
    logging.info("Starting Fold %d/%d", fold_idx, CFG.n_folds)
    logging.info("Fold train size: %d | Fold valid size: %d", len(train_idx), len(valid_idx))

    (tr_a, tr_b, tr_y), (va_a, va_b, va_y) = build_fold_texts_and_labels(train_idx, valid_idx)

    logging.info("Tokenizing train/valid for fold %d with max_length=%d", fold_idx, CFG.max_len)
    tr_enc = tokenize_text_pairs(tr_a, tr_b)
    va_enc = tokenize_text_pairs(va_a, va_b)

    if CFG.use_class_weights:
        tr_weights = compute_sample_weights(tr_y)
    else:
        tr_weights = np.ones_like(tr_y, dtype=np.float32)

    train_dataset = PatentPairsDataset(tr_enc, tr_y, tr_weights)
    valid_dataset = PatentPairsDataset(va_enc, va_y, None)

    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=CFG.pad_to_multiple_of)
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.train_bs,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collator,
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.valid_bs,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collator,
        drop_last=False,
    )

    model = PatentRegressor(
        CFG.model_name,
        pooling=CFG.pooling,
        last_n_layers=CFG.last_n_layers,
        mlp_hidden=CFG.mlp_hidden,
        dropout=CFG.dropout,
    ).to(device)

    if CFG.llrd:
        optimizer = get_optimizer_llrd(model)
    else:
        logging.info("Preparing optimizer (AdamW) with weight decay=%f and lr=%f", CFG.weight_decay, CFG.base_lr)
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.base_lr, weight_decay=CFG.weight_decay)

    num_train_steps = len(train_loader) * CFG.epochs // CFG.grad_accum_steps
    scheduler = get_scheduler(optimizer, num_train_steps=num_train_steps)

    scaler = torch.amp.GradScaler('cuda', enabled=CFG.use_fp16)
    if CFG.use_smooth_l1:
        criterion = nn.SmoothL1Loss(beta=CFG.smooth_l1_beta, reduction="none")
        logging.info("Using SmoothL1Loss with beta=%.4f", CFG.smooth_l1_beta)
    else:
        criterion = nn.MSELoss(reduction="none")
        logging.info("Using MSELoss")

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
            weights = batch["weights"].to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=CFG.use_fp16, dtype=torch.float16):
                preds = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                loss_vec = criterion(preds, labels)
                loss = (loss_vec * weights).mean()

            if CFG.grad_accum_steps > 1:
                loss = loss / CFG.grad_accum_steps

            scaler.scale(loss).backward()
            epoch_loss += loss.item()

            # Unscale before clipping for correct scaling
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)

            if (step + 1) % CFG.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            if (step + 1) % CFG.log_interval == 0:
                lrs = [pg["lr"] for pg in optimizer.param_groups]
                logging.info(
                    "Fold %d | Epoch %d | Step %d/%d | Loss: %.6f | LR(head/base): %.8f / %.8f",
                    fold_idx,
                    epoch + 1,
                    step + 1,
                    len(train_loader),
                    loss.item(),
                    max(lrs),
                    min(lrs),
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
            with torch.amp.autocast('cuda', enabled=CFG.use_fp16, dtype=torch.float16):
                preds = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            valid_preds.append(preds.detach().cpu().numpy())
            valid_labels.append(labels)
    valid_preds = np.concatenate(valid_preds).astype(np.float32)
    valid_labels = np.concatenate(valid_labels).astype(np.float32)
    fold_pearson = compute_pearsonr(valid_labels, valid_preds)
    logging.info("Fold %d | Validation Pearson: %.6f", fold_idx, fold_pearson)

    # Test inference for this fold with optional TTA (swap)
    logging.info("Fold %d | Tokenizing test set", fold_idx)
    test_enc = tokenize_text_pairs(test_text_a, test_text_b)
    test_dataset = PatentPairsDataset(test_enc, None, None)
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

    logging.info("Fold %d | Test inference start", fold_idx)
    model.eval()
    fold_test_preds = []
    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            token_type_ids = batch.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=CFG.use_fp16, dtype=torch.float16):
                preds = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            fold_test_preds.append(preds.detach().cpu().numpy())
    fold_test_preds = np.concatenate(fold_test_preds).astype(np.float32)

    if CFG.tta_swap and test_text_a_sw is not None and test_text_b_sw is not None:
        logging.info("Fold %d | TTA swap enabled: running swapped test inference", fold_idx)
        test_enc_sw = tokenize_text_pairs(test_text_a_sw, test_text_b_sw)
        test_dataset_sw = PatentPairsDataset(test_enc_sw, None, None)
        test_loader_sw = DataLoader(
            test_dataset_sw,
            batch_size=CFG.valid_bs,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=collator,
            drop_last=False,
        )
        fold_test_preds_sw = []
        with torch.no_grad():
            for step, batch in enumerate(test_loader_sw):
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                token_type_ids = batch.get("token_type_ids", None)
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=CFG.use_fp16, dtype=torch.float16):
                    preds = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                fold_test_preds_sw.append(preds.detach().cpu().numpy())
        fold_test_preds_sw = np.concatenate(fold_test_preds_sw).astype(np.float32)
        fold_test_preds = 0.5 * (fold_test_preds + fold_test_preds_sw)
        logging.info("Fold %d | TTA averaging completed", fold_idx)

    logging.info("Fold %d | Test inference completed", fold_idx)

    return valid_preds, fold_test_preds

# =========================
# Build test texts once (+ swapped for TTA)
# =========================
test_text_a, test_text_b = build_input_texts(test_merged)
if CFG.tta_swap:
    test_text_a_sw, test_text_b_sw = build_swapped_input_texts(test_merged)
else:
    test_text_a_sw, test_text_b_sw = None, None

# =========================
# Stratified K-Fold setup
# =========================
y = train_df["score"].values
y_bins = np.floor(y * 4).astype(int)  # bins: 0->0.0, 1->0.25, 2->0.5, 3->0.75, 4->1.0
skf = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)
logging.info("Prepared StratifiedKFold with %d folds.", CFG.n_folds)

# =========================
# Run folds
# =========================
oof_pred = np.zeros(len(train_df), dtype=np.float32)
test_pred_folds = []

fold_counter = 0
for tr_idx, va_idx in skf.split(np.zeros(len(y_bins)), y_bins):
    fold_counter += 1
    valid_preds, test_preds = train_one_fold(fold_counter, tr_idx, va_idx, test_text_a, test_text_b, test_text_a_sw, test_text_b_sw)
    oof_pred[va_idx] = valid_preds
    test_pred_folds.append(test_preds)

# Overall OOF Pearson
oof_score = compute_pearsonr(train_df["score"].values.astype(np.float32), oof_pred)
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

# Diagnostics
for i in range(3):
    logging.info("Sample submission row %d: id=%s, score=%.5f", i, submission.iloc[i]["id"], submission.iloc[i]["score"])

logging.info("Script completed successfully.")