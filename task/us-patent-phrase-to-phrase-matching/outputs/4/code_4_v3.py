import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import math
import time
import random
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    DataCollatorWithPadding,
    get_cosine_schedule_with_warmup,
)
from sklearn.model_selection import StratifiedKFold

# =============================================================================
# Paths, Config, Logging
# =============================================================================
BASE_DIR = "task/us-patent-phrase-to-phrase-matching"
OUT_DIR = os.path.join(BASE_DIR, "outputs", "4")
os.makedirs(OUT_DIR, exist_ok=True)
LOG_FILE = os.path.join(OUT_DIR, "code_4_v3.txt")

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
    model_name: str = "microsoft/deberta-v3-large"  # stronger cross-encoder
    max_length: int = 192  # longer context
    train_bs: int = 16
    valid_bs: int = 32
    epochs: int = 3
    lr: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    num_workers: int = 4
    fold_id: int = 0  # single fold for this iteration
    n_splits: int = 5
    gradient_accumulation_steps: int = 1
    fp16: bool = True
    optimizer_eps: float = 1e-8
    max_grad_norm: float = 1.0
    dropout: float = 0.2
    n_msd: int = 5  # multi-sample dropout passes
    last_n_layers: int = 4  # weighted layer pooling over last N layers
    layer_decay: float = 0.95  # layer-wise LR decay
    gradient_checkpointing: bool = True
    # multi-task loss weights
    lambda_reg: float = 0.5      # direct regression MSE
    lambda_exp: float = 0.25     # expected value MSE from class probs
    lambda_cls: float = 0.25     # KL divergence for class probs
    # blend weights for inference/regression vs class-expected
    blend_alpha: float = 0.5
    train_file: str = os.path.join(BASE_DIR, "train.csv")
    test_file: str = os.path.join(BASE_DIR, "test.csv")
    titles_file: str = os.path.join(BASE_DIR, "titles.csv")
    submission_file: str = os.path.join(OUT_DIR, "submission_3.csv")
    log_interval: int = 100

CFG = Config()
logging.info(f"Config: {asdict(CFG)}")

# =============================================================================
# Reproducibility and CUDA
# =============================================================================
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

# =============================================================================
# Load data and enrich with CPC titles
# =============================================================================
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

# =============================================================================
# Research plan EDA checks (and logging)
# =============================================================================
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

# =============================================================================
# Build sentence pairs: prepend context metadata to anchor as first sequence
# =============================================================================
def build_pair_texts(row: pd.Series) -> Tuple[str, str]:
    cpc_parts = [
        f"code: {str(row['context'])}",
        f"title: {str(row['title'])}" if 'title' in row and pd.notna(row['title']) else "title: ",
        f"section: {str(row['section'])}" if 'section' in row and pd.notna(row['section']) else "section: ",
        f"class: {str(row['class'])}" if 'class' in row and pd.notna(row['class']) else "class: ",
        f"subclass: {str(row['subclass'])}" if 'subclass' in row and pd.notna(row['subclass']) else "subclass: ",
    ]
    cpc_str = " | ".join(cpc_parts)
    first_seq = f"context: {cpc_str} [CTX] anchor: {str(row['anchor'])}"
    second_seq = f"target: {str(row['target'])}"
    return first_seq, second_seq

logging.info("Building sentence pairs with CPC context prepended.")
pairs_train = df_train.apply(build_pair_texts, axis=1)
df_train["first_seq"] = pairs_train.map(lambda x: x[0])
df_train["second_seq"] = pairs_train.map(lambda x: x[1])
pairs_test = df_test.apply(build_pair_texts, axis=1)
df_test["first_seq"] = pairs_test.map(lambda x: x[0])
df_test["second_seq"] = pairs_test.map(lambda x: x[1])

logging.info("Sample train pairs:")
for i in range(min(3, len(df_train))):
    logging.info(f"FIRST: {df_train['first_seq'].iloc[i]}")
    logging.info(f"SECOND: {df_train['second_seq'].iloc[i]}")

# =============================================================================
# Stratified split (single fold)
# =============================================================================
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

# =============================================================================
# Tokenizer, Dataset, DataLoaders
# =============================================================================
logging.info("Loading tokenizer.")
tokenizer = AutoTokenizer.from_pretrained(CFG.model_name, use_fast=True)
tokenizer.model_max_length = CFG.max_length

class PatentPairDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: AutoTokenizer, max_length: int, with_labels: bool = True, swap: bool = False):
        if swap:
            self.first_seq = df["second_seq"].astype(str).tolist()
            self.second_seq = df["first_seq"].astype(str).tolist()
        else:
            self.first_seq = df["first_seq"].astype(str).tolist()
            self.second_seq = df["second_seq"].astype(str).tolist()
        self.with_labels = with_labels
        self.labels = df["score"].astype(np.float32).to_numpy() if with_labels and "score" in df.columns else None
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.first_seq)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.first_seq[idx],
            self.second_seq[idx],
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True
        )
        if self.with_labels:
            enc["labels"] = float(self.labels[idx])
        return enc

logging.info("Creating datasets and dataloaders.")
train_dataset = PatentPairDataset(train_df, tokenizer, CFG.max_length, with_labels=True, swap=False)
valid_dataset = PatentPairDataset(valid_df, tokenizer, CFG.max_length, with_labels=True, swap=False)
test_dataset = PatentPairDataset(df_test, tokenizer, CFG.max_length, with_labels=False, swap=False)
test_dataset_swapped = PatentPairDataset(df_test, tokenizer, CFG.max_length, with_labels=False, swap=True)

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
test_loader_swapped = DataLoader(
    test_dataset_swapped,
    batch_size=CFG.valid_bs,
    shuffle=False,
    num_workers=CFG.num_workers,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=collator
)
logging.info(f"Train batches: {len(train_loader)}, Valid batches: {len(valid_loader)}, Test batches: {len(test_loader)}")

# =============================================================================
# Model with Weighted Layer Pooling + Mean Pooling + Multi-Sample Dropout
# Multi-task: regression head + 5-class classification head
# =============================================================================
class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers: int, last_n_layers: int = 4):
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        self.last_n_layers = last_n_layers
        self.layer_weights = nn.Parameter(torch.ones(last_n_layers, dtype=torch.float32))

    def forward(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        # hidden_states: [emb, layer1, ..., layerN], length = N+1
        selected = hidden_states[-self.last_n_layers:]  # list of tensors [B, T, H]
        stack = torch.stack(selected, dim=0)  # [L, B, T, H]
        norm_w = torch.softmax(self.layer_weights, dim=0).view(self.last_n_layers, 1, 1, 1)
        weighted = (norm_w * stack).sum(dim=0)  # [B, T, H]
        return weighted

def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # token_embeddings: [B, T, H], attention_mask: [B, T]
    input_mask_expanded = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)  # [B, T, 1]
    sum_embeddings = (token_embeddings * input_mask_expanded).sum(dim=1)  # [B, H]
    sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-6)  # [B, 1]
    return sum_embeddings / sum_mask

class CrossEncoderMTL(nn.Module):
    def __init__(self, model_name: str, dropout: float, last_n_layers: int):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()
        self.pooler = WeightedLayerPooling(num_hidden_layers=self.config.num_hidden_layers, last_n_layers=last_n_layers)
        self.hidden_size = self.config.hidden_size
        # multi-sample dropout layers
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(CFG.n_msd)])
        # heads
        self.regressor = nn.Linear(self.hidden_size, 1)
        self.classifier = nn.Linear(self.hidden_size, 5)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs.last_hidden_state  # [B, T, H]
        # combine last layers
        pooled_layers = self.pooler(outputs.hidden_states)  # [B, T, H]
        features = mean_pooling(pooled_layers, attention_mask)  # [B, H]

        reg_logits_accum = 0.0
        cls_logits_accum = 0.0
        for d in self.dropouts:
            x = d(features)
            reg_logits_accum = reg_logits_accum + self.regressor(x)  # [B, 1]
            cls_logits_accum = cls_logits_accum + self.classifier(x)  # [B, 5]
        reg_logits = reg_logits_accum / len(self.dropouts)  # [B, 1]
        cls_logits = cls_logits_accum / len(self.dropouts)  # [B, 5]
        return {"regression": reg_logits.squeeze(-1), "class_logits": cls_logits}

logging.info("Building model with weighted-layer pooling, mean pooling, and multi-sample dropout.")
model = CrossEncoderMTL(CFG.model_name, dropout=CFG.dropout, last_n_layers=CFG.last_n_layers)
if CFG.gradient_checkpointing:
    logging.info("Gradient checkpointing is enabled on backbone.")
else:
    logging.info("Gradient checkpointing requested but disabled in config.")
model.to(device)
logging.info(f"Model loaded: {CFG.model_name}")
logging.info(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
logging.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# =============================================================================
# Optimizer with Layer-wise LR Decay, Cosine schedule, Scaler
# =============================================================================
def get_layer_id_for_deberta(name: str, num_layers: int) -> int:
    # embeddings -> 0, encoder.layer.X -> X+1, heads -> num_layers+1
    if name.startswith("backbone.embeddings"):
        return 0
    if name.startswith("backbone.encoder.layer."):
        parts = name.split(".")
        # ... encoder.layer.{i}.xxx
        idx = parts.index("layer") + 1
        layer_num = int(parts[idx])
        return layer_num + 1
    # heads or others
    return num_layers + 1

def get_optimizer_grouped_parameters(model: nn.Module, base_lr: float, weight_decay: float, layer_decay: float) -> List[Dict]:
    logging.info("Creating optimizer parameter groups with layer-wise LR decay.")
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "layernorm.weight"]
    num_layers = model.config.num_hidden_layers
    param_groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        layer_id = get_layer_id_for_deberta(name, num_layers)
        decay_flag = not any(nd in name for nd in no_decay)
        group_key = (layer_id, decay_flag)
        if group_key not in param_groups:
            lr = base_lr * (layer_decay ** (num_layers + 1 - layer_id))
            wd = weight_decay if decay_flag else 0.0
            param_groups[group_key] = {"params": [], "lr": lr, "weight_decay": wd}
        param_groups[group_key]["params"].append(param)
    groups = list(param_groups.values())
    for i, g in enumerate(groups):
        logging.info(f"Group {i}: params={len(g['params'])}, lr={g['lr']:.8f}, wd={g['weight_decay']}")
    return groups

optimizer = torch.optim.AdamW(
    get_optimizer_grouped_parameters(model, CFG.lr, CFG.weight_decay, CFG.layer_decay),
    lr=CFG.lr,
    eps=CFG.optimizer_eps,
)
num_update_steps_per_epoch = math.ceil(len(train_loader) / CFG.gradient_accumulation_steps)
num_training_steps = CFG.epochs * num_update_steps_per_epoch
num_warmup_steps = int(CFG.warmup_ratio * num_training_steps)
logging.info(f"Training steps: {num_training_steps}, Warmup steps: {num_warmup_steps}")

scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
)
scaler = torch.amp.GradScaler('cuda', enabled=CFG.fp16)

# Losses
mse_loss = nn.MSELoss(reduction="mean")
kldiv_loss = nn.KLDivLoss(reduction="batchmean")

# Constants
CLASS_BINS = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float32, device=device)

# =============================================================================
# Utilities
# =============================================================================
def to_device(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out

def labels_to_soft_targets(labels: torch.Tensor) -> torch.Tensor:
    # triangular distribution over bins with width 0.25
    diffs = torch.abs(labels.unsqueeze(1) - CLASS_BINS.unsqueeze(0))  # [B, 5]
    weights = torch.clamp(1.0 - (diffs / 0.25), min=0.0)  # [B, 5]
    norm = torch.sum(weights, dim=1, keepdim=True).clamp(min=1e-6)
    probs = weights / norm
    return probs

def compute_pearsonr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])

# =============================================================================
# Training
# =============================================================================
logging.info("Starting training with CUDA fp16, multi-task objectives, cosine schedule, and LLRD.")
best_val_pearson = -1.0
best_epoch = -1
history = []
global_step = 0

for epoch in range(CFG.epochs):
    model.train()
    epoch_loss = 0.0
    epoch_reg_loss = 0.0
    epoch_exp_loss = 0.0
    epoch_cls_loss = 0.0
    t0 = time.time()

    for step, batch in enumerate(train_loader):
        batch = to_device(batch)
        labels = batch.pop("labels").float()
        soft_targets = labels_to_soft_targets(labels)

        with torch.amp.autocast('cuda', enabled=CFG.fp16):
            outputs = model(**batch)
            pred_reg = outputs["regression"]          # [B]
            logits_cls = outputs["class_logits"]      # [B, 5]
            probs_cls = torch.softmax(logits_cls, dim=-1)  # [B, 5]
            expected_val = torch.sum(probs_cls * CLASS_BINS, dim=-1)  # [B]

            loss_reg = mse_loss(pred_reg, labels)
            loss_exp = mse_loss(expected_val, labels)
            loss_cls = kldiv_loss(torch.log_softmax(logits_cls, dim=-1), soft_targets)

            loss = CFG.lambda_reg * loss_reg + CFG.lambda_exp * loss_exp + CFG.lambda_cls * loss_cls
            loss = loss / CFG.gradient_accumulation_steps

        scaler.scale(loss).backward()

        epoch_loss += (loss.item() * CFG.gradient_accumulation_steps)
        epoch_reg_loss += loss_reg.item()
        epoch_exp_loss += loss_exp.item()
        epoch_cls_loss += loss_cls.item()

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
                f"Loss: {loss.item()*CFG.gradient_accumulation_steps:.6f} "
                f"(reg={loss_reg.item():.6f}, exp={loss_exp.item():.6f}, cls={loss_cls.item():.6f}) | "
                f"LR: {scheduler.get_last_lr()[0]:.8f}"
            )

    avg_train_loss = epoch_loss / len(train_loader)
    avg_reg = epoch_reg_loss / len(train_loader)
    avg_exp = epoch_exp_loss / len(train_loader)
    avg_cls = epoch_cls_loss / len(train_loader)
    elapsed = time.time() - t0
    logging.info(f"Epoch {epoch+1} done. Avg train loss: {avg_train_loss:.6f} "
                 f"(reg={avg_reg:.6f}, exp={avg_exp:.6f}, cls={avg_cls:.6f}) | Time: {elapsed:.2f}s")

    # Validation
    model.eval()
    val_preds_reg = []
    val_preds_exp = []
    val_targets = []
    with torch.no_grad():
        for batch in valid_loader:
            batch = to_device(batch)
            labels = batch.pop("labels").float()
            with torch.amp.autocast('cuda', enabled=CFG.fp16):
                outputs = model(**batch)
                pred_reg = outputs["regression"]
                logits_cls = outputs["class_logits"]
                probs_cls = torch.softmax(logits_cls, dim=-1)
                expected_val = torch.sum(probs_cls * CLASS_BINS, dim=-1)

            val_preds_reg.append(pred_reg.detach().float().cpu().numpy())
            val_preds_exp.append(expected_val.detach().float().cpu().numpy())
            val_targets.append(labels.detach().float().cpu().numpy())

    val_preds_reg = np.concatenate(val_preds_reg)
    val_preds_exp = np.concatenate(val_preds_exp)
    val_targets = np.concatenate(val_targets)
    val_preds_blend = CFG.blend_alpha * val_preds_reg + (1.0 - CFG.blend_alpha) * val_preds_exp

    val_mse_reg = float(np.mean((val_preds_reg - val_targets) ** 2))
    val_mse_exp = float(np.mean((val_preds_exp - val_targets) ** 2))
    val_mse_blend = float(np.mean((val_preds_blend - val_targets) ** 2))
    val_pearson_reg = compute_pearsonr(val_targets, val_preds_reg)
    val_pearson_exp = compute_pearsonr(val_targets, val_preds_exp)
    val_pearson_blend = compute_pearsonr(val_targets, val_preds_blend)

    history.append({
        "epoch": epoch+1,
        "train_loss": avg_train_loss,
        "val_mse_reg": val_mse_reg,
        "val_mse_exp": val_mse_exp,
        "val_mse_blend": val_mse_blend,
        "val_pearson_reg": val_pearson_reg,
        "val_pearson_exp": val_pearson_exp,
        "val_pearson_blend": val_pearson_blend
    })
    logging.info(
        f"Validation epoch {epoch+1}: "
        f"MSE(reg)={val_mse_reg:.6f}, MSE(exp)={val_mse_exp:.6f}, MSE(blend)={val_mse_blend:.6f} | "
        f"Pearson(reg)={val_pearson_reg:.6f}, Pearson(exp)={val_pearson_exp:.6f}, Pearson(blend)={val_pearson_blend:.6f}"
    )

    if val_pearson_blend > best_val_pearson:
        best_val_pearson = val_pearson_blend
        best_epoch = epoch + 1
        logging.info(f"New best validation Pearson (blend): {best_val_pearson:.6f} at epoch {best_epoch}")

logging.info(f"Final best validation Pearson correlation (blend): {best_val_pearson:.6f} (epoch {best_epoch})")
logging.info("Validation history:")
for rec in history:
    logging.info(
        f"  Epoch {rec['epoch']}: train_loss={rec['train_loss']:.6f}, "
        f"MSE(reg)={rec['val_mse_reg']:.6f}, MSE(exp)={rec['val_mse_exp']:.6f}, MSE(blend)={rec['val_mse_blend']:.6f}, "
        f"Pearson(reg)={rec['val_pearson_reg']:.6f}, Pearson(exp)={rec['val_pearson_exp']:.6f}, Pearson(blend)={rec['val_pearson_blend']:.6f}"
    )

# Final validation inference for logging
logging.info("Running final validation inference for logging (blend).")
model.eval()
val_preds_reg = []
val_preds_exp = []
val_targets = []
with torch.no_grad():
    for batch in valid_loader:
        batch = to_device(batch)
        labels = batch.pop("labels").float()
        with torch.amp.autocast('cuda', enabled=CFG.fp16):
            outputs = model(**batch)
            pred_reg = outputs["regression"]
            logits_cls = outputs["class_logits"]
            probs_cls = torch.softmax(logits_cls, dim=-1)
            expected_val = torch.sum(probs_cls * CLASS_BINS, dim=-1)
        val_preds_reg.append(pred_reg.detach().float().cpu().numpy())
        val_preds_exp.append(expected_val.detach().float().cpu().numpy())
        val_targets.append(labels.detach().float().cpu().numpy())
val_preds_reg = np.concatenate(val_preds_reg)
val_preds_exp = np.concatenate(val_preds_exp)
val_targets = np.concatenate(val_targets)
val_preds_blend = CFG.blend_alpha * val_preds_reg + (1.0 - CFG.blend_alpha) * val_preds_exp
val_mse_final = float(np.mean((val_preds_blend - val_targets) ** 2))
val_pearson_final = compute_pearsonr(val_targets, val_preds_blend)
logging.info(f"Final Validation Results (blend) -> MSE: {val_mse_final:.6f}, Pearson: {val_pearson_final:.6f}")

# =============================================================================
# Test inference with TTA (swap anchor/target) and blend
# =============================================================================
def run_inference(loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    preds_reg = []
    preds_exp = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch)
            with torch.amp.autocast('cuda', enabled=CFG.fp16):
                outputs = model(**batch)
                pred_reg = outputs["regression"]
                logits_cls = outputs["class_logits"]
                probs_cls = torch.softmax(logits_cls, dim=-1)
                expected_val = torch.sum(probs_cls * CLASS_BINS, dim=-1)
            preds_reg.append(pred_reg.detach().float().cpu().numpy())
            preds_exp.append(expected_val.detach().float().cpu().numpy())
    return np.concatenate(preds_reg), np.concatenate(preds_exp)

logging.info("Running inference on test set (original and swapped TTA).")
t_reg, t_exp = run_inference(test_loader)
t_reg_sw, t_exp_sw = run_inference(test_loader_swapped)

t_blend = CFG.blend_alpha * t_reg + (1.0 - CFG.blend_alpha) * t_exp
t_blend_sw = CFG.blend_alpha * t_reg_sw + (1.0 - CFG.blend_alpha) * t_exp_sw
test_preds = 0.5 * (t_blend + t_blend_sw)
test_preds = np.clip(test_preds, 0.0, 1.0)

# =============================================================================
# Save submission
# =============================================================================
logging.info("Preparing submission file.")
submission = pd.DataFrame({
    "id": df_test["id"],
    "score": test_preds
})
submission.to_csv(CFG.submission_file, index=False)
logging.info(f"Submission saved to: {CFG.submission_file}")

# =============================================================================
# Extra analytics
# =============================================================================
logging.info("Extra analytics and diagnostics.")
train_score_counts = df_train["score"].value_counts().sort_index()
for s, c in train_score_counts.items():
    logging.info(f"Train score={s}: count={c}, pct={100.0*c/len(df_train):.2f}%")

logging.info("Token length diagnostics (first 100 train samples).")
lens = []
for i in range(min(100, len(df_train))):
    enc = tokenizer(
        df_train["first_seq"].iloc[i],
        df_train["second_seq"].iloc[i],
        truncation=True,
        max_length=CFG.max_length,
        add_special_tokens=True
    )
    lens.append(len(enc["input_ids"]))
lens = np.array(lens)
logging.info(f"Token length stats -> mean={lens.mean():.2f}, max={lens.max()}, min={lens.min()}")

logging.info("Pipeline completed successfully with CUDA fp16, large cross-encoder, MTL head, LLRD, cosine schedule, and TTA. Ready for submission.")