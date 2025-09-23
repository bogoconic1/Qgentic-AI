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
from sklearn.model_selection import StratifiedKFold, GroupKFold

# =========================
# Paths and Logging Setup
# =========================
BASE_DIR = "task/us-patent-phrase-to-phrase-matching"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "3")
LOG_PATH = os.path.join(OUTPUT_DIR, "code_3_v10.txt")
SUB_PATH = os.path.join(OUTPUT_DIR, "submission_10.csv")

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
    model_name: str = "microsoft/deberta-v3-small"  # a bit larger than xsmall for potential gains
    use_fast_tokenizer: bool = True
    max_len: int = 128
    # Optimization
    epochs: int = 3
    train_bs: int = 64
    valid_bs: int = 128
    base_lr: float = 2e-5
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    scheduler: str = "cosine"
    # Mixed Precision
    use_fp16: bool = True
    # Pooling and head
    pooling: str = "wlp_attn"    # options: mean_last_n | cls | mean | wlp_attn
    last_n_layers: int = 4       # used for mean_last_n
    wlp_use_last_n: int = -1     # -1 means use all encoder layers for WLP (excl. embeddings)
    mlp_hidden: int = 256
    dropout: float = 0.2
    # Multi-sample dropout
    msd_num: int = 5
    # LLRD
    llrd: bool = True
    llrd_decay: float = 0.9
    # Regularization / Loss
    use_smooth_l1: bool = True
    smooth_l1_beta: float = 0.1
    use_class_weights: bool = True
    # Ordinal/classification head
    use_class_head: bool = True
    num_classes: int = 5
    ord_sigma: float = 0.75
    loss_w_reg: float = 1.0
    loss_w_cls: float = 0.2
    loss_w_emd: float = 0.2
    loss_w_align: float = 0.05  # align regression with class expectation
    # R-Drop consistency
    rdrop_alpha: float = 0.5
    # Blend of regression and classification expectation for final prediction
    pred_blend_alpha: float = 0.85
    # Data Augmentation
    augment_swap: bool = True
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
    # Gradient checkpointing
    grad_checkpointing: bool = False
    # EMA
    ema: bool = True
    ema_decay: float = 0.999
    # CV
    n_folds: int = 5
    cv_type: str = "group_anchor"  # "group_anchor" | "stratified"
    group_mode: str = "anchor"     # "anchor" | "anchor_context"
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
# Targets utilities
# =========================
def bin_from_score(score: float) -> int:
    return int(np.clip(round(score * 4.0), 0, CFG.num_classes - 1))

def soft_target_from_score(score: float, num_classes: int, sigma: float) -> np.ndarray:
    b = bin_from_score(score)
    idxs = np.arange(num_classes, dtype=np.float32)
    weights = np.exp(-0.5 * ((idxs - b) / sigma) ** 2)
    weights = weights / np.sum(weights)
    return weights.astype(np.float32)

def build_soft_targets(scores: np.ndarray) -> np.ndarray:
    logging.info("Building soft class targets using Gaussian sigma=%.3f over %d classes.", CFG.ord_sigma, CFG.num_classes)
    out = np.stack([soft_target_from_score(s, CFG.num_classes, CFG.ord_sigma) for s in scores], axis=0)
    return out

# =========================
# Dataset
# =========================
class PatentPairsDataset(Dataset):
    def __init__(self, encodings: Dict[str, List[int]], labels: Optional[np.ndarray] = None, cls_targets: Optional[np.ndarray] = None, weights: Optional[np.ndarray] = None):
        self.encodings = encodings
        self.labels = labels
        self.cls_targets = cls_targets
        self.weights = weights

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        if self.cls_targets is not None:
            item["cls_targets"] = torch.tensor(self.cls_targets[idx], dtype=torch.float)
        if self.weights is not None:
            item["weights"] = torch.tensor(self.weights[idx], dtype=torch.float)
        return item

# =========================
# Pooling Modules
# =========================
class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

class WeightedLayerPooling(nn.Module):
    def __init__(self, num_layers: int, use_last_n: int = -1):
        super().__init__()
        self.num_layers_total = num_layers
        self.use_last_n = use_last_n if use_last_n > 0 else num_layers
        self.layer_weights = nn.Parameter(torch.ones(self.use_last_n, dtype=torch.float32))
        logging.info("Initialized WeightedLayerPooling over last %d layers (of %d total).", self.use_last_n, num_layers)

    def forward(self, all_hidden_states: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        selected = all_hidden_states[-self.use_last_n:]  # list of [B, T, H]
        stacked = torch.stack(selected, dim=0)           # [L, B, T, H]
        norm_w = torch.softmax(self.layer_weights, dim=0).view(self.use_last_n, 1, 1, 1)
        fused = torch.sum(norm_w * stacked, dim=0)       # [B, T, H]
        return fused

class TokenAttentionPooling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        logging.info("Initialized TokenAttentionPooling with hidden_size=%d", hidden_size)

    def forward(self, sequence_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        scores = self.attn(sequence_hidden).squeeze(-1)  # [B, T]
        mask = (attention_mask == 0)
        fill_value = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(mask, fill_value)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B, T, 1]
        pooled = torch.sum(sequence_hidden * weights, dim=1)  # [B, H]
        return pooled

# =========================
# Model
# =========================
class PatentRegressor(nn.Module):
    def __init__(self, model_name: str, pooling: str = "wlp_attn", last_n_layers: int = 4, mlp_hidden: int = 256, dropout: float = 0.2, msd_num: int = 5, num_classes: int = 5, use_class_head: bool = True, wlp_use_last_n: int = -1):
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
        hidden_size = self.config.hidden_size
        self.hidden_size = hidden_size

        if self.pooling_type == "wlp_attn":
            num_layers_total = self.config.num_hidden_layers + 1  # embeddings + hidden layers
            use_last = wlp_use_last_n if wlp_use_last_n > 0 else self.config.num_hidden_layers
            self.wlp = WeightedLayerPooling(num_layers=num_layers_total, use_last_n=use_last)
            self.token_pool = TokenAttentionPooling(hidden_size)
        else:
            self.token_pool = MeanPooling()

        self.msd_num = msd_num
        self.use_class_head = use_class_head
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(msd_num)])

        self.regressor_head = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

        if self.use_class_head:
            self.classifier_head = nn.Sequential(
                nn.Linear(hidden_size, mlp_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, num_classes),
            )
        else:
            self.classifier_head = None

        logging.info("Backbone hidden size: %d", hidden_size)
        logging.info("Pooling: %s | last_n=%d | wlp_use_last_n=%d | MSD=%d | Class head=%s",
                     pooling, last_n_layers, use_last, msd_num, str(self.use_class_head))

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            return_dict=True,
        )

        if self.pooling_type == "mean_last_n":
            hidden_states = outputs.hidden_states
            selected = hidden_states[-self.last_n_layers:]
            stacked = torch.stack(selected, dim=0)
            avg_last = stacked.mean(dim=0)  # [B, T, H]
            pooled = self.token_pool(avg_last, attention_mask)
        elif self.pooling_type == "cls":
            pooled = outputs.last_hidden_state[:, 0]
        elif self.pooling_type == "wlp_attn":
            fused_seq = self.wlp(outputs.hidden_states)  # [B, T, H]
            pooled = self.token_pool(fused_seq, attention_mask)  # [B, H]
        else:
            pooled = self.token_pool(outputs.last_hidden_state, attention_mask)

        reg_outs = []
        cls_outs = []
        for i, dp in enumerate(self.dropouts):
            d = dp(pooled)
            reg_outs.append(self.regressor_head(d).squeeze(-1))
            if self.use_class_head:
                cls_outs.append(self.classifier_head(d))

        reg_pred = torch.stack(reg_outs, dim=0).mean(dim=0)
        if self.use_class_head:
            cls_logits = torch.stack(cls_outs, dim=0).mean(dim=0)
        else:
            cls_logits = None

        return reg_pred, cls_logits

# =========================
# EMA (Exponential Moving Average) of parameters
# =========================
class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.detach().clone()

    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_avg = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_avg.detach().clone()

    def apply_shadow(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.detach().clone()
                param.data = self.shadow[name].detach().clone()

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name].detach().clone()
        self.backup = {}

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
    logging.info("Weights per bin (inverse counts): %s", inv.tolist())
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

    # Head params (regressor + classifier)
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
# Fold-specific data building (with augmentation)
# =========================
def build_fold_texts_and_labels(train_idx: np.ndarray, valid_idx: np.ndarray):
    logging.info("Building fold texts and labels with augmentation=%s", str(CFG.augment_swap))
    tr_df = train_merged.iloc[train_idx].reset_index(drop=True)
    va_df = train_merged.iloc[valid_idx].reset_index(drop=True)

    tr_a_1, tr_b_1 = build_input_texts(tr_df)
    tr_labels = tr_df["score"].values.astype(np.float32)
    tr_cls_targets = build_soft_targets(tr_labels)

    if CFG.augment_swap:
        tr_a_2, tr_b_2 = build_swapped_input_texts(tr_df)
        tr_text_a = tr_a_1 + tr_a_2
        tr_text_b = tr_b_1 + tr_b_2
        tr_labels = np.concatenate([tr_labels, tr_labels], axis=0)
        tr_cls_targets = np.concatenate([tr_cls_targets, tr_cls_targets], axis=0)
        logging.info("Train augmentation doubled samples: %d -> %d", len(tr_a_1), len(tr_text_a))
    else:
        tr_text_a, tr_text_b = tr_a_1, tr_b_1

    va_text_a, va_text_b = build_input_texts(va_df)
    va_labels = va_df["score"].values.astype(np.float32)
    va_cls_targets = build_soft_targets(va_labels)

    return (tr_text_a, tr_text_b, tr_labels, tr_cls_targets), (va_text_a, va_text_b, va_labels, va_cls_targets)

# =========================
# Losses
# =========================
def classification_losses_to_targets(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    log_probs = torch.log_softmax(logits, dim=1)
    probs = torch.softmax(logits, dim=1)
    eps = 1e-8
    log_targets = torch.log(torch.clamp(targets, min=eps))
    kl_per_sample = torch.sum(targets * (log_targets - log_probs), dim=1)
    cdf_probs = torch.cumsum(probs, dim=1)
    cdf_targets = torch.cumsum(targets, dim=1)
    emd_per_sample = torch.mean((cdf_probs - cdf_targets) ** 2, dim=1)
    return kl_per_sample, emd_per_sample

def symmetric_kl_logits(logits1: torch.Tensor, logits2: torch.Tensor) -> torch.Tensor:
    logp1 = torch.log_softmax(logits1, dim=1)
    logp2 = torch.log_softmax(logits2, dim=1)
    p1 = torch.softmax(logits1, dim=1)
    p2 = torch.softmax(logits2, dim=1)
    kl12 = torch.sum(p1 * (logp1 - logp2), dim=1)
    kl21 = torch.sum(p2 * (logp2 - logp1), dim=1)
    return 0.5 * (kl12 + kl21)

def exp_score_from_logits(cls_logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(cls_logits, dim=1)
    class_values = torch.arange(CFG.num_classes, device=probs.device, dtype=probs.dtype) / (CFG.num_classes - 1)
    exp_score = (probs * class_values.unsqueeze(0)).sum(dim=1)
    return exp_score

def blend_predictions(reg_pred: torch.Tensor, cls_logits: Optional[torch.Tensor]) -> torch.Tensor:
    if cfg_use_cls_head():
        exp_score = exp_score_from_logits(cls_logits)
        final_pred = CFG.pred_blend_alpha * reg_pred + (1.0 - CFG.pred_blend_alpha) * exp_score
        return final_pred
    else:
        return reg_pred

def cfg_use_cls_head() -> bool:
    return CFG.use_class_head and CFG.num_classes > 1

# =========================
# Evaluation helper
# =========================
def evaluate_model(model: nn.Module, loader: DataLoader, use_ema: bool, ema_obj: Optional['EMA']) -> Tuple[np.ndarray, np.ndarray, float]:
    logging.info("Evaluation start (use_ema=%s)", str(use_ema))
    model.eval()
    if use_ema and ema_obj is not None:
        logging.info("Applying EMA shadow weights for evaluation.")
        ema_obj.apply_shadow(model)
    preds_all = []
    labels_all = []
    with torch.no_grad():
        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            token_type_ids = batch.get("token_type_ids", None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device, non_blocking=True)
            labels = batch.get("labels", None)
            with torch.amp.autocast('cuda', enabled=CFG.use_fp16, dtype=torch.float16):
                reg_pred, cls_logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                pred = blend_predictions(reg_pred, cls_logits)
            preds_all.append(pred.detach().cpu().numpy())
            if labels is not None:
                labels_all.append(labels.detach().cpu().numpy())
    preds_np = np.concatenate(preds_all).astype(np.float32)
    if len(labels_all) > 0:
        labels_np = np.concatenate(labels_all).astype(np.float32)
        score = compute_pearsonr(labels_np, preds_np)
    else:
        labels_np = np.array([], dtype=np.float32)
        score = float("nan")
    if use_ema and ema_obj is not None:
        logging.info("Restoring original (non-EMA) weights after evaluation.")
        ema_obj.restore(model)
    logging.info("Evaluation done. Pearson=%.6f", score if not math.isnan(score) else -1)
    return preds_np, labels_np, score

# =========================
# Training and Validation per-fold
# =========================
def train_one_fold(fold_idx: int, train_idx: np.ndarray, valid_idx: np.ndarray, test_text_a: List[str], test_text_b: List[str], test_text_a_sw: Optional[List[str]], test_text_b_sw: Optional[List[str]]) -> Tuple[np.ndarray, np.ndarray]:
    logging.info("=" * 80)
    logging.info("Starting Fold %d/%d", fold_idx, CFG.n_folds)
    logging.info("Fold train size: %d | Fold valid size: %d", len(train_idx), len(valid_idx))

    (tr_a, tr_b, tr_y, tr_y_cls), (va_a, va_b, va_y, va_y_cls) = build_fold_texts_and_labels(train_idx, valid_idx)

    logging.info("Tokenizing train/valid for fold %d with max_length=%d", fold_idx, CFG.max_len)
    tr_enc = tokenize_text_pairs(tr_a, tr_b)
    va_enc = tokenize_text_pairs(va_a, va_b)

    if CFG.use_class_weights:
        tr_weights = compute_sample_weights(tr_y)
    else:
        tr_weights = np.ones_like(tr_y, dtype=np.float32)

    train_dataset = PatentPairsDataset(tr_enc, tr_y, tr_y_cls if cfg_use_cls_head() else None, tr_weights)
    valid_dataset = PatentPairsDataset(va_enc, va_y, va_y_cls if cfg_use_cls_head() else None, None)

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
        msd_num=CFG.msd_num,
        num_classes=CFG.num_classes,
        use_class_head=cfg_use_cls_head(),
        wlp_use_last_n=CFG.wlp_use_last_n if CFG.wlp_use_last_n != -1 else (AutoConfig.from_pretrained(CFG.model_name).num_hidden_layers),
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
        reg_criterion = nn.SmoothL1Loss(beta=CFG.smooth_l1_beta, reduction="none")
        logging.info("Using SmoothL1Loss with beta=%.4f", CFG.smooth_l1_beta)
    else:
        reg_criterion = nn.MSELoss(reduction="none")
        logging.info("Using MSELoss")

    ema = EMA(model, CFG.ema_decay) if CFG.ema else None

    best_score = -1.0
    best_state = None

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
            if cfg_use_cls_head():
                cls_targets = batch["cls_targets"].to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=CFG.use_fp16, dtype=torch.float16):
                reg_pred1, cls_logits1 = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                reg_loss_vec1 = reg_criterion(reg_pred1, labels)
                if cfg_use_cls_head():
                    kl_vec1, emd_vec1 = classification_losses_to_targets(cls_logits1, cls_targets)
                else:
                    kl_vec1 = torch.zeros_like(reg_loss_vec1)
                    emd_vec1 = torch.zeros_like(reg_loss_vec1)

                reg_pred2, cls_logits2 = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                reg_loss_vec2 = reg_criterion(reg_pred2, labels)
                if cfg_use_cls_head():
                    kl_vec2, emd_vec2 = classification_losses_to_targets(cls_logits2, cls_targets)
                else:
                    kl_vec2 = torch.zeros_like(reg_loss_vec2)
                    emd_vec2 = torch.zeros_like(reg_loss_vec2)

                base_loss_vec = CFG.loss_w_reg * 0.5 * (reg_loss_vec1 + reg_loss_vec2) \
                                + CFG.loss_w_cls * 0.5 * (kl_vec1 + kl_vec2) \
                                + CFG.loss_w_emd * 0.5 * (emd_vec1 + emd_vec2)

                reg_cons_vec = reg_criterion(reg_pred1, reg_pred2)
                if cfg_use_cls_head():
                    skl_vec = symmetric_kl_logits(cls_logits1, cls_logits2)
                    exp1 = exp_score_from_logits(cls_logits1)
                    exp2 = exp_score_from_logits(cls_logits2)
                    align_vec = 0.5 * (reg_criterion(reg_pred1, exp1) + reg_criterion(reg_pred2, exp2))
                else:
                    skl_vec = torch.zeros_like(reg_cons_vec)
                    align_vec = torch.zeros_like(reg_cons_vec)

                loss_vec = base_loss_vec + CFG.rdrop_alpha * (reg_cons_vec + skl_vec) + CFG.loss_w_align * align_vec
                loss = (loss_vec * weights).mean()

            if CFG.grad_accum_steps > 1:
                loss = loss / CFG.grad_accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % CFG.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                if CFG.ema:
                    ema.update(model)
                global_step += 1

            epoch_loss += loss.item()

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

        logging.info("Fold %d | Epoch %d | Validation (in-epoch) start", fold_idx, epoch + 1)
        _, _, val_score = evaluate_model(model, valid_loader, use_ema=CFG.ema, ema_obj=ema)
        if val_score > best_score:
            best_score = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            logging.info("Fold %d | Epoch %d | New best Pearson: %.6f -> saved model state.", fold_idx, epoch + 1, best_score)

    if best_state is not None:
        logging.info("Fold %d | Loading best model state with Pearson=%.6f", fold_idx, best_score)
        model.load_state_dict(best_state)

    logging.info("Fold %d | Final Validation with best model", fold_idx)
    valid_preds, valid_labels, fold_pearson = evaluate_model(model, valid_loader, use_ema=CFG.ema, ema_obj=ema)
    logging.info("Fold %d | Validation Pearson: %.6f", fold_idx, fold_pearson)

    if CFG.ema:
        logging.info("Applying EMA weights for test inference.")
        ema.apply_shadow(model)

    logging.info("Fold %d | Tokenizing test set", fold_idx)
    test_enc = tokenize_text_pairs(test_text_a, test_text_b)
    test_dataset = PatentPairsDataset(test_enc, None, None, None)
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
                reg_pred, cls_logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                pred = blend_predictions(reg_pred, cls_logits)
            fold_test_preds.append(pred.detach().cpu().numpy())
    fold_test_preds = np.concatenate(fold_test_preds).astype(np.float32)

    if CFG.tta_swap and test_text_a_sw is not None and test_text_b_sw is not None:
        logging.info("Fold %d | TTA swap enabled: running swapped test inference", fold_idx)
        test_enc_sw = tokenize_text_pairs(test_text_a_sw, test_text_b_sw)
        test_dataset_sw = PatentPairsDataset(test_enc_sw, None, None, None)
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
                    reg_pred, cls_logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                    pred = blend_predictions(reg_pred, cls_logits)
                fold_test_preds_sw.append(pred.detach().cpu().numpy())
        fold_test_preds_sw = np.concatenate(fold_test_preds_sw).astype(np.float32)
        fold_test_preds = 0.5 * (fold_test_preds + fold_test_preds_sw)
        logging.info("Fold %d | TTA averaging completed", fold_idx)

    if CFG.ema:
        logging.info("Restoring original weights after EMA test inference.")
        ema.restore(model)

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
# Cross-Validation setup (GroupKFold by anchor or StratifiedKFold)
# =========================
y = train_df["score"].values
if CFG.cv_type == "group_anchor":
    logging.info("Using GroupKFold CV with groups based on '%s'", CFG.group_mode)
    if CFG.group_mode == "anchor":
        groups = train_df["anchor"].astype(str).values
    else:
        groups = (train_df["anchor"].astype(str) + "||" + train_df["context"].astype(str)).values
    gkf = GroupKFold(n_splits=CFG.n_folds)
    splits = list(gkf.split(np.zeros(len(y)), y, groups=groups))
else:
    logging.info("Using StratifiedKFold CV on score bins.")
    y_bins = np.floor(y * 4).astype(int)
    skf = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)
    splits = list(skf.split(np.zeros(len(y)), y_bins))

logging.info("Prepared %d folds.", len(splits))

# =========================
# Run folds
# =========================
oof_pred = np.zeros(len(train_df), dtype=np.float32)
test_pred_folds = []

for fold_counter, (tr_idx, va_idx) in enumerate(splits, start=1):
    valid_preds, test_preds = train_one_fold(fold_counter, tr_idx, va_idx, test_text_a, test_text_b, None if not CFG.tta_swap else test_text_a_sw, None if not CFG.tta_swap else test_text_b_sw)
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