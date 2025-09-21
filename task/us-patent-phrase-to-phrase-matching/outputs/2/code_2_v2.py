import os
import math
import random
import json
import time
import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader

# =========================================
# Paths and logging (versioned v2 artifacts)
# =========================================

BASE_DIR = "task/us-patent-phrase-to-phrase-matching"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "2")
LOG_FILE = os.path.join(OUTPUT_DIR, "code_2_v2.txt")
SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission_2.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    filemode="w",
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logging.info("Starting refined script execution (v2).")
logging.info(f"Base directory: {BASE_DIR}")
logging.info(f"Output directory: {OUTPUT_DIR}")
logging.info(f"Log file: {LOG_FILE}")
logging.info(f"Submission path: {SUBMISSION_PATH}")

# ===========
# Reproducible
# ===========

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True
logging.info(f"Set random seed to {SEED} and enabled cudnn.benchmark=True")

# ===========
# Device setup
# ===========

assert torch.cuda.is_available(), "CUDA is required but not available."
DEVICE = torch.device("cuda")
logging.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")

# =============
# Hyperparameters
# =============

@dataclass
class HParams:
    max_word_len: int = 72         # per branch (anchor/context or target/context)
    max_char_len: int = 128        # per branch for char-CNN
    vocab_min_freq: int = 1
    char_vocab_min_freq: int = 1

    d_model: int = 384             # word-transformer hidden size
    nhead: int = 8
    dim_ff: int = 1024
    num_layers: int = 6
    dropout: float = 0.15

    char_dim: int = 64
    char_num_filters: int = 128
    char_kernel_sizes: Tuple[int, ...] = (3, 4, 5)

    fusion_hidden: int = 512

    lr: float = 1.5e-3
    weight_decay: float = 1e-2
    epochs: int = 5
    batch_size: int = 192
    num_folds: int = 4
    grad_clip_norm: float = 1.0
    warmup_ratio: float = 0.06
    fp16: bool = True

    swap_prob: float = 0.5         # data augmentation: swap anchor/target
    oversample_bins: Tuple[int, ...] = (0, 4)  # emphasize extremes 0.0 and 1.0
    oversample_factor: float = 1.0  # additional multiplicative factor on weights

HP = HParams()
logging.info(f"Hyperparameters: {json.dumps(HP.__dict__, indent=2)}")

# ==========================================
# CPC enrichment (titles heuristic without IO)
# ==========================================

CPC_SECTION_TITLES = {
    "A": "human necessities",
    "B": "performing operations transporting",
    "C": "chemistry metallurgy",
    "D": "textiles paper",
    "E": "fixed constructions",
    "F": "mechanical engineering lighting heating weapons blasting",
    "G": "physics",
    "H": "electricity",
    "Y": "general tagging of new technological developments",
}

# Frequently seen CPC prefixes with descriptive titles (heuristic enrichment)
CPC_PREFIX_TITLES = {
    "G06": "computing calculating counting",
    "G01": "measuring testing",
    "G02": "optics optical elements systems",
    "H01": "basic electric elements",
    "H02": "generation conversion distribution of electric power",
    "H03": "basic electronic circuitry",
    "H04": "electric communication transmission",
    "H04W": "wireless communication networks",
    "A61": "medical or veterinary science hygiene",
    "C07": "organic chemistry",
    "C08": "organic macromolecular compounds polymers",
    "C09": "dyes paints polishes resins",
    "B60": "vehicles in general",
    "B65": "conveying packing storing handling thin materials",
    "F16": "engineering elements general measures",
    "B01": "physical or chemical processes or apparatus in general",
    "C12": "biochemistry microbiology genetic engineering fermentation",
    "E05": "locks keys window or door fittings",
    "F24": "heating ranges ventilating",
    "F41": "weapons",
    "H05": "electric techniques not otherwise provided for",
}

def enrich_context(ctx: str) -> str:
    ctx = str(ctx).strip()
    code = ctx.upper()
    # Try 4-char, then 3-char prefix
    for k in sorted(CPC_PREFIX_TITLES.keys(), key=len, reverse=True):
        if code.startswith(k):
            title = CPC_PREFIX_TITLES[k]
            return f"{ctx} {title}"
    section = code[0] if len(code) > 0 else "Y"
    title = CPC_SECTION_TITLES.get(section, "technology")
    return f"{ctx} {title}"

# ==================
# Data Preparation
# ==================

TRAIN_PATH = os.path.join(BASE_DIR, "train.csv")
TEST_PATH = os.path.join(BASE_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_DIR, "sample_submission.csv")

logging.info(f"Reading train from {TRAIN_PATH}")
train_df = pd.read_csv(TRAIN_PATH)
logging.info(f"Reading test from {TEST_PATH}")
test_df = pd.read_csv(TEST_PATH)
logging.info(f"Read datasets: train={train_df.shape}, test={test_df.shape}")

# Construct dual-branch texts: anchor/context and target/context
def make_branch_texts(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    logging.info("Constructing dual-branch texts: (anchor + context_enriched), (target + context_enriched).")
    ctx_enriched = df["context"].map(enrich_context).str.lower()
    a_text = (df["anchor"].astype(str).str.lower() + " [SEP] " + ctx_enriched)
    b_text = (df["target"].astype(str).str.lower() + " [SEP] " + ctx_enriched)
    return a_text, b_text

train_a_text, train_b_text = make_branch_texts(train_df)
test_a_text, test_b_text = make_branch_texts(test_df)

# Score bins: 0.0, 0.25, 0.5, 0.75, 1.0 -> 0..4
train_df["bin"] = (train_df["score"] * 4).astype(int)
logging.info("Assigned score bins for stratification.")

# =================
# Tokenizer & Vocab
# =================

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[SEP]", "[CLS]"]
PAD_ID = 0
UNK_ID = 1
SEP_ID = 2
CLS_ID = 3

def normalize_text_for_tokenization(s: str) -> str:
    punct = [",", ".", ";", ":", "(", ")", "[", "]", "{", "}", "/", "\\", "-", "+", "=", "*", "&", "^", "%", "$", "#", "@", "!", "?", "|", "<", ">"]
    for p in punct:
        s = s.replace(p, f" {p} ")
    s = " ".join(s.split())
    return s

def whitespace_tokenize(text: str) -> List[str]:
    return normalize_text_for_tokenization(text).split()

def build_vocab(texts: List[str], min_freq: int) -> Tuple[Dict[str,int], Dict[int,str]]:
    logging.info("Building word vocabulary from train+test branch texts.")
    freq = {}
    for t in texts:
        for tok in whitespace_tokenize(t):
            freq[tok] = freq.get(tok, 0) + 1
    stoi = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    itos = {i: tok for tok, i in stoi.items()}
    for tok, c in sorted(freq.items(), key=lambda x: (-x[1], x[0])):
        if c >= min_freq and tok not in stoi:
            idx = len(stoi)
            stoi[tok] = idx
            itos[idx] = tok
    logging.info(f"Word vocabulary size: {len(stoi)} (min_freq={min_freq})")
    return stoi, itos

def build_char_vocab(texts: List[str], min_freq: int) -> Tuple[Dict[str,int], Dict[int,str]]:
    logging.info("Building char vocabulary from train+test branch texts.")
    freq = {}
    for t in texts:
        for ch in t:
            freq[ch] = freq.get(ch, 0) + 1
    # Reserve: 0 [PAD], 1 [UNK]
    cstoi = {"[PAD]": 0, "[UNK]": 1}
    citas = {0: "[PAD]", 1: "[UNK]"}
    for ch, c in sorted(freq.items(), key=lambda x: (-x[1], x[0])):
        if c >= min_freq and ch not in cstoi:
            idx = len(cstoi)
            cstoi[ch] = idx
            citas[idx] = ch
    logging.info(f"Char vocabulary size: {len(cstoi)} (min_freq={min_freq})")
    return cstoi, citas

def encode_words(text: str, stoi: Dict[str,int], max_len: int) -> List[int]:
    toks = ["[CLS]"] + whitespace_tokenize(text)
    ids = [stoi.get(tok, UNK_ID) for tok in toks]
    if len(ids) > max_len:
        ids = ids[:max_len]
    else:
        ids = ids + [PAD_ID] * (max_len - len(ids))
    return ids

def encode_chars(text: str, cstoi: Dict[str,int], max_len: int) -> List[int]:
    ids = [cstoi.get(ch, 1) for ch in text]  # 1 is [UNK]
    if len(ids) > max_len:
        ids = ids[:max_len]
    else:
        ids = ids + [0] * (max_len - len(ids))  # 0 is [PAD]
    return ids

all_branch_texts = list(train_a_text.values) + list(train_b_text.values) + list(test_a_text.values) + list(test_b_text.values)
word_stoi, word_itos = build_vocab(all_branch_texts, HP.vocab_min_freq)
char_stoi, char_itos = build_char_vocab(all_branch_texts, HP.char_vocab_min_freq)

# =========
# Dataset
# =========

class SiamesePatentDataset(Dataset):
    def __init__(self, a_texts: List[str], b_texts: List[str], scores: List[float] = None,
                 weights: List[float] = None, max_word_len: int = 72, max_char_len: int = 128,
                 swap_prob: float = 0.0):
        self.a_texts = a_texts
        self.b_texts = b_texts
        self.scores = scores
        self.has_labels = scores is not None
        self.weights = weights
        self.max_word_len = max_word_len
        self.max_char_len = max_char_len
        self.swap_prob = swap_prob

    def __len__(self):
        return len(self.a_texts)

    def __getitem__(self, idx):
        ta = self.a_texts[idx]
        tb = self.b_texts[idx]
        swapped = False
        if self.has_labels and self.swap_prob > 0.0 and random.random() < self.swap_prob:
            ta, tb = tb, ta
            swapped = True

        a_w = encode_words(ta, word_stoi, self.max_word_len)
        b_w = encode_words(tb, word_stoi, self.max_word_len)
        a_mask = [0 if i == PAD_ID else 1 for i in a_w]
        b_mask = [0 if i == PAD_ID else 1 for i in b_w]

        a_c = encode_chars(ta, char_stoi, self.max_char_len)
        b_c = encode_chars(tb, char_stoi, self.max_char_len)

        item = {
            "a_input_ids": torch.tensor(a_w, dtype=torch.long),
            "a_attention_mask": torch.tensor(a_mask, dtype=torch.long),
            "b_input_ids": torch.tensor(b_w, dtype=torch.long),
            "b_attention_mask": torch.tensor(b_mask, dtype=torch.long),
            "a_char_ids": torch.tensor(a_c, dtype=torch.long),
            "b_char_ids": torch.tensor(b_c, dtype=torch.long),
        }
        if self.has_labels:
            item["targets"] = torch.tensor(self.scores[idx], dtype=torch.float32)
            if self.weights is not None:
                item["weights"] = torch.tensor(self.weights[idx], dtype=torch.float32)
            else:
                item["weights"] = torch.tensor(1.0, dtype=torch.float32)
            item["swapped"] = torch.tensor(1 if swapped else 0, dtype=torch.long)
        return item

# =====================
# Model Implementation
# =====================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        return x + self.pe[:, :L, :].to(dtype=x.dtype)

class WordEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, nhead: int, dim_ff: int, num_layers: int, dropout: float, pad_id: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model=d_model, max_len=4096)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        x = self.pos(x)
        key_padding_mask = (attention_mask == 0)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        cls = x[:, 0, :]
        mask = attention_mask.unsqueeze(-1)
        mean = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        pooled = torch.cat([cls, mean], dim=-1)  # (B, 2*d)
        return self.dropout(pooled)

class CharCNNEncoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, num_filters: int, kernel_sizes: Tuple[int, ...], dropout: float):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, num_filters, k, padding=0) for k in kernel_sizes])
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        # char_ids: (B, Lc)
        x = self.embed(char_ids)  # (B, Lc, E)
        x = x.transpose(1, 2)     # (B, E, Lc)
        feats = []
        for conv in self.convs:
            h = conv(x)                   # (B, F, Lc-k+1)
            h = self.act(h)
            h = torch.max(h, dim=2).values  # (B, F)
            feats.append(h)
        out = torch.cat(feats, dim=1)  # (B, F * len(kernels))
        out = self.dropout(out)
        return out

class SiameseRegressor(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 char_vocab_size: int,
                 d_model: int,
                 nhead: int,
                 dim_ff: int,
                 num_layers: int,
                 dropout: float,
                 pad_id: int,
                 char_dim: int,
                 char_num_filters: int,
                 char_kernel_sizes: Tuple[int, ...],
                 fusion_hidden: int):
        super().__init__()
        self.word_enc = WordEncoder(vocab_size, d_model, nhead, dim_ff, num_layers, dropout, pad_id)
        self.word_proj = nn.Linear(d_model * 2, d_model)
        self.char_enc = CharCNNEncoder(char_vocab_size, char_dim, char_num_filters, char_kernel_sizes, dropout)
        char_out_dim = char_num_filters * len(char_kernel_sizes)
        fused_in = d_model + char_out_dim
        self.fuse = nn.Sequential(
            nn.LayerNorm(fused_in),
            nn.Linear(fused_in, fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        comb_in = fusion_hidden * 4 + 1  # a, b, |a-b|, a*b, and cosine scalar
        self.head = nn.Sequential(
            nn.LayerNorm(comb_in),
            nn.Linear(comb_in, fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def encode_branch(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, char_ids: torch.Tensor) -> torch.Tensor:
        w = self.word_enc(input_ids, attention_mask)
        w = self.word_proj(w)
        c = self.char_enc(char_ids)
        z = torch.cat([w, c], dim=-1)
        z = self.fuse(z)
        return z

    def forward(self,
                a_input_ids: torch.Tensor,
                a_attention_mask: torch.Tensor,
                a_char_ids: torch.Tensor,
                b_input_ids: torch.Tensor,
                b_attention_mask: torch.Tensor,
                b_char_ids: torch.Tensor) -> torch.Tensor:
        za = self.encode_branch(a_input_ids, a_attention_mask, a_char_ids)
        zb = self.encode_branch(b_input_ids, b_attention_mask, b_char_ids)
        diff = torch.abs(za - zb)
        prod = za * zb
        # cosine similarity
        cos = torch.nn.functional.cosine_similarity(za, zb, dim=-1, eps=1e-8).unsqueeze(-1)
        comb = torch.cat([za, zb, diff, prod, cos], dim=-1)
        out = self.head(comb).squeeze(-1)
        out = self.sigmoid(out)  # constrain to [0,1]
        return out

# ==========================
# Stratified K-Fold (custom)
# ==========================

def stratified_kfold_indices(labels: np.ndarray, n_splits: int, seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    logging.info(f"Creating stratified {n_splits}-fold indices.")
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    per_class_indices = {c: np.where(labels == c)[0] for c in uniq}
    rng = np.random.RandomState(seed)
    for c in uniq:
        rng.shuffle(per_class_indices[c])

    folds = [[] for _ in range(n_splits)]
    for c in uniq:
        idxs = per_class_indices[c]
        parts = np.array_split(idxs, n_splits)
        for f in range(n_splits):
            folds[f].extend(parts[f].tolist())

    splits = []
    all_indices = np.arange(len(labels))
    for f in range(n_splits):
        val_idx = np.array(sorted(folds[f]))
        train_mask = np.ones(len(labels), dtype=bool)
        train_mask[val_idx] = False
        train_idx = all_indices[train_mask]
        splits.append((train_idx, val_idx))
    logging.info("Completed stratified fold construction.")
    return splits

# ==========================
# Weighted Loss Preparation
# ==========================

def compute_sample_weights(df: pd.DataFrame, train_indices: np.ndarray) -> np.ndarray:
    logging.info("Computing per-sample weights: inverse frequency of (context, bin) with oversample of extremes.")
    sub = df.iloc[train_indices].copy()
    grp = sub.groupby(["context", "bin"]).size().rename("count").reset_index()
    merge_df = sub.merge(grp, on=["context", "bin"], how="left")
    base_weights = 1.0 / merge_df["count"].astype(float).values
    base_weights = base_weights / base_weights.mean()

    # Emphasize extreme bins if configured
    multiplier = np.ones_like(base_weights, dtype=np.float32)
    if len(HP.oversample_bins) > 0 and HP.oversample_factor > 0:
        for i, b in enumerate(merge_df["bin"].values):
            if b in HP.oversample_bins:
                multiplier[i] *= (1.0 + HP.oversample_factor)
    weights = base_weights * multiplier
    weights = weights / weights.mean()

    logging.info(f"Computed weights: mean={weights.mean():.6f}, std={weights.std():.6f}, min={weights.min():.6f}, max={weights.max():.6f}")
    full_weights = np.ones(len(df), dtype=np.float32)
    full_weights[train_indices] = weights.astype(np.float32)
    return full_weights

# ==========================
# Metrics
# ==========================

def pearsonr_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = y_true - y_true.mean()
    y_pred = y_pred - y_pred.mean()
    num = (y_true * y_pred).sum()
    den = np.sqrt((y_true ** 2).sum()) * np.sqrt((y_pred ** 2).sum())
    if den == 0.0:
        return 0.0
    return float(num / den)

# ==========================
# Scheduler
# ==========================

def get_scheduler(optimizer, num_warmup_steps: int, num_training_steps: int):
    logging.info(f"Creating linear warmup scheduler: warmup={num_warmup_steps}, total={num_training_steps}")
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)),
        )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ==========================
# Training & Evaluation loop
# ==========================

def train_one_fold(
    fold: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    df: pd.DataFrame,
    a_texts: pd.Series,
    b_texts: pd.Series
) -> Tuple[float, np.ndarray, Dict[str, torch.Tensor]]:
    logging.info(f"===== Fold {fold} | train size: {len(train_idx)}, val size: {len(val_idx)} =====")
    weights_full = compute_sample_weights(df, train_idx)

    tr_a = a_texts.iloc[train_idx].tolist()
    tr_b = b_texts.iloc[train_idx].tolist()
    tr_y = df.iloc[train_idx]["score"].astype(np.float32).tolist()
    tr_w = weights_full[train_idx].tolist()

    va_a = a_texts.iloc[val_idx].tolist()
    va_b = b_texts.iloc[val_idx].tolist()
    va_y = df.iloc[val_idx]["score"].astype(np.float32).tolist()

    train_ds = SiamesePatentDataset(tr_a, tr_b, tr_y, tr_w, max_word_len=HP.max_word_len, max_char_len=HP.max_char_len, swap_prob=HP.swap_prob)
    val_ds = SiamesePatentDataset(va_a, va_b, va_y, [1.0]*len(va_y), max_word_len=HP.max_word_len, max_char_len=HP.max_char_len, swap_prob=0.0)

    train_loader = DataLoader(train_ds, batch_size=HP.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=HP.batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)

    model = SiameseRegressor(
        vocab_size=len(word_stoi),
        char_vocab_size=len(char_stoi),
        d_model=HP.d_model,
        nhead=HP.nhead,
        dim_ff=HP.dim_ff,
        num_layers=HP.num_layers,
        dropout=HP.dropout,
        pad_id=PAD_ID,
        char_dim=HP.char_dim,
        char_num_filters=HP.char_num_filters,
        char_kernel_sizes=HP.char_kernel_sizes,
        fusion_hidden=HP.fusion_hidden
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Fold {fold} model initialized. Total params: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=HP.lr, weight_decay=HP.weight_decay)
    num_training_steps = HP.epochs * len(train_loader)
    num_warmup_steps = int(HP.warmup_ratio * num_training_steps)
    scheduler = get_scheduler(optimizer, num_warmup_steps, num_training_steps)
    scaler = GradScaler(enabled=HP.fp16)

    best_val_pearson = -1.0
    best_state = None

    mse = nn.MSELoss(reduction="none")

    global_step = 0
    for epoch in range(HP.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for batch in train_loader:
            a_input_ids = batch["a_input_ids"].to(DEVICE, non_blocking=True)
            a_attention_mask = batch["a_attention_mask"].to(DEVICE, non_blocking=True)
            a_char_ids = batch["a_char_ids"].to(DEVICE, non_blocking=True)
            b_input_ids = batch["b_input_ids"].to(DEVICE, non_blocking=True)
            b_attention_mask = batch["b_attention_mask"].to(DEVICE, non_blocking=True)
            b_char_ids = batch["b_char_ids"].to(DEVICE, non_blocking=True)
            targets = batch["targets"].to(DEVICE, non_blocking=True)
            w = batch["weights"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if HP.fp16:
                with autocast():
                    preds = model(a_input_ids, a_attention_mask, a_char_ids, b_input_ids, b_attention_mask, b_char_ids)
                    loss_raw = mse(preds, targets)
                    loss = (loss_raw * w).mean()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), HP.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(a_input_ids, a_attention_mask, a_char_ids, b_input_ids, b_attention_mask, b_char_ids)
                loss_raw = mse(preds, targets)
                loss = (loss_raw * w).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), HP.grad_clip_norm)
                optimizer.step()

            scheduler.step()
            epoch_loss += loss.item()
            global_step += 1

        epoch_loss /= max(1, len(train_loader))
        logging.info(f"Fold {fold} | Epoch {epoch+1}/{HP.epochs} | Train loss: {epoch_loss:.6f} | Time: {time.time()-t0:.2f}s")

        # Validation
        model.eval()
        val_preds = []
        val_tgts = []
        with torch.no_grad():
            for batch in val_loader:
                a_input_ids = batch["a_input_ids"].to(DEVICE, non_blocking=True)
                a_attention_mask = batch["a_attention_mask"].to(DEVICE, non_blocking=True)
                a_char_ids = batch["a_char_ids"].to(DEVICE, non_blocking=True)
                b_input_ids = batch["b_input_ids"].to(DEVICE, non_blocking=True)
                b_attention_mask = batch["b_attention_mask"].to(DEVICE, non_blocking=True)
                b_char_ids = batch["b_char_ids"].to(DEVICE, non_blocking=True)
                targets = batch["targets"].to(DEVICE, non_blocking=True)
                if HP.fp16:
                    with autocast():
                        preds = model(a_input_ids, a_attention_mask, a_char_ids, b_input_ids, b_attention_mask, b_char_ids)
                else:
                    preds = model(a_input_ids, a_attention_mask, a_char_ids, b_input_ids, b_attention_mask, b_char_ids)
                val_preds.append(preds.detach().float().cpu().numpy())
                val_tgts.append(targets.detach().float().cpu().numpy())
        val_preds = np.concatenate(val_preds)
        val_tgts = np.concatenate(val_tgts)

        val_preds_clipped = np.clip(val_preds, 0.0, 1.0)
        pearson = pearsonr_np(val_tgts, val_preds_clipped)
        logging.info(f"Fold {fold} | Epoch {epoch+1} | Val Pearson: {pearson:.6f}")

        if pearson > best_val_pearson:
            best_val_pearson = pearson
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            logging.info(f"Fold {fold} | Epoch {epoch+1} | New best model saved with Pearson {best_val_pearson:.6f}")

    # Load best state
    model.load_state_dict(best_state)
    logging.info(f"Fold {fold} | Loaded best model state with Pearson {best_val_pearson:.6f}")

    # Final validation predictions with best model
    val_loader = DataLoader(val_ds, batch_size=HP.batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)
    model.eval()
    val_preds_final = []
    with torch.no_grad():
        for batch in val_loader:
            a_input_ids = batch["a_input_ids"].to(DEVICE, non_blocking=True)
            a_attention_mask = batch["a_attention_mask"].to(DEVICE, non_blocking=True)
            a_char_ids = batch["a_char_ids"].to(DEVICE, non_blocking=True)
            b_input_ids = batch["b_input_ids"].to(DEVICE, non_blocking=True)
            b_attention_mask = batch["b_attention_mask"].to(DEVICE, non_blocking=True)
            b_char_ids = batch["b_char_ids"].to(DEVICE, non_blocking=True)
            if HP.fp16:
                with autocast():
                    preds = model(a_input_ids, a_attention_mask, a_char_ids, b_input_ids, b_attention_mask, b_char_ids)
            else:
                preds = model(a_input_ids, a_attention_mask, a_char_ids, b_input_ids, b_attention_mask, b_char_ids)
            val_preds_final.append(preds.detach().float().cpu().numpy())
    val_preds_final = np.concatenate(val_preds_final)
    val_preds_final = np.clip(val_preds_final, 0.0, 1.0)

    final_pearson = pearsonr_np(df.iloc[val_idx]["score"].values, val_preds_final)
    logging.info(f"Fold {fold} | Final Val Pearson: {final_pearson:.6f}")

    return final_pearson, val_preds_final, best_state

# ==========================
# Cross-validation training
# ==========================

folds = stratified_kfold_indices(train_df["bin"].values, n_splits=HP.num_folds, seed=SEED)
oof_preds = np.zeros(len(train_df), dtype=np.float32)
fold_scores = []

# Prepare test dataset loader once
test_ds = SiamesePatentDataset(
    test_a_text.tolist(),
    test_b_text.tolist(),
    None,
    None,
    max_word_len=HP.max_word_len,
    max_char_len=HP.max_char_len,
    swap_prob=0.0
)
test_loader = DataLoader(test_ds, batch_size=HP.batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)

fold_test_preds = []

for fold_num, (tr_idx, va_idx) in enumerate(folds):
    logging.info(f"Starting training for fold {fold_num}")
    fold_pearson, val_preds, best_state = train_one_fold(fold_num, tr_idx, va_idx, train_df, train_a_text, train_b_text)
    oof_preds[va_idx] = val_preds
    fold_scores.append(fold_pearson)

    # Instantiate model and load best state for test inference
    model = SiameseRegressor(
        vocab_size=len(word_stoi),
        char_vocab_size=len(char_stoi),
        d_model=HP.d_model,
        nhead=HP.nhead,
        dim_ff=HP.dim_ff,
        num_layers=HP.num_layers,
        dropout=HP.dropout,
        pad_id=PAD_ID,
        char_dim=HP.char_dim,
        char_num_filters=HP.char_num_filters,
        char_kernel_sizes=HP.char_kernel_sizes,
        fusion_hidden=HP.fusion_hidden
    ).to(DEVICE)
    model.load_state_dict(best_state)
    model.eval()

    # Test inference
    test_preds_fold = []
    with torch.no_grad():
        for batch in test_loader:
            a_input_ids = batch["a_input_ids"].to(DEVICE, non_blocking=True)
            a_attention_mask = batch["a_attention_mask"].to(DEVICE, non_blocking=True)
            a_char_ids = batch["a_char_ids"].to(DEVICE, non_blocking=True)
            b_input_ids = batch["b_input_ids"].to(DEVICE, non_blocking=True)
            b_attention_mask = batch["b_attention_mask"].to(DEVICE, non_blocking=True)
            b_char_ids = batch["b_char_ids"].to(DEVICE, non_blocking=True)
            if HP.fp16:
                with autocast():
                    preds = model(a_input_ids, a_attention_mask, a_char_ids, b_input_ids, b_attention_mask, b_char_ids)
            else:
                preds = model(a_input_ids, a_attention_mask, a_char_ids, b_input_ids, b_attention_mask, b_char_ids)
            test_preds_fold.append(preds.detach().float().cpu().numpy())
    test_preds_fold = np.concatenate(test_preds_fold)
    test_preds_fold = np.clip(test_preds_fold, 0.0, 1.0)
    fold_test_preds.append(test_preds_fold)
    logging.info(f"Completed fold {fold_num}: Val Pearson={fold_pearson:.6f}, Test preds shape={test_preds_fold.shape}")

# Final OOF metrics
oof_clip = np.clip(oof_preds, 0.0, 1.0)
oof_pearson = pearsonr_np(train_df["score"].values, oof_clip)
logging.info(f"OOF Pearson (clipped [0,1]): {oof_pearson:.6f}")
logging.info(f"Fold-wise Pearson: {json.dumps([float(x) for x in fold_scores])}")
logging.info(f"Mean Fold Pearson: {np.mean(fold_scores):.6f} | Std: {np.std(fold_scores):.6f}")

# =====================
# Ensemble & Submission
# =====================

logging.info("Averaging test predictions across folds (mean ensemble).")
test_preds_mean = np.mean(np.stack(fold_test_preds, axis=0), axis=0)
test_preds_mean = np.clip(test_preds_mean, 0.0, 1.0)

submission = pd.DataFrame({
    "id": test_df["id"].values,
    "score": test_preds_mean.astype(np.float32)
})
submission.to_csv(SUBMISSION_PATH, index=False)
logging.info(f"Wrote submission to {SUBMISSION_PATH}")
logging.info("Finished refined execution successfully (v2).")