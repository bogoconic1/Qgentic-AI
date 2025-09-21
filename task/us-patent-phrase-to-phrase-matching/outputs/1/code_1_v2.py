import os
import re
import math
import time
import random
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================
# Configuration and Logging
# =========================

BASE_DIR = "task/us-patent-phrase-to-phrase-matching"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "1")
LOG_PATH = os.path.join(OUTPUT_DIR, "code_1_v2.txt")
SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission_2.csv")

# Repro/Training config
SEED = 42
N_FOLDS = 5
EPOCHS = 5
BATCH_SIZE = 192
MAX_LEN = 64

# Model config (refined)
EMBED_DIM = 192
N_HEAD = 6
N_LAYERS = 3
FFN_DIM = 768
DROPOUT = 0.15
TOKEN_TYPE_SIZE = 4  # 0=context, 1=anchor, 2=target, 3=special/pad
CONTEXT_EMBED_DIM = 32
HEAD_HIDDEN = 384

# Optimization config
LR = 2e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
WARMUP_RATIO = 0.1

# Loss weights
W_MSE = 0.6
W_CE = 0.3
W_CORR = 0.1

# Always use CUDA and FP16 as per constraints
DEVICE = torch.device("cuda")

# Prepare paths and logging
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    filename=LOG_PATH,
    filemode="w",
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logging.info("Initialized logging and created output directories for v2 pipeline.")

# ==============
# Reproducibility
# ==============
def set_seed(seed: int):
    logging.info(f"Setting all random seeds to {seed}.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

set_seed(SEED)

# =================
# Data Loading Utils
# =================
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TEST_CSV = os.path.join(BASE_DIR, "test.csv")

logging.info(f"Reading train data from {TRAIN_CSV}.")
train_df = pd.read_csv(TRAIN_CSV)
logging.info(f"Reading test data from {TEST_CSV}.")
test_df = pd.read_csv(TEST_CSV)

logging.info(f"Train shape: {train_df.shape}, columns: {list(train_df.columns)}")
logging.info(f"Test shape:  {test_df.shape}, columns: {list(test_df.columns)}")

# Check for missing values as per findings
train_nulls = train_df.isnull().sum().to_dict()
test_nulls = test_df.isnull().sum().to_dict()
logging.info(f"Train missing values by column: {train_nulls}")
logging.info(f"Test missing values by column: {test_nulls}")

# ==================
# Tokenizer and Vocab
# ==================
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[CTX]", "[EOS]"]
PAD_IDX = 0
UNK_IDX = 1
CLS_IDX = 2
SEP_IDX = 3
CTX_IDX = 4
EOS_IDX = 5

def normalize_text(s: str) -> str:
    s = str(s)
    s = s.lower()
    s = re.sub(r"[^0-9a-zA-Z\-\+_/\.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(s: str):
    s = normalize_text(s)
    if not s:
        return []
    return s.split(" ")

def segment_tokenize(context: str, anchor: str, target: str):
    # Return parallel lists: tokens, token_type_ids
    tokens = ["[CLS]"]
    types = [3]
    tokens.append("[CTX]"); types.append(3)
    ctx_tokens = tokenize(context)
    tokens.extend(ctx_tokens); types.extend([0]*len(ctx_tokens))
    tokens.append("[SEP]"); types.append(3)
    anc_tokens = tokenize(anchor)
    tokens.extend(anc_tokens); types.extend([1]*len(anc_tokens))
    tokens.append("[SEP]"); types.append(3)
    tgt_tokens = tokenize(target)
    tokens.extend(tgt_tokens); types.extend([2]*len(tgt_tokens))
    tokens.append("[EOS]"); types.append(3)
    return tokens, types

def build_vocab_from_dfs(train_df: pd.DataFrame, test_df: pd.DataFrame):
    logging.info("Building vocabulary from segmented corpus.")
    vocab = {}
    inv_vocab = []
    for tok in SPECIAL_TOKENS:
        vocab[tok] = len(vocab)
        inv_vocab.append(tok)
    freq = {}
    # Train
    for row in train_df.itertuples(index=False):
        ctx, anc, tgt = getattr(row, "context"), getattr(row, "anchor"), getattr(row, "target")
        toks, _ = segment_tokenize(ctx, anc, tgt)
        for tok in toks:
            if tok in SPECIAL_TOKENS:
                continue
            freq[tok] = freq.get(tok, 0) + 1
    # Test
    for row in test_df.itertuples(index=False):
        ctx, anc, tgt = getattr(row, "context"), getattr(row, "anchor"), getattr(row, "target")
        toks, _ = segment_tokenize(ctx, anc, tgt)
        for tok in toks:
            if tok in SPECIAL_TOKENS:
                continue
            freq[tok] = freq.get(tok, 0) + 1
    sorted_tokens = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    for tok, _ in sorted_tokens:
        if tok not in vocab:
            vocab[tok] = len(vocab)
            inv_vocab.append(tok)
    logging.info(f"Built vocabulary of size {len(vocab)}.")
    return vocab, inv_vocab

vocab, inv_vocab = build_vocab_from_dfs(train_df, test_df)

def numericalize_with_types(context: str, anchor: str, target: str, vocab: dict, max_len: int):
    toks, types = segment_tokenize(context, anchor, target)
    toks = toks[:max_len]
    types = types[:max_len]
    ids = [vocab.get(tok, UNK_IDX) for tok in toks]
    # Pad
    pad_len = max_len - len(ids)
    if pad_len > 0:
        ids = ids + [PAD_IDX] * pad_len
        types = types + [3] * pad_len
    attn_mask = [1 if idx != PAD_IDX else 0 for idx in ids]
    return np.array(ids, dtype=np.int64), np.array(attn_mask, dtype=np.int64), np.array(types, dtype=np.int64)

def preprocess_dataframe(df: pd.DataFrame, vocab: dict, max_len: int):
    logging.info(f"Tokenizing and numericalizing dataframe with {len(df)} rows (with token types).")
    n = len(df)
    input_ids = np.zeros((n, max_len), dtype=np.int64)
    attention_masks = np.zeros((n, max_len), dtype=np.int64)
    token_types = np.zeros((n, max_len), dtype=np.int64)
    for i, row in enumerate(df.itertuples(index=False)):
        ids, mask, types = numericalize_with_types(getattr(row, "context"), getattr(row, "anchor"), getattr(row, "target"), vocab, max_len)
        input_ids[i] = ids
        attention_masks[i] = mask
        token_types[i] = types
        if (i + 1) % 5000 == 0:
            logging.info(f"Processed {i + 1} rows.")
    return input_ids, attention_masks, token_types

train_ids, train_masks, train_types = preprocess_dataframe(train_df, vocab, MAX_LEN)
test_ids, test_masks, test_types = preprocess_dataframe(test_df, vocab, MAX_LEN)

labels_float = train_df["score"].values.astype(np.float32)
labels_cls = (labels_float * 4.0 + 1e-6).astype(np.int64)

logging.info(f"Labels statistics: min={labels_float.min()}, max={labels_float.max()}, mean={labels_float.mean():.4f}, std={labels_float.std():.4f}")

# Context indexing
contexts_all = sorted(list(set(train_df["context"].astype(str).tolist() + test_df["context"].astype(str).tolist())))
context2id = {c: i for i, c in enumerate(contexts_all)}
train_ctx_ids = np.array([context2id[c] for c in train_df["context"].astype(str).tolist()], dtype=np.int64)
test_ctx_ids = np.array([context2id[c] for c in test_df["context"].astype(str).tolist()], dtype=np.int64)
logging.info(f"Found {len(contexts_all)} unique context codes across train+test.")

# Class weights for CE (inverse frequency)
class_counts = np.bincount(labels_cls, minlength=5).astype(np.float32)
class_weights = (1.0 / np.maximum(class_counts, 1.0))
class_weights = class_weights / class_weights.max()
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
logging.info(f"Class counts: {class_counts.tolist()} | CE weights: {class_weights.tolist()}")

# ================
# Stratified K-Fold
# ================
def make_stratified_folds(y_cls: np.ndarray, n_folds: int, seed: int):
    logging.info("Creating stratified folds on 5-class bins.")
    bin_to_indices = {}
    for idx, b in enumerate(y_cls):
        bin_to_indices.setdefault(int(b), []).append(idx)
    rng = np.random.RandomState(seed)
    for b in bin_to_indices:
        rng.shuffle(bin_to_indices[b])
    folds = [[] for _ in range(n_folds)]
    for b, idxs in bin_to_indices.items():
        for i, idx in enumerate(idxs):
            folds[i % n_folds].append(idx)
    fold_arrays = [np.array(sorted(f), dtype=np.int64) for f in folds]
    for i, f in enumerate(fold_arrays):
        logging.info(f"Fold {i}: {len(f)} samples.")
    return fold_arrays

fold_indices = make_stratified_folds(labels_cls, N_FOLDS, SEED)

# ==============
# PyTorch Dataset
# ==============
class PatentDataset(Dataset):
    def __init__(self, input_ids: np.ndarray, attention_masks: np.ndarray, token_types: np.ndarray, context_ids: np.ndarray, labels_float: np.ndarray = None, labels_cls: np.ndarray = None):
        logging.info(f"Initializing PatentDataset with {len(input_ids)} samples.")
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.token_types = token_types
        self.context_ids = context_ids
        self.labels_float = labels_float
        self.labels_cls = labels_cls

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.from_numpy(self.input_ids[idx]),
            "attention_mask": torch.from_numpy(self.attention_masks[idx]),
            "token_types": torch.from_numpy(self.token_types[idx]),
            "context_ids": torch.tensor(self.context_ids[idx], dtype=torch.long),
        }
        if self.labels_float is not None:
            item["labels_float"] = torch.tensor(self.labels_float[idx], dtype=torch.float32)
        if self.labels_cls is not None:
            item["labels_cls"] = torch.tensor(self.labels_cls[idx], dtype=torch.long)
        return item

# =======================
# Transformer Regressor++
# =======================
class TransformerRegressor(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, n_head: int, num_layers: int, ffn_dim: int, max_len: int, dropout: float,
                 token_type_size: int, num_contexts: int, context_embed_dim: int, head_hidden: int):
        super().__init__()
        logging.info(f"Initializing TransformerRegressor++ with vocab_size={vocab_size}, embed_dim={embed_dim}, heads={n_head}, layers={num_layers}, max_len={max_len}, ctxs={num_contexts}")
        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        self.seg_emb = nn.Embedding(token_type_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_head,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=False,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Context embedding
        self.context_emb = nn.Embedding(num_contexts, context_embed_dim)

        # Attention pooling vector
        self.attn_vec = nn.Linear(embed_dim, 1, bias=False)

        # Head over concatenated features: [all, anchor, target, |a-b|, a*b, ctx]
        feat_dim = embed_dim * 5 + context_embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head_reg = nn.Linear(head_hidden, 1)
        self.head_cls = nn.Linear(head_hidden, 5)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def masked_mean(self, x, mask):
        # x: (B, L, D), mask: (B, L) float in {0,1}
        mask = mask.unsqueeze(-1)
        summed = (x * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts

    def masked_attn_pool(self, x, mask):
        # x: (B, L, D), mask: (B, L) float in {0,1}
        logits = self.attn_vec(x).squeeze(-1)  # (B, L)
        logits = logits.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(logits, dim=1)  # (B, L)
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (B, D)
        return pooled

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_types: torch.Tensor, context_ids: torch.Tensor):
        B, L = input_ids.shape
        pos = torch.arange(0, L, device=input_ids.device, dtype=torch.long).unsqueeze(0).expand(B, L)
        tok = self.token_emb(input_ids)
        pos_e = self.pos_emb(pos)
        seg = self.seg_emb(token_types)
        x = tok + pos_e + seg
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = x.transpose(0, 1)  # (L, B, D)
        key_padding_mask = attention_mask == 0  # (B, L)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = x.transpose(0, 1)  # (B, L, D)

        attn_mask = attention_mask.float()
        pooled_all_mean = self.masked_mean(x, attn_mask)                 # (B, D)
        pooled_all_attn = self.masked_attn_pool(x, attn_mask)            # (B, D)
        pooled_all = 0.5 * (pooled_all_mean + pooled_all_attn)           # (B, D)

        # Segment-specific masks
        mask_anchor = (token_types == 1).float() * attn_mask
        mask_target = (token_types == 2).float() * attn_mask
        pooled_anchor = self.masked_mean(x, mask_anchor)
        pooled_target = self.masked_mean(x, mask_target)

        # Cross features
        ab_diff = torch.abs(pooled_anchor - pooled_target)
        ab_prod = pooled_anchor * pooled_target

        ctx_e = self.context_emb(context_ids)  # (B, C)
        feats = torch.cat([pooled_all, pooled_anchor, pooled_target, ab_diff, ab_prod, ctx_e], dim=-1)
        h = self.mlp(feats)
        reg_logits = self.head_reg(h).squeeze(-1)
        reg = torch.sigmoid(reg_logits)  # [0,1]
        cls_logits = self.head_cls(h)
        return reg, cls_logits

# ==============
# Loss and Metric
# ==============
def mse_loss(preds: torch.Tensor, targets: torch.Tensor):
    return torch.mean((preds - targets) ** 2)

def ce_loss(logits: torch.Tensor, targets: torch.Tensor, weight: torch.Tensor):
    return nn.functional.cross_entropy(logits, targets, weight=weight)

def pearson_corrcoef(preds: torch.Tensor, targets: torch.Tensor):
    x = preds
    y = targets
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
    return corr

def corr_loss(preds: torch.Tensor, targets: torch.Tensor):
    return 1.0 - pearson_corrcoef(preds, targets)

def pearsonr_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.ndim > 1:
        y_true = y_true.reshape(-1)
    if y_pred.ndim > 1:
        y_pred = y_pred.reshape(-1)
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    y_true_mean = y_true.mean()
    y_pred_mean = y_pred.mean()
    num = np.sum((y_true - y_true_mean) * (y_pred - y_pred_mean))
    den = math.sqrt(np.sum((y_true - y_true_mean) ** 2) * np.sum((y_pred - y_pred_mean) ** 2))
    if den == 0.0:
        return 0.0
    return float(num / den)

# ===================
# Scheduler (Warmup+Cosine)
# ===================
def get_warmup_cosine_scheduler(optimizer, num_warmup_steps, num_training_steps):
    logging.info(f"Creating Warmup+Cosine scheduler: warmup_steps={num_warmup_steps}, total_steps={num_training_steps}.")
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ===================
# Training and Eval
# ===================
def train_one_epoch(model, loader, optimizer, scaler, scheduler=None):
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_ce = 0.0
    total_corr = 0.0
    steps = 0
    start_time = time.time()
    for batch_idx, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        token_types = batch["token_types"].to(DEVICE, non_blocking=True)
        context_ids = batch["context_ids"].to(DEVICE, non_blocking=True)
        labels_f = batch["labels_float"].to(DEVICE, non_blocking=True)
        labels_c = batch["labels_cls"].to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            pred_reg, pred_logits = model(input_ids, attention_mask, token_types, context_ids)
            loss_mse = mse_loss(pred_reg, labels_f)
            loss_ce = ce_loss(pred_logits, labels_c, class_weights_tensor)
            loss_corr = corr_loss(pred_reg, labels_f)
            loss = W_MSE * loss_mse + W_CE * loss_ce + W_CORR * loss_corr

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.detach().float().item()
        total_mse += loss_mse.detach().float().item()
        total_ce += loss_ce.detach().float().item()
        total_corr += loss_corr.detach().float().item()
        steps += 1

        if (batch_idx + 1) % 50 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            logging.info(f"Train step {batch_idx + 1}/{len(loader)} - Loss: {total_loss / steps:.6f} (MSE {total_mse/steps:.6f}, CE {total_ce/steps:.6f}, Corr {total_corr/steps:.6f}) | LR: {current_lr:.6e}")

    elapsed = time.time() - start_time
    avg_loss = total_loss / max(steps, 1)
    logging.info(f"Epoch training completed in {elapsed:.2f}s with average loss {avg_loss:.6f}")
    return avg_loss

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    total_mse = 0.0
    total_ce = 0.0
    steps = 0
    for batch_idx, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        token_types = batch["token_types"].to(DEVICE, non_blocking=True)
        context_ids = batch["context_ids"].to(DEVICE, non_blocking=True)
        labels_f = batch["labels_float"].to(DEVICE, non_blocking=True)
        labels_c = batch["labels_cls"].to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            preds_reg, preds_logits = model(input_ids, attention_mask, token_types, context_ids)
            loss_mse = mse_loss(preds_reg, labels_f)
            loss_ce = ce_loss(preds_logits, labels_c, class_weights_tensor)
        total_mse += loss_mse.detach().float().item()
        total_ce += loss_ce.detach().float().item()
        steps += 1
        all_preds.append(preds_reg.detach().float().cpu().numpy())
        all_labels.append(labels_f.detach().float().cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    pearson = pearsonr_np(all_labels, all_preds)
    avg_mse = total_mse / max(steps, 1)
    avg_ce = total_ce / max(steps, 1)
    return avg_mse, avg_ce, pearson, all_preds, all_labels

@torch.no_grad()
def predict(model, loader):
    model.eval()
    all_preds = []
    for batch_idx, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        token_types = batch["token_types"].to(DEVICE, non_blocking=True)
        context_ids = batch["context_ids"].to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            preds_reg, _ = model(input_ids, attention_mask, token_types, context_ids)
        all_preds.append(preds_reg.detach().float().cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    return all_preds

# ===============
# Cross-Validation
# ===============
oof_preds = np.zeros(len(train_df), dtype=np.float32)
test_preds_folds = []

# Build test dataset/loader once
test_dataset = PatentDataset(test_ids, test_masks, test_types, test_ctx_ids, labels_float=None, labels_cls=None)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

for fold_id in range(N_FOLDS):
    logging.info(f"========== Fold {fold_id + 1}/{N_FOLDS} ==========")
    val_idx = fold_indices[fold_id]
    train_idx = np.array(sorted(list(set(range(len(train_df))) - set(val_idx.tolist()))), dtype=np.int64)

    x_train_ids = train_ids[train_idx]
    x_train_masks = train_masks[train_idx]
    x_train_types = train_types[train_idx]
    x_train_ctx = train_ctx_ids[train_idx]
    y_train_f = labels_float[train_idx]
    y_train_c = labels_cls[train_idx]

    x_val_ids = train_ids[val_idx]
    x_val_masks = train_masks[val_idx]
    x_val_types = train_types[val_idx]
    x_val_ctx = train_ctx_ids[val_idx]
    y_val_f = labels_float[val_idx]
    y_val_c = labels_cls[val_idx]

    logging.info(f"Train split: {len(train_idx)} samples; Val split: {len(val_idx)} samples.")
    train_dataset = PatentDataset(x_train_ids, x_train_masks, x_train_types, x_train_ctx, y_train_f, y_train_c)
    val_dataset = PatentDataset(x_val_ids, x_val_masks, x_val_types, x_val_ctx, y_val_f, y_val_c)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    model = TransformerRegressor(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        n_head=N_HEAD,
        num_layers=N_LAYERS,
        ffn_dim=FFN_DIM,
        max_len=MAX_LEN,
        dropout=DROPOUT,
        token_type_size=TOKEN_TYPE_SIZE,
        num_contexts=len(contexts_all),
        context_embed_dim=CONTEXT_EMBED_DIM,
        head_hidden=HEAD_HIDDEN
    )
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    total_steps = EPOCHS * max(1, math.ceil(len(train_loader)))
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler()

    best_val_pearson = -1.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        logging.info(f"Fold {fold_id} - Epoch {epoch}/{EPOCHS} started.")
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, scheduler)
        val_mse, val_ce, val_pearson, val_pred, val_true = evaluate(model, val_loader)
        logging.info(f"Fold {fold_id} - Epoch {epoch} - Train Loss: {train_loss:.6f} | Val MSE: {val_mse:.6f} | Val CE: {val_ce:.6f} | Val Pearson: {val_pearson:.6f}")
        if val_pearson > best_val_pearson:
            best_val_pearson = val_pearson
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            logging.info(f"Fold {fold_id} - Epoch {epoch} - New best model with Pearson {best_val_pearson:.6f}.")

    # Load best state and generate predictions
    model.load_state_dict(best_state)
    val_mse_final, val_ce_final, val_pearson_final, val_pred_final, val_true_final = evaluate(model, val_loader)
    logging.info(f"Fold {fold_id} - Final Val Pearson: {val_pearson_final:.6f} | Final Val MSE: {val_mse_final:.6f} | Final Val CE: {val_ce_final:.6f}")
    # Store OOF
    oof_preds[val_idx] = val_pred_final.astype(np.float32)

    # Predict on test
    fold_test_preds = predict(model, test_loader).astype(np.float32)
    test_preds_folds.append(fold_test_preds)

    # Clean up GPU memory between folds
    del model
    del optimizer
    del scheduler
    del scaler
    torch.cuda.empty_cache()

# ==========================
# OOF Performance and Logging
# ==========================
oof_pearson = pearsonr_np(labels_float, oof_preds)
oof_mse = float(np.mean((labels_float - oof_preds) ** 2))
logging.info(f"OOF Pearson correlation across all folds (pre-rule): {oof_pearson:.6f}")
logging.info(f"OOF MSE across all folds (pre-rule): {oof_mse:.6f}")

# ==========================
# Ensemble Test Predictions
# ==========================
logging.info("Ensembling test predictions from all folds by averaging.")
test_preds_stack = np.stack(test_preds_folds, axis=0)  # (F, N)
test_preds_mean = np.mean(test_preds_stack, axis=0)    # (N,)

# ==========================
# Rule-based Post-processing
# ==========================
logging.info("Applying rule-based post-processing for anchor == target -> score = 1.0")
anchor_eq_target_mask_test = (test_df["anchor"].astype(str).values == test_df["target"].astype(str).values)
num_equal_test = int(anchor_eq_target_mask_test.sum())
logging.info(f"Found {num_equal_test} exact anchor==target pairs in test set.")
test_preds_mean[anchor_eq_target_mask_test] = 1.0

anchor_eq_target_mask_oof = (train_df["anchor"].astype(str).values == train_df["target"].astype(str).values)
num_equal_oof = int(anchor_eq_target_mask_oof.sum())
logging.info(f"Found {num_equal_oof} exact anchor==target pairs in train (OOF) set; applying same rule to OOF predictions.")
oof_preds[anchor_eq_target_mask_oof] = 1.0

# Recompute OOF metric after rule application
oof_pearson_rule = pearsonr_np(labels_float, oof_preds)
oof_mse_rule = float(np.mean((labels_float - oof_preds) ** 2))
logging.info(f"OOF Pearson after rule: {oof_pearson_rule:.6f}")
logging.info(f"OOF MSE after rule: {oof_mse_rule:.6f}")
logging.info("Logging final validation results (OOF, post-rule):")
logging.info(f"Final Validation Pearson: {oof_pearson_rule:.6f} | Final Validation MSE: {oof_mse_rule:.6f}")

# ===============
# Build Submission
# ===============
submission = pd.DataFrame({
    "id": test_df["id"].values,
    "score": test_preds_mean
})
submission["score"] = submission["score"].clip(0.0, 1.0)

logging.info(f"Writing submission to {SUBMISSION_PATH}")
submission.to_csv(SUBMISSION_PATH, index=False)
logging.info("Submission file created successfully.")
logging.info("All done (v2).")