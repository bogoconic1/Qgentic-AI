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
LOG_PATH = os.path.join(OUTPUT_DIR, "code_1_v8.txt")
SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission_8.csv")

# Repro/Training config
SEED = 42
N_FOLDS = 5
EPOCHS = 6
BATCH_SIZE = 192
MAX_LEN = 64

# Model config
EMBED_DIM = 256
N_HEAD = 8
N_LAYERS = 4
FFN_DIM = 1024
DROPOUT = 0.15
MS_DROPOUTS = 5
TOKEN_TYPE_SIZE = 4  # 0=context, 1=anchor, 2=target, 3=special/pad
CONTEXT_EMBED_DIM = 64
HEAD_HIDDEN = 640
ALPHA_REG_FUSION = 0.6  # final_pred = alpha*reg + (1-alpha)*expected_from_cls
FEAT_EMBED_DIM = 32     # projection size for handcrafted features

# Optimization config
LR = 2.0e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
WARMUP_RATIO = 0.1
EMA_DECAY = 0.998

# Loss weights
W_MSE = 0.55
W_SOFTCE = 0.30
W_CORR = 0.08
W_CONSISTENCY = 0.02       # consistency between reg and expected score
W_COSSIM_AUX = 0.05        # auxiliary cosine-similarity loss weight
W_RANK = 0.05              # pairwise ranking loss weight
W_RDROP_CLS = 0.02         # R-Drop for classification head
W_RDROP_REG = 0.01         # R-Drop for regression head

# Ranking loss config
RANK_PAIRS = 256
RANK_MARGIN = 0.125

# Augmentation and inference
USE_SWAP_AUG = True
WORD_DROPOUT_PROB = 0.05   # token-level dropout to [UNK] during training
USE_TTA_SWAP = True        # test/val-time anchor<->target swap prediction average

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
logging.info("Initialized logging and created output directories for v8 pipeline.")

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

# Check for missing values
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

def preprocess_dataframe_swapped(df: pd.DataFrame, vocab: dict, max_len: int):
    logging.info(f"Tokenizing and numericalizing SWAPPED dataframe with {len(df)} rows (anchor<->target).")
    n = len(df)
    input_ids = np.zeros((n, max_len), dtype=np.int64)
    attention_masks = np.zeros((n, max_len), dtype=np.int64)
    token_types = np.zeros((n, max_len), dtype=np.int64)
    for i, row in enumerate(df.itertuples(index=False)):
        ids, mask, types = numericalize_with_types(getattr(row, "context"), getattr(row, "target"), getattr(row, "anchor"), vocab, max_len)
        input_ids[i] = ids
        attention_masks[i] = mask
        token_types[i] = types
        if (i + 1) % 5000 == 0:
            logging.info(f"Processed {i + 1} swapped rows.")
    return input_ids, attention_masks, token_types

train_ids, train_masks, train_types = preprocess_dataframe(train_df, vocab, MAX_LEN)
test_ids, test_masks, test_types = preprocess_dataframe(test_df, vocab, MAX_LEN)
if USE_TTA_SWAP:
    test_ids_sw, test_masks_sw, test_types_sw = preprocess_dataframe_swapped(test_df, vocab, MAX_LEN)
    logging.info("Prepared swapped TTA arrays for test data.")

# Precompute swapped augmentation for train
if USE_SWAP_AUG:
    train_ids_sw, train_masks_sw, train_types_sw = preprocess_dataframe_swapped(train_df, vocab, MAX_LEN)
    logging.info("Prepared swapped augmentation arrays for training.")
else:
    train_ids_sw = None
    train_masks_sw = None
    train_types_sw = None

# ======================
# Handcrafted Features
# ======================
def lcs_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i

def levenshtein_distance(a: str, b: str) -> int:
    la = len(a)
    lb = len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    curr = [0] * (lb + 1)
    for i in range(1, la + 1):
        curr[0] = i
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cb = b[j - 1]
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[lb]

def compute_features(anchors: pd.Series, targets: pd.Series):
    logging.info("Computing handcrafted similarity features.")
    n = len(anchors)
    feats = np.zeros((n, 8), dtype=np.float32)
    for i in range(n):
        a_raw = str(anchors.iloc[i])
        b_raw = str(targets.iloc[i])
        a = normalize_text(a_raw)
        b = normalize_text(b_raw)
        ta = tokenize(a)
        tb = tokenize(b)
        sa = set(ta)
        sb = set(tb)
        inter = len(sa & sb)
        union = len(sa | sb)
        mina = min(len(sa), len(sb)) if min(len(sa), len(sb)) > 0 else 1
        maxa = max(len(sa), len(sb)) if max(len(sa), len(sb)) > 0 else 1
        jacc = inter / union if union > 0 else 0.0
        overlap = inter / mina if mina > 0 else 0.0
        len_ratio_tok = (mina / maxa) if maxa > 0 else 0.0
        la = len(a)
        lb = len(b)
        mx = max(la, lb) if max(la, lb) > 0 else 1
        lcp = lcs_prefix_len(a, b)
        lcp_norm = lcp / mx
        lev = levenshtein_distance(a, b)
        sim_lev = 1.0 - (lev / mx)
        eq_flag = 1.0 if a_raw == b_raw else 0.0
        starts_flag = 1.0 if a.startswith(b) or b.startswith(a) else 0.0
        feats[i, 0] = jacc
        feats[i, 1] = overlap
        feats[i, 2] = len_ratio_tok
        feats[i, 3] = lcp_norm
        feats[i, 4] = sim_lev
        feats[i, 5] = eq_flag
        feats[i, 6] = starts_flag
        feats[i, 7] = (len(ta) + len(tb)) / 20.0
        if (i + 1) % 5000 == 0:
            logging.info(f"Computed handcrafted features for {i + 1} rows.")
    logging.info("Finished computing handcrafted features.")
    return feats

train_feats = compute_features(train_df["anchor"], train_df["target"])
test_feats = compute_features(test_df["anchor"], test_df["target"])
FEAT_DIM = train_feats.shape[1]
logging.info(f"Handcrafted feature dimension: {FEAT_DIM}")

labels_float = train_df["score"].values.astype(np.float32)
labels_cls = (labels_float * 4.0 + 1e-6).astype(np.int64)
logging.info(f"Labels statistics: min={labels_float.min()}, max={labels_float.max()}, mean={labels_float.mean():.4f}, std={labels_float.std():.4f}")

# Context indexing
contexts_all = sorted(list(set(train_df["context"].astype(str).tolist() + test_df["context"].astype(str).tolist())))
context2id = {c: i for i, c in enumerate(contexts_all)}
train_ctx_ids = np.array([context2id[c] for c in train_df["context"].astype(str).tolist()], dtype=np.int64)
test_ctx_ids = np.array([context2id[c] for c in test_df["context"].astype(str).tolist()], dtype=np.int64)
logging.info(f"Found {len(contexts_all)} unique context codes across train+test.")

# Class weights for soft CE
class_counts = np.bincount(labels_cls, minlength=5).astype(np.float32)
class_weights = (1.0 / np.maximum(class_counts, 1.0))
class_weights = class_weights / class_weights.max()
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
logging.info(f"Class counts: {class_counts.tolist()} | CE weights: {class_weights.tolist()}")

# Target values for class expectation
class_values = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float32, device=DEVICE)

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
    def __init__(self, input_ids: np.ndarray, attention_masks: np.ndarray, token_types: np.ndarray, context_ids: np.ndarray,
                 features: np.ndarray, labels_float: np.ndarray = None, labels_cls: np.ndarray = None,
                 is_training: bool = False, word_dropout_prob: float = 0.0):
        logging.info(f"Initializing PatentDataset with {len(input_ids)} samples.")
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.token_types = token_types
        self.context_ids = context_ids
        self.features = features
        self.labels_float = labels_float
        self.labels_cls = labels_cls
        self.is_training = is_training
        self.word_dropout_prob = float(word_dropout_prob)

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        ids = torch.from_numpy(self.input_ids[idx]).clone()
        mask = torch.from_numpy(self.attention_masks[idx]).clone()
        types = torch.from_numpy(self.token_types[idx]).clone()
        ctx_id = torch.tensor(self.context_ids[idx], dtype=torch.long)
        feats = torch.tensor(self.features[idx], dtype=torch.float32)
        if self.is_training and self.word_dropout_prob > 0.0:
            eligible = (mask == 1) & (types != 3) & (ids != PAD_IDX) & (ids != CLS_IDX) & (ids != SEP_IDX) & (ids != CTX_IDX) & (ids != EOS_IDX)
            if eligible.any():
                drop = torch.rand_like(ids, dtype=torch.float32) < self.word_dropout_prob
                drop = drop & eligible
                ids[drop] = UNK_IDX
        item = {
            "input_ids": ids,
            "attention_mask": mask,
            "token_types": types,
            "context_ids": ctx_id,
            "features": feats,
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
                 token_type_size: int, num_contexts: int, context_embed_dim: int, head_hidden: int,
                 feat_dim: int, feat_embed_dim: int, ms_dropouts: int):
        super().__init__()
        logging.info(f"Initializing TransformerRegressor++ with vocab_size={vocab_size}, embed_dim={embed_dim}, heads={n_head}, layers={num_layers}, max_len={max_len}, ctxs={num_contexts}, feat_dim={feat_dim}, msd={ms_dropouts}")
        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        self.seg_emb = nn.Embedding(token_type_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_head,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Context embedding
        self.context_emb = nn.Embedding(num_contexts, context_embed_dim)

        # Attention pooling vector
        self.attn_vec = nn.Linear(embed_dim, 1, bias=False)

        # Feature projector
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, feat_embed_dim),
            nn.GELU(),
        )

        # Projection for cosine similarity auxiliary
        self.sim_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Head over concatenated features: [all, anchor, target, |a-b|, a*b, ctx, feat_emb]
        feat_cat_dim = embed_dim * 5 + context_embed_dim + feat_embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(feat_cat_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.ms_drops = nn.ModuleList([nn.Dropout(dropout) for _ in range(ms_dropouts)])
        self.head_reg = nn.Linear(head_hidden, 1)
        self.head_cls = nn.Linear(head_hidden, 5)

        # Context-specific scalar bias to reg head
        self.ctx_bias = nn.Embedding(num_contexts, 1)

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
        mask = mask.unsqueeze(-1)
        summed = (x * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts

    def masked_attn_pool(self, x, mask):
        bool_mask = mask.to(torch.bool)
        logits = self.attn_vec(x).squeeze(-1)
        logits32 = logits.float().masked_fill(~bool_mask, float('-inf'))
        attn = torch.softmax(logits32, dim=-1)
        attn = attn.to(x.dtype)
        pooled = torch.bmm(attn.unsqueeze(1), x).squeeze(1)
        return pooled

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_types: torch.Tensor, context_ids: torch.Tensor, features: torch.Tensor):
        B, L = input_ids.shape
        pos = torch.arange(0, L, device=input_ids.device, dtype=torch.long).unsqueeze(0).expand(B, L)
        tok = self.token_emb(input_ids)
        pos_e = self.pos_emb(pos)
        seg = self.seg_emb(token_types)
        x = tok + pos_e + seg
        x = self.layer_norm(x)
        x = self.dropout(x)
        key_padding_mask = attention_mask == 0
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)

        attn_mask = attention_mask.float()
        pooled_all_mean = self.masked_mean(x, attn_mask)
        pooled_all_attn = self.masked_attn_pool(x, attn_mask)
        pooled_all = 0.5 * (pooled_all_mean + pooled_all_attn)

        mask_anchor = (token_types == 1).float() * attn_mask
        mask_target = (token_types == 2).float() * attn_mask
        pooled_anchor = self.masked_mean(x, mask_anchor)
        pooled_target = self.masked_mean(x, mask_target)

        ab_diff = torch.abs(pooled_anchor - pooled_target)
        ab_prod = pooled_anchor * pooled_target

        ctx_e = self.context_emb(context_ids)
        feat_emb = self.feat_proj(features)

        feats = torch.cat([pooled_all, pooled_anchor, pooled_target, ab_diff, ab_prod, ctx_e, feat_emb], dim=-1)
        h = self.mlp(feats)

        # Multi-sample dropout heads
        reg_logits_acc = 0.0
        cls_logits_acc = 0.0
        for drop in self.ms_drops:
            h_drop = drop(h)
            reg_logits_acc = reg_logits_acc + self.head_reg(h_drop).squeeze(-1)
            cls_logits_acc = cls_logits_acc + self.head_cls(h_drop)
        reg_logits = reg_logits_acc / float(len(self.ms_drops))
        cls_logits = cls_logits_acc / float(len(self.ms_drops))

        # add context bias to regression logits
        reg_logits = reg_logits + self.ctx_bias(context_ids).squeeze(-1)

        reg = torch.sigmoid(reg_logits)

        pa = nn.functional.normalize(self.sim_proj(pooled_anchor), dim=-1)
        pt = nn.functional.normalize(self.sim_proj(pooled_target), dim=-1)
        return reg, cls_logits, pa, pt

# ==============
# Loss and Metric
# ==============
def mse_loss(preds: torch.Tensor, targets: torch.Tensor):
    return torch.mean((preds - targets) ** 2)

def soft_targets_from_scores(scores: torch.Tensor, sigma: float = 0.20):
    # scores: (B,)
    centers = class_values.view(1, -1)  # (1,5)
    s = scores.view(-1, 1)              # (B,1)
    dist2 = (s - centers) ** 2          # (B,5)
    weights = torch.exp(-dist2 / (2.0 * (sigma ** 2)))  # (B,5)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)
    return weights

def soft_ce_with_class_weights(logits: torch.Tensor, soft_targets: torch.Tensor, class_weights: torch.Tensor):
    log_probs = nn.functional.log_softmax(logits, dim=-1)  # (B,5)
    w = class_weights.view(1, -1)                          # (1,5)
    loss_vec = -(soft_targets * w) * log_probs
    loss = loss_vec.sum(dim=-1).mean()
    return loss

def pearson_corrcoef(preds: torch.Tensor, targets: torch.Tensor):
    x = preds
    y = targets
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
    return corr

def corr_loss(preds: torch.Tensor, targets: torch.Tensor):
    return 1.0 - pearson_corrcoef(preds, targets)

def pairwise_ranking_loss(preds: torch.Tensor, targets: torch.Tensor, pairs: int, margin: float):
    B = preds.shape[0]
    idx1 = torch.randint(0, B, (pairs,), device=preds.device)
    idx2 = torch.randint(0, B, (pairs,), device=preds.device)
    y1 = targets[idx1]
    y2 = targets[idx2]
    p1 = preds[idx1]
    p2 = preds[idx2]
    ydiff = y1 - y2
    sd = torch.sign(ydiff)
    mask = (torch.abs(ydiff) >= margin)
    if mask.any():
        sd = sd[mask]
        p1 = p1[mask]
        p2 = p2[mask]
        logits = (p1 - p2) * sd
        loss = -nn.functional.logsigmoid(logits).mean()
        return loss
    else:
        return torch.tensor(0.0, device=preds.device, dtype=preds.dtype)

def symm_kl_with_logit_pairs(logits1: torch.Tensor, logits2: torch.Tensor):
    logp = nn.functional.log_softmax(logits1, dim=-1)
    logq = nn.functional.log_softmax(logits2, dim=-1)
    p = logp.exp()
    q = logq.exp()
    kl_pq = nn.functional.kl_div(logp, q, reduction="batchmean", log_target=False)
    kl_qp = nn.functional.kl_div(logq, p, reduction="batchmean", log_target=False)
    return 0.5 * (kl_pq + kl_qp)

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
# Helpers
# ===================
def fuse_prediction(reg: torch.Tensor, cls_logits: torch.Tensor):
    probs = torch.softmax(cls_logits, dim=-1)  # (B, 5)
    expected = torch.sum(probs * class_values.unsqueeze(0), dim=-1)  # (B,)
    final_pred = ALPHA_REG_FUSION * reg + (1.0 - ALPHA_REG_FUSION) * expected
    return final_pred, expected

def init_ema_state(model: nn.Module):
    logging.info("Initializing EMA state from current model weights.")
    state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
    return state

def update_ema_state(ema_state: dict, model: nn.Module, decay: float):
    msd = model.state_dict()
    for k, v in msd.items():
        ema_state[k].mul_(decay).add_(v.detach().clone().cpu(), alpha=(1.0 - decay))

def evaluate_with_state(model: nn.Module, loader, state_dict_to_use: dict):
    logging.info("Evaluating with a provided state dict (e.g., EMA).")
    current = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
    model.load_state_dict(state_dict_to_use)
    model.to(DEVICE)
    val_mse, val_ce, val_pearson, val_pred, val_true = evaluate(model, loader)
    model.load_state_dict(current)
    model.to(DEVICE)
    return val_mse, val_ce, val_pearson, val_pred, val_true

# ===================
# Training and Eval
# ===================
def train_one_epoch(model, loader, optimizer, scaler, scheduler=None, ema_state=None):
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_ce = 0.0
    total_corr = 0.0
    total_cons = 0.0
    total_caux = 0.0
    total_rank = 0.0
    total_rdrop_cls = 0.0
    total_rdrop_reg = 0.0
    steps = 0
    start_time = time.time()
    for batch_idx, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        token_types = batch["token_types"].to(DEVICE, non_blocking=True)
        context_ids = batch["context_ids"].to(DEVICE, non_blocking=True)
        features = batch["features"].to(DEVICE, non_blocking=True)
        labels_f = batch["labels_float"].to(DEVICE, non_blocking=True)
        labels_soft = soft_targets_from_scores(labels_f, sigma=0.20)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            # Forward pass 1
            reg1, cls_logits1, pa1, pt1 = model(input_ids, attention_mask, token_types, context_ids, features)
            fused1, expected1 = fuse_prediction(reg1, cls_logits1)
            loss_mse = mse_loss(fused1, labels_f)
            loss_softce = soft_ce_with_class_weights(cls_logits1, labels_soft, class_weights_tensor)
            loss_corr = corr_loss(fused1, labels_f)
            loss_cons = mse_loss(reg1, expected1.detach())
            cos_sim1 = nn.functional.cosine_similarity(pa1, pt1, dim=-1)
            cos_sim1_scaled = 0.5 * (cos_sim1 + 1.0)
            loss_caux = mse_loss(cos_sim1_scaled, labels_f)
            loss_rank = pairwise_ranking_loss(fused1, labels_f, RANK_PAIRS, RANK_MARGIN)

            # Forward pass 2 (R-Drop)
            reg2, cls_logits2, _, _ = model(input_ids, attention_mask, token_types, context_ids, features)
            loss_rdrop_cls = symm_kl_with_logit_pairs(cls_logits1, cls_logits2)
            loss_rdrop_reg = mse_loss(reg1, reg2)

            loss = (W_MSE * loss_mse + W_SOFTCE * loss_softce + W_CORR * loss_corr +
                    W_CONSISTENCY * loss_cons + W_COSSIM_AUX * loss_caux + W_RANK * loss_rank +
                    W_RDROP_CLS * loss_rdrop_cls + W_RDROP_REG * loss_rdrop_reg)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        if ema_state is not None:
            update_ema_state(ema_state, model, EMA_DECAY)

        total_loss += loss.detach().float().item()
        total_mse += loss_mse.detach().float().item()
        total_ce += loss_softce.detach().float().item()
        total_corr += loss_corr.detach().float().item()
        total_cons += loss_cons.detach().float().item()
        total_caux += loss_caux.detach().float().item()
        total_rank += loss_rank.detach().float().item()
        total_rdrop_cls += loss_rdrop_cls.detach().float().item()
        total_rdrop_reg += loss_rdrop_reg.detach().float().item()
        steps += 1

        if (batch_idx + 1) % 50 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            logging.info(f"Train step {batch_idx + 1}/{len(loader)} - Loss: {total_loss / steps:.6f} (MSE {total_mse/steps:.6f}, SoftCE {total_ce/steps:.6f}, Corr {total_corr/steps:.6f}, Cons {total_cons/steps:.6f}, CSim {total_caux/steps:.6f}, Rank {total_rank/steps:.6f}, RDropCLS {total_rdrop_cls/steps:.6f}, RDropREG {total_rdrop_reg/steps:.6f}) | LR: {current_lr:.6e}")

    elapsed = time.time() - start_time
    avg_loss = total_loss / max(steps, 1)
    logging.info(f"Epoch training completed in {elapsed:.2f}s with average loss {avg_loss:.6f}")
    return avg_loss, ema_state

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
        features = batch["features"].to(DEVICE, non_blocking=True)
        labels_f = batch["labels_float"].to(DEVICE, non_blocking=True)
        labels_soft = soft_targets_from_scores(labels_f, sigma=0.20)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            reg, cls_logits, pa, pt = model(input_ids, attention_mask, token_types, context_ids, features)
            fused, _ = fuse_prediction(reg, cls_logits)
            loss_mse = mse_loss(fused, labels_f)
            loss_softce = soft_ce_with_class_weights(cls_logits, labels_soft, class_weights_tensor)
        total_mse += loss_mse.detach().float().item()
        total_ce += loss_softce.detach().float().item()
        steps += 1
        all_preds.append(fused.detach().float().cpu().numpy())
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
        features = batch["features"].to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            reg, cls_logits, pa, pt = model(input_ids, attention_mask, token_types, context_ids, features)
            fused, _ = fuse_prediction(reg, cls_logits)
        all_preds.append(fused.detach().float().cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    return all_preds

# ===============
# Cross-Validation
# ===============
oof_preds = np.zeros(len(train_df), dtype=np.float32)
test_preds_folds = []

# Build test dataset/loader once
test_dataset = PatentDataset(test_ids, test_masks, test_types, test_ctx_ids, test_feats, labels_float=None, labels_cls=None, is_training=False, word_dropout_prob=0.0)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)
if USE_TTA_SWAP:
    test_dataset_sw = PatentDataset(test_ids_sw, test_masks_sw, test_types_sw, test_ctx_ids, test_feats, labels_float=None, labels_cls=None, is_training=False, word_dropout_prob=0.0)
    test_loader_sw = DataLoader(
        test_dataset_sw,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

for fold_id in range(N_FOLDS):
    logging.info(f"========== Fold {fold_id + 1}/{N_FOLDS} ==========")
    val_idx = fold_indices[fold_id]
    train_idx = np.array(sorted(list(set(range(len(train_df))) - set(val_idx.tolist()))), dtype=np.int64)

    # Base train split
    x_train_ids = train_ids[train_idx]
    x_train_masks = train_masks[train_idx]
    x_train_types = train_types[train_idx]
    x_train_ctx = train_ctx_ids[train_idx]
    x_train_feats = train_feats[train_idx]
    y_train_f = labels_float[train_idx]
    y_train_c = labels_cls[train_idx]

    # Augment with swapped split
    if USE_SWAP_AUG:
        x_train_ids = np.concatenate([x_train_ids, train_ids_sw[train_idx]], axis=0)
        x_train_masks = np.concatenate([x_train_masks, train_masks_sw[train_idx]], axis=0)
        x_train_types = np.concatenate([x_train_types, train_types_sw[train_idx]], axis=0)
        x_train_ctx = np.concatenate([x_train_ctx, train_ctx_ids[train_idx]], axis=0)
        x_train_feats = np.concatenate([x_train_feats, train_feats[train_idx]], axis=0)
        y_train_f = np.concatenate([y_train_f, labels_float[train_idx]], axis=0)
        y_train_c = np.concatenate([y_train_c, labels_cls[train_idx]], axis=0)
        logging.info(f"Applied swap augmentation. New train size: {len(y_train_f)}")

    # Validation split
    x_val_ids = train_ids[val_idx]
    x_val_masks = train_masks[val_idx]
    x_val_types = train_types[val_idx]
    x_val_ctx = train_ctx_ids[val_idx]
    x_val_feats = train_feats[val_idx]
    y_val_f = labels_float[val_idx]
    y_val_c = labels_cls[val_idx]

    # Prepare swapped validation for TTA if enabled
    if USE_TTA_SWAP:
        x_val_ids_sw = train_ids_sw[val_idx]
        x_val_masks_sw = train_masks_sw[val_idx]
        x_val_types_sw = train_types_sw[val_idx]
        x_val_ctx_sw = train_ctx_ids[val_idx]  # context same
        x_val_feats_sw = x_val_feats           # symmetric handcrafted features
        logging.info("Prepared swapped TTA arrays for validation data.")

    logging.info(f"Train split: {len(y_train_f)} samples; Val split: {len(y_val_f)} samples.")
    train_dataset = PatentDataset(x_train_ids, x_train_masks, x_train_types, x_train_ctx, x_train_feats, y_train_f, y_train_c, is_training=True, word_dropout_prob=WORD_DROPOUT_PROB)
    val_dataset = PatentDataset(x_val_ids, x_val_masks, x_val_types, x_val_ctx, x_val_feats, y_val_f, y_val_c, is_training=False, word_dropout_prob=0.0)
    if USE_TTA_SWAP:
        val_dataset_sw = PatentDataset(x_val_ids_sw, x_val_masks_sw, x_val_types_sw, x_val_ctx_sw, x_val_feats_sw, y_val_f, y_val_c, is_training=False, word_dropout_prob=0.0)

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
    if USE_TTA_SWAP:
        val_loader_sw = DataLoader(
            val_dataset_sw,
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
        head_hidden=HEAD_HIDDEN,
        feat_dim=FEAT_DIM,
        feat_embed_dim=FEAT_EMBED_DIM,
        ms_dropouts=MS_DROPOUTS
    )
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    total_steps = EPOCHS * max(1, math.ceil(len(train_loader)))
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda")

    ema_state = init_ema_state(model)
    best_val_pearson = -1.0
    best_ema_state = None

    for epoch in range(1, EPOCHS + 1):
        logging.info(f"Fold {fold_id} - Epoch {epoch}/{EPOCHS} started.")
        train_loss, ema_state = train_one_epoch(model, train_loader, optimizer, scaler, scheduler, ema_state)
        # Evaluate with EMA state
        val_mse, val_ce, val_pearson, val_pred, val_true = evaluate_with_state(model, val_loader, ema_state)
        logging.info(f"Fold {fold_id} - Epoch {epoch} - Train Loss: {train_loss:.6f} | EMA Val MSE: {val_mse:.6f} | EMA Val SoftCE: {val_ce:.6f} | EMA Val Pearson: {val_pearson:.6f}")
        if val_pearson > best_val_pearson:
            best_val_pearson = val_pearson
            best_ema_state = {k: v.clone() for k, v in ema_state.items()}
            logging.info(f"Fold {fold_id} - Epoch {epoch} - New best EMA model with Pearson {best_val_pearson:.6f}.")

    # Load best EMA state and generate predictions
    model.load_state_dict(best_ema_state)
    model.to(DEVICE)
    val_mse_final, val_ce_final, val_pearson_final, val_pred_final, val_true_final = evaluate(model, val_loader)
    logging.info(f"Fold {fold_id} - Final Val Pearson (no TTA): {val_pearson_final:.6f} | Final Val MSE: {val_mse_final:.6f} | Final Val SoftCE: {val_ce_final:.6f}")

    # If TTA swap, compute swapped predictions and average for OOF storage
    if USE_TTA_SWAP:
        _, _, _, val_pred_sw, _ = evaluate(model, val_loader_sw)
        val_pred_avg = 0.5 * (val_pred_final + val_pred_sw)
        pearson_tta = pearsonr_np(val_true_final, val_pred_avg)
        logging.info(f"Fold {fold_id} - Final Val Pearson (with TTA swap): {pearson_tta:.6f}")
        oof_preds[val_idx] = val_pred_avg.astype(np.float32)
    else:
        oof_preds[val_idx] = val_pred_final.astype(np.float32)

    # Predict on test with TTA swap if enabled
    fold_test_preds = predict(model, test_loader).astype(np.float32)
    if USE_TTA_SWAP:
        fold_test_preds_sw = predict(model, test_loader_sw).astype(np.float32)
        fold_test_preds = 0.5 * (fold_test_preds + fold_test_preds_sw)
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
logging.info(f"OOF Pearson correlation across all folds (pre-rule, pre-calibration): {oof_pearson:.6f}")
logging.info(f"OOF MSE across all folds (pre-rule, pre-calibration): {oof_mse:.6f}")

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
logging.info(f"OOF Pearson after rule (pre-calibration): {oof_pearson_rule:.6f}")
logging.info(f"OOF MSE after rule (pre-calibration): {oof_mse_rule:.6f}")

# ==========================
# Isotonic Calibration (PAV) on OOF
# ==========================
def isotonic_fit(x: np.ndarray, y: np.ndarray):
    logging.info("Fitting isotonic regression (PAV) calibrator on OOF predictions.")
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    sums = []
    weights = []
    xmins = []
    xmaxs = []
    for i in range(len(x_sorted)):
        sums.append(float(y_sorted[i]))
        weights.append(1.0)
        xmins.append(float(x_sorted[i]))
        xmaxs.append(float(x_sorted[i]))
        while len(sums) >= 2:
            avg_prev = sums[-2] / weights[-2]
            avg_last = sums[-1] / weights[-1]
            if avg_prev <= avg_last:
                break
            sums[-2] = sums[-2] + sums[-1]
            weights[-2] = weights[-2] + weights[-1]
            xmins[-2] = min(xmins[-2], xmins[-1])
            xmaxs[-2] = max(xmaxs[-2], xmaxs[-1])
            sums.pop(-1)
            weights.pop(-1)
            xmins.pop(-1)
            xmaxs.pop(-1)
    b_avg = np.array([sums[i] / weights[i] for i in range(len(sums))], dtype=np.float32)
    b_xmin = np.array(xmins, dtype=np.float32)
    b_xmax = np.array(xmaxs, dtype=np.float32)
    logging.info(f"Isotonic produced {len(b_avg)} monotonic blocks.")
    return b_xmin, b_xmax, b_avg

def isotonic_predict(x_new: np.ndarray, b_xmin: np.ndarray, b_xmax: np.ndarray, b_avg: np.ndarray):
    idx = np.searchsorted(b_xmax, x_new, side="right")
    idx = np.clip(idx, 0, len(b_avg) - 1)
    return b_avg[idx]

# Fit calibrator on OOF after rule
b_xmin, b_xmax, b_avg = isotonic_fit(oof_preds.copy(), labels_float.copy())
# Apply calibrator to OOF and Test
oof_preds_cal = isotonic_predict(oof_preds.copy(), b_xmin, b_xmax, b_avg).astype(np.float32)
test_preds_cal = isotonic_predict(test_preds_mean.copy(), b_xmin, b_xmax, b_avg).astype(np.float32)
# Clip to [0,1]
oof_preds_cal = np.clip(oof_preds_cal, 0.0, 1.0)
test_preds_cal = np.clip(test_preds_cal, 0.0, 1.0)

# Metrics after calibration
oof_pearson_cal = pearsonr_np(labels_float, oof_preds_cal)
oof_mse_cal = float(np.mean((labels_float - oof_preds_cal) ** 2))
logging.info(f"OOF Pearson after isotonic calibration: {oof_pearson_cal:.6f}")
logging.info(f"OOF MSE after isotonic calibration: {oof_mse_cal:.6f}")
logging.info("Logging final validation results (OOF, post-rule, post-calibration):")
logging.info(f"Final Validation Pearson: {oof_pearson_cal:.6f} | Final Validation MSE: {oof_mse_cal:.6f}")

# ===============
# Build Submission
# ===============
submission = pd.DataFrame({
    "id": test_df["id"].values,
    "score": test_preds_cal
})
submission["score"] = submission["score"].clip(0.0, 1.0)

logging.info(f"Writing submission to {SUBMISSION_PATH}")
submission.to_csv(SUBMISSION_PATH, index=False)
logging.info("Submission file created successfully.")
logging.info("All done (v8).")