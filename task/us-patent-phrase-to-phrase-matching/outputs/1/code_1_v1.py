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
LOG_PATH = os.path.join(OUTPUT_DIR, "code_1_v1.txt")
SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission_1.csv")

SEED = 42
N_FOLDS = 5
EPOCHS = 5
BATCH_SIZE = 256
MAX_LEN = 64
EMBED_DIM = 128
N_HEAD = 4
N_LAYERS = 2
FFN_DIM = 512
DROPOUT = 0.1
LR = 2e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

# Always use CUDA and FP16 as per constraints
DEVICE = torch.device("cuda")

os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    filename=LOG_PATH,
    filemode="w",
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logging.info("Initialized logging and created output directories.")

# ==============
# Reproducibility
# ==============
def set_seed(seed: int):
    logging.info(f"Setting all random seeds to {seed}.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Enable cudnn benchmark for performance as we always use CUDA
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
    # Lowercase and basic punctuation spacing
    s = s.lower()
    s = re.sub(r"[^0-9a-zA-Z\-\+_/\.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(s: str):
    s = normalize_text(s)
    if not s:
        return []
    return s.split(" ")

def build_text_row(context: str, anchor: str, target: str) -> str:
    # Concatenate context, anchor, target with special tokens
    return f"[CTX] {context} [SEP] {anchor} [SEP] {target} [EOS]"

def build_vocab(corpus_texts):
    logging.info("Building vocabulary from corpus.")
    vocab = dict()
    inv_vocab = []
    # Add special tokens first
    for tok in SPECIAL_TOKENS:
        vocab[tok] = len(vocab)
        inv_vocab.append(tok)

    # Collect frequencies to decide ordering (optional)
    freq = {}
    for t in corpus_texts:
        toks = tokenize(t)
        for tok in toks:
            freq[tok] = freq.get(tok, 0) + 1

    # Sort tokens by frequency (desc) then lexicographically to stabilize
    sorted_tokens = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    for tok, _ in sorted_tokens:
        if tok not in vocab:
            vocab[tok] = len(vocab)
            inv_vocab.append(tok)

    logging.info(f"Built vocabulary of size {len(vocab)}.")
    return vocab, inv_vocab

def numericalize(text: str, vocab: dict, max_len: int):
    toks = []
    # Add CLS at the start
    toks.append("[CLS]")
    toks.extend(tokenize(text))
    # Truncate to leave room for PAD if needed
    toks = toks[:max_len]
    # Map to ids
    ids = [vocab.get(tok, UNK_IDX) for tok in toks]
    # Pad
    if len(ids) < max_len:
        ids = ids + [PAD_IDX] * (max_len - len(ids))
    attn_mask = [1 if idx != PAD_IDX else 0 for idx in ids]
    return np.array(ids, dtype=np.int64), np.array(attn_mask, dtype=np.int64)

logging.info("Preparing combined corpus for vocabulary.")
full_corpus = []
for _, row in train_df.iterrows():
    full_corpus.append(build_text_row(row["context"], row["anchor"], row["target"]))
for _, row in test_df.iterrows():
    full_corpus.append(build_text_row(row["context"], row["anchor"], row["target"]))

vocab, inv_vocab = build_vocab(full_corpus)

# Precompute tokenized arrays for efficiency
def preprocess_dataframe(df: pd.DataFrame, vocab: dict, max_len: int):
    logging.info(f"Tokenizing and numericalizing dataframe with {len(df)} rows.")
    input_ids = np.zeros((len(df), max_len), dtype=np.int64)
    attention_masks = np.zeros((len(df), max_len), dtype=np.int64)
    for i, row in df.iterrows():
        text = build_text_row(row["context"], row["anchor"], row["target"])
        ids, mask = numericalize(text, vocab, max_len)
        input_ids[i] = ids
        attention_masks[i] = mask
        if (i + 1) % 5000 == 0:
            logging.info(f"Processed {i + 1} rows.")
    return input_ids, attention_masks

train_ids, train_masks = preprocess_dataframe(train_df, vocab, MAX_LEN)
test_ids, test_masks = preprocess_dataframe(test_df, vocab, MAX_LEN)

labels = train_df["score"].values.astype(np.float32)
logging.info(f"Labels statistics: min={labels.min()}, max={labels.max()}, mean={labels.mean():.4f}, std={labels.std():.4f}")

# ================
# Stratified K-Fold
# ================
def make_stratified_folds(y: np.ndarray, n_folds: int, seed: int):
    logging.info("Creating stratified folds.")
    # Convert labels to bins (0,1,2,3,4) based on 0.25 increments
    bins = (y * 4.0 + 1e-6).astype(int)  # avoid float issues
    # Group indices by bin
    bin_to_indices = {}
    for idx, b in enumerate(bins):
        bin_to_indices.setdefault(b, []).append(idx)
    # Shuffle within each bin
    rng = np.random.RandomState(seed)
    for b in bin_to_indices:
        rng.shuffle(bin_to_indices[b])
    # Initialize fold lists
    folds = [[] for _ in range(n_folds)]
    # Distribute indices round-robin across folds per bin
    for b, idxs in bin_to_indices.items():
        for i, idx in enumerate(idxs):
            fold_id = i % n_folds
            folds[fold_id].append(idx)
    # Convert to arrays
    fold_arrays = [np.array(sorted(f), dtype=np.int64) for f in folds]
    for i, f in enumerate(fold_arrays):
        logging.info(f"Fold {i}: {len(f)} samples.")
    return fold_arrays

fold_indices = make_stratified_folds(labels, N_FOLDS, SEED)

# ==============
# PyTorch Dataset
# ==============
class PatentDataset(Dataset):
    def __init__(self, input_ids: np.ndarray, attention_masks: np.ndarray, labels: np.ndarray = None):
        logging.info(f"Initializing PatentDataset with {len(input_ids)} samples.")
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.from_numpy(self.input_ids[idx]),
            "attention_mask": torch.from_numpy(self.attention_masks[idx]),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item

# =======================
# Transformer Regressor
# =======================
class TransformerRegressor(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, n_head: int, num_layers: int, ffn_dim: int, max_len: int, dropout: float):
        super().__init__()
        logging.info(f"Initializing TransformerRegressor with vocab_size={vocab_size}, embed_dim={embed_dim}, heads={n_head}, layers={num_layers}, max_len={max_len}")
        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.pos_emb = nn.Embedding(max_len, embed_dim)
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
        self.head = nn.Linear(embed_dim, 1)
        # Initialize weights
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

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        # input_ids: (B, L), attention_mask: (B, L) with 1 for tokens, 0 for padding
        B, L = input_ids.shape
        positions = torch.arange(0, L, device=input_ids.device, dtype=torch.long).unsqueeze(0).expand(B, L)
        tok = self.token_emb(input_ids)  # (B, L, D)
        pos = self.pos_emb(positions)    # (B, L, D)
        x = tok + pos
        x = self.layer_norm(x)
        x = self.dropout(x)
        # Transformer expects (S, N, E)
        x = x.transpose(0, 1)  # (L, B, D)
        # src_key_padding_mask expects True for pads
        key_padding_mask = attention_mask == 0  # (B, L)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = x.transpose(0, 1)  # (B, L, D)
        # Masked mean pooling
        mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
        summed = torch.sum(x * mask, dim=1)         # (B, D)
        counts = torch.clamp(mask.sum(dim=1), min=1.0)  # (B, 1)
        pooled = summed / counts
        pooled = self.dropout(pooled)
        logits = self.head(pooled).squeeze(-1)  # (B,)
        preds = torch.sigmoid(logits)           # constrain to [0,1]
        return preds

# ==============
# Loss and Metric
# ==============
def mse_loss(preds: torch.Tensor, targets: torch.Tensor):
    return torch.mean((preds - targets) ** 2)

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
# Training and Eval
# ===================
def train_one_epoch(model, loader, optimizer, scaler, scheduler=None):
    model.train()
    total_loss = 0.0
    steps = 0
    start_time = time.time()
    for batch_idx, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        labels = batch["labels"].to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            preds = model(input_ids, attention_mask)
            loss = mse_loss(preds, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.detach().float().item()
        steps += 1
        if (batch_idx + 1) % 50 == 0:
            logging.info(f"Train step {batch_idx + 1}/{len(loader)} - Loss: {total_loss / steps:.6f}")
    elapsed = time.time() - start_time
    avg_loss = total_loss / max(steps, 1)
    logging.info(f"Epoch training completed in {elapsed:.2f}s with average loss {avg_loss:.6f}")
    return avg_loss

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    steps = 0
    for batch_idx, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        labels = batch["labels"].to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            preds = model(input_ids, attention_mask)
            loss = mse_loss(preds, labels)
        total_loss += loss.detach().float().item()
        steps += 1
        all_preds.append(preds.detach().float().cpu().numpy())
        all_labels.append(labels.detach().float().cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    pearson = pearsonr_np(all_labels, all_preds)
    avg_loss = total_loss / max(steps, 1)
    return avg_loss, pearson, all_preds, all_labels

@torch.no_grad()
def predict(model, loader):
    model.eval()
    all_preds = []
    for batch_idx, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            preds = model(input_ids, attention_mask)
        all_preds.append(preds.detach().float().cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    return all_preds

# ===============
# Cross-Validation
# ===============
oof_preds = np.zeros(len(train_df), dtype=np.float32)
test_preds_folds = []

# Build test dataset/loader once
test_dataset = PatentDataset(test_ids, test_masks, labels=None)
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
    y_train = labels[train_idx]

    x_val_ids = train_ids[val_idx]
    x_val_masks = train_masks[val_idx]
    y_val = labels[val_idx]

    logging.info(f"Train split: {len(train_idx)} samples; Val split: {len(val_idx)} samples.")
    train_dataset = PatentDataset(x_train_ids, x_train_masks, y_train)
    val_dataset = PatentDataset(x_val_ids, x_val_masks, y_val)

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
        dropout=DROPOUT
    )
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # Cosine schedule over total steps
    total_steps = EPOCHS * max(1, math.ceil(len(train_loader)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=LR * 0.05)
    scaler = torch.cuda.amp.GradScaler()

    best_val_pearson = -1.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        logging.info(f"Fold {fold_id} - Epoch {epoch}/{EPOCHS} started.")
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, scheduler)
        val_loss, val_pearson, val_pred, val_true = evaluate(model, val_loader)
        logging.info(f"Fold {fold_id} - Epoch {epoch} - Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val Pearson: {val_pearson:.6f}")
        if val_pearson > best_val_pearson:
            best_val_pearson = val_pearson
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            logging.info(f"Fold {fold_id} - Epoch {epoch} - New best model with Pearson {best_val_pearson:.6f}.")

    # Load best state and generate predictions
    model.load_state_dict(best_state)
    _, val_pearson_final, val_pred_final, val_true_final = evaluate(model, val_loader)
    logging.info(f"Fold {fold_id} - Final Val Pearson: {val_pearson_final:.6f}")
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
oof_pearson = pearsonr_np(labels, oof_preds)
oof_mse = float(np.mean((labels - oof_preds) ** 2))
logging.info(f"OOF Pearson correlation across all folds: {oof_pearson:.6f}")
logging.info(f"OOF MSE across all folds: {oof_mse:.6f}")
logging.info("Logging final validation results (OOF):")
logging.info(f"Final Validation Pearson: {oof_pearson:.6f} | Final Validation MSE: {oof_mse:.6f}")

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
logging.info(f"Found {num_equal_oof} exact anchor==target pairs in train (OOF) set; applying same rule to OOF predictions for metric sanity.")
oof_preds[anchor_eq_target_mask_oof] = 1.0

# Recompute OOF metric after rule application
oof_pearson_rule = pearsonr_np(labels, oof_preds)
oof_mse_rule = float(np.mean((labels - oof_preds) ** 2))
logging.info(f"OOF Pearson after rule: {oof_pearson_rule:.6f}")
logging.info(f"OOF MSE after rule: {oof_mse_rule:.6f}")

# ===============
# Build Submission
# ===============
submission = pd.DataFrame({
    "id": test_df["id"].values,
    "score": test_preds_mean
})
# Ensure [0,1] range
submission["score"] = submission["score"].clip(0.0, 1.0)

logging.info(f"Writing submission to {SUBMISSION_PATH}")
submission.to_csv(SUBMISSION_PATH, index=False)
logging.info("Submission file created successfully.")
logging.info("All done.")