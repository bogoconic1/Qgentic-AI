import os
import re
import time
import json
import random
import hashlib
import logging
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from sklearn.model_selection import StratifiedKFold

# =========================
# Configuration and Logging
# =========================

BASE_DIR = "task/learning-agency-lab-automated-essay-scoring-2"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "6")
LOG_FILE = os.path.join(OUTPUT_DIR, "code_6_v3.txt")
SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ],
)
logging.info("Starting AES 2.0 training script (v3) with CUDA and fp16 mixed precision (AMP).")

# ===========
# Seeding
# ===========

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
logging.info(f"Random seeds set to {SEED}.")

# ============================
# Device (Use CUDA everywhere)
# ============================

device = torch.device("cuda")
logging.info(f"Using device: {device}")

# ============================
# Data Paths
# ============================

TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TEST_CSV = os.path.join(BASE_DIR, "test.csv")
logging.info(f"Data paths set: train={TRAIN_CSV}, test={TEST_CSV}")

# ============================
# Utility: Quadratic Weighted Kappa
# ============================

def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray, min_rating: int = 1, max_rating: int = 6) -> float:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    N = max_rating - min_rating + 1
    O = np.zeros((N, N), dtype=np.float64)
    for a, p in zip(y_true, y_pred):
        O[a - min_rating, p - min_rating] += 1.0
    act_hist = np.zeros(N, dtype=np.float64)
    pred_hist = np.zeros(N, dtype=np.float64)
    for a in y_true:
        act_hist[a - min_rating] += 1.0
    for p in y_pred:
        pred_hist[p - min_rating] += 1.0
    E = np.outer(act_hist, pred_hist)
    E = E / E.sum() * O.sum()
    W = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            W[i, j] = ((i - j) ** 2) / ((N - 1) ** 2)
    num = (W * O).sum()
    den = (W * E).sum()
    kappa = 1.0 - num / den
    return float(kappa)

# ============================
# Tokenization and Feature Engineering
# ============================

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)

STOPWORDS = set("""
a about above after again against all am an and any are as at be because been before being below between both
but by could did do does doing down during each few for from further had has have having he he'd he'll he's her here here's
hers herself him himself his how how's i i'd i'll i'm i've if in into is it it's its itself let's me more most my myself
nor of on once only or other ought our ours ourselves out over own same she she'd she'll she's should so some such than
that that's the their theirs them themselves then there there's these they they'd they'll they're they've this those through to
too under until up very was we we'd we'll we're we've were what what's when when's where where's which while who who's whom
why why's with would you you'd you'll you're you've your yours yourself yourselves
""".split())

TRANSITION_WORDS = set("""
additionally also although alternatively and another as a result because besides but consequently conversely finally first
further furthermore hence however in addition in conclusion indeed instead lastly moreover nevertheless next nonetheless
otherwise overall similarly since then therefore thus whereas while yet
""".split())

def tokenize(text: str, lower: bool = True) -> List[str]:
    if lower:
        text = text.lower()
    tokens = TOKEN_PATTERN.findall(text)
    return tokens

def stable_hash_token(token: str, vocab_size: int) -> int:
    h = hashlib.md5(token.encode("utf-8")).hexdigest()
    return (int(h, 16) % (vocab_size - 2)) + 2  # reserve 0 for pad, 1 for [CLS]

def count_in_list(tokens: List[str], words: set) -> int:
    return sum(1 for t in tokens if t in words)

def text_features(text: str, tokens: List[str]) -> Dict[str, float]:
    char_count = len(text)
    word_tokens = [t for t in tokens if re.match(r"^\w+$", t)]
    word_count = len(word_tokens)
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    sentence_count = len(sentences) if len(sentences) > 0 else 1
    avg_sentence_len = (word_count / sentence_count) if sentence_count > 0 else float(word_count)
    paragraph_count = text.count("\n\n") + 1

    # Punctuation counts
    comma_count = text.count(",")
    period_count = text.count(".")
    question_count = text.count("?")
    exclam_count = text.count("!")
    quote_count = text.count('"') + text.count("'")
    paren_count = text.count("(") + text.count(")")
    colon_count = text.count(":")
    semicolon_count = text.count(";")
    dash_count = text.count("-")

    # Vocabulary richness and lengths
    unique_words = len(set(word_tokens))
    ttr = (unique_words / max(1, word_count))
    avg_word_len = float(np.mean([len(w) for w in word_tokens])) if word_tokens else 0.0
    long_word_ratio = float(np.mean([len(w) >= 7 for w in word_tokens])) if word_tokens else 0.0

    # Uppercase ratio
    letters = [c for c in text if c.isalpha()]
    uppercase_ratio = (sum(1 for c in letters if c.isupper()) / max(1, len(letters)))

    # Stopwords and content words
    stopword_count = count_in_list(word_tokens, STOPWORDS)
    stopword_ratio = stopword_count / max(1, word_count)
    content_ratio = 1.0 - stopword_ratio

    # Transition words
    transition_count = count_in_list(word_tokens, TRANSITION_WORDS)
    transitions_per_100w = 100.0 * transition_count / max(1, word_count)

    # Ratios per word / per paragraph
    punctuation_diversity = len(set([p for p in re.findall(r"[^\w\s]", text)]))
    punct_total = comma_count + period_count + question_count + exclam_count + colon_count + semicolon_count + dash_count
    punct_per_word = punct_total / max(1, word_count)
    words_per_paragraph = word_count / max(1, paragraph_count)
    sentences_per_paragraph = sentence_count / max(1, paragraph_count)

    # Digits and unique sentences
    digit_ratio = sum(c.isdigit() for c in text) / max(1, len(text))
    unique_sentence_ratio = (len(set(sentences)) / max(1, sentence_count))

    return {
        "char_count": float(char_count),
        "word_count": float(word_count),
        "sentence_count": float(sentence_count),
        "avg_sentence_len": float(avg_sentence_len),
        "paragraph_count": float(paragraph_count),

        "comma_count": float(comma_count),
        "period_count": float(period_count),
        "question_count": float(question_count),
        "exclam_count": float(exclam_count),
        "quote_count": float(quote_count),
        "paren_count": float(paren_count),
        "colon_count": float(colon_count),
        "semicolon_count": float(semicolon_count),
        "dash_count": float(dash_count),

        "ttr": float(ttr),
        "avg_word_len": float(avg_word_len),
        "long_word_ratio": float(long_word_ratio),
        "uppercase_ratio": float(uppercase_ratio),
        "stopword_ratio": float(stopword_ratio),
        "content_ratio": float(content_ratio),
        "transition_count": float(transition_count),
        "transitions_per_100w": float(transitions_per_100w),

        "punctuation_diversity": float(punctuation_diversity),
        "punct_per_word": float(punct_per_word),
        "words_per_paragraph": float(words_per_paragraph),
        "sentences_per_paragraph": float(sentences_per_paragraph),
        "digit_ratio": float(digit_ratio),
        "unique_sentence_ratio": float(unique_sentence_ratio),
    }

# ============================
# Dataset and Collator
# ============================

@dataclass
class Sample:
    input_ids: List[int]
    features: np.ndarray
    label: int = -1  # 0-5 for train, -1 for test

class EssayDataset(Dataset):
    def __init__(self, samples: List[Sample]):
        self.samples = samples
        logging.info(f"EssayDataset initialized with {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]

class Collator:
    def __init__(self, max_len: int, num_features: int, cls_id: int):
        self.max_len = max_len
        self.num_features = num_features
        self.cls_id = cls_id
        logging.info(f"Collator created with max_len={self.max_len}, num_features={self.num_features}, cls_id={self.cls_id}")

    def __call__(self, batch: List[Sample]) -> Dict[str, torch.Tensor]:
        bsz = len(batch)
        ids = torch.zeros((bsz, self.max_len), dtype=torch.long)
        feats = torch.zeros((bsz, self.num_features), dtype=torch.float32)
        labels = torch.full((bsz,), -1, dtype=torch.long)
        for i, s in enumerate(batch):
            seq = s.input_ids
            # prepend [CLS]
            seq = [self.cls_id] + seq
            if len(seq) >= self.max_len:
                seq = seq[: self.max_len]
            else:
                seq = seq + [0] * (self.max_len - len(seq))
            ids[i] = torch.tensor(seq, dtype=torch.long)
            feats[i] = torch.tensor(s.features, dtype=torch.float32)
            labels[i] = int(s.label)
        return {
            "input_ids": ids,
            "features": feats,
            "labels": labels,
        }

# ============================
# Ordinal Regression (CORN) Helpers
# ============================

def build_corn_targets(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    # labels: (B,) in 0..K-1; return targets: (B, K-1) where t[k]=1 if y > k else 0
    B = labels.size(0)
    K = num_classes
    ks = torch.arange(K - 1, device=labels.device).unsqueeze(0).expand(B, -1)
    y = labels.unsqueeze(1).expand(-1, K - 1)
    targets = (y > ks).float()
    return targets  # (B, K-1)

def corn_predict_from_logits(logits: torch.Tensor) -> torch.Tensor:
    # logits: (B, K-1); return predicted class index in 0..K-1
    probs = torch.sigmoid(logits)
    preds = torch.sum(probs > 0.5, dim=1)
    return preds.long()

# ============================
# Model
# ============================

class TransfOrdinalClassifier(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, max_len: int, num_features: int, num_classes: int,
                 nheads: int = 8, nlayers: int = 2, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=nheads, dim_feedforward=emb_dim * ff_mult,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.attn_vec = nn.Linear(emb_dim, 1)
        self.text_proj = nn.Sequential(
            nn.Linear(emb_dim * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.feat_proj = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.fuse = nn.Sequential(
            nn.Linear(256 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.out = nn.Linear(256, num_classes - 1)  # CORN K-1 logits

    def forward(self, input_ids: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        # input_ids: (B, L)
        mask_pad = (input_ids == 0)  # True where pad
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = self.embedding(input_ids) + self.pos_embedding(pos)  # (B, L, E)
        x = self.encoder(x, src_key_padding_mask=mask_pad)  # (B, L, E)

        # Masked mean pooling
        mask = (~mask_pad).float().unsqueeze(-1)  # (B, L, 1)
        sum_x = (x * mask).sum(dim=1)  # (B, E)
        len_mask = mask.sum(dim=1).clamp(min=1.0)
        mean_pool = sum_x / len_mask

        # Masked max pooling
        x_masked = x.masked_fill(mask_pad.unsqueeze(-1), float("-inf"))
        max_pool = torch.amax(x_masked, dim=1)  # (B, E)

        # Attention pooling
        scores = self.attn_vec(torch.tanh(x)).squeeze(-1)  # (B, L)
        scores = scores.masked_fill(mask_pad, float("-inf"))
        attn = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, L, 1)
        attn_pool = torch.sum(x * attn, dim=1)  # (B, E)

        text_repr = torch.cat([mean_pool, max_pool, attn_pool], dim=1)  # (B, 3E)
        text_repr = self.text_proj(text_repr)  # (B, 256)
        feat_repr = self.feat_proj(features)   # (B, 64)
        fused = self.fuse(torch.cat([text_repr, feat_repr], dim=1))  # (B, 256)
        logits = self.out(fused)  # (B, K-1)
        return logits

# ============================
# Training Helpers
# ============================

def compute_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    total = counts.sum()
    weights = total / (num_classes * counts)
    logging.info(f"Class counts: {counts.tolist()}")
    logging.info(f"Computed class weights: {weights.tolist()}")
    return torch.tensor(weights, dtype=torch.float32, device=device)

def corn_loss(logits: torch.Tensor, labels: torch.Tensor, class_weights: torch.Tensor, label_smoothing: float = 0.05) -> torch.Tensor:
    # logits: (B, K-1), labels: (B,), class_weights: (K,)
    targets = build_corn_targets(labels, num_classes=class_weights.numel())  # (B, K-1)
    if label_smoothing > 0.0:
        targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing
    bsz, kminus1 = targets.shape
    elem_weights = class_weights[labels].unsqueeze(1).expand(-1, kminus1)  # (B, K-1)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    loss_mat = loss_fn(logits, targets)
    loss = (loss_mat * elem_weights).mean()
    return loss

def train_one_epoch(model, loader, optimizer, scaler, class_weights, label_smoothing):
    model.train()
    total_loss = 0.0
    total_steps = 0
    start = time.time()
    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        features = batch["features"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(input_ids, features)
            loss = corn_loss(logits, labels, class_weights, label_smoothing=label_smoothing)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.item())
        total_steps += 1
        if (step + 1) % 50 == 0:
            logging.info(f"Train step {step+1}/{len(loader)} | Loss: {total_loss/total_steps:.4f}")
    epoch_loss = total_loss / max(1, total_steps)
    elapsed = time.time() - start
    logging.info(f"Epoch train loss: {epoch_loss:.4f} | Time: {elapsed:.2f}s")
    return epoch_loss

@torch.no_grad()
def validate(model, loader, num_classes: int):
    model.eval()
    all_preds = []
    all_labels = []
    start = time.time()
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        features = batch["features"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(input_ids, features)
        preds = corn_predict_from_logits(logits)  # 0..K-1
        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    kappa = quadratic_weighted_kappa(all_labels + 1, all_preds + 1, 1, num_classes)
    elapsed = time.time() - start
    logging.info(f"Validation QWK: {kappa:.6f} | Time: {elapsed:.2f}s")
    return kappa, all_preds, all_labels

@torch.no_grad()
def predict_logits(model, loader):
    model.eval()
    all_logits = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        features = batch["features"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(input_ids, features)
        all_logits.append(logits.detach().float().cpu().numpy())
    return np.concatenate(all_logits, axis=0)

# ============================
# Main
# ============================

def main():
    start_total = time.time()
    # Hyperparameters
    vocab_size = 100000 + 2  # account for [PAD]=0 and [CLS]=1 reservations done in hashing
    emb_dim = 256
    max_len = 512
    num_classes = 6
    n_splits = 5
    batch_size = 16
    epochs = 3
    lr = 1.0e-3
    weight_decay = 0.01
    nheads = 8
    nlayers = 2
    label_smoothing = 0.05
    cls_id = 1

    logging.info("Reading CSVs...")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    logging.info(f"Train shape: {train_df.shape} | Test shape: {test_df.shape}")

    # Basic EDA logs
    score_counts = train_df["score"].value_counts().sort_index()
    logging.info(f"Score distribution (train): {json.dumps(score_counts.to_dict(), indent=2)}")

    # Preprocess: tokenize and features
    logging.info("Tokenizing and extracting features for train...")
    train_tokens = []
    train_feats = []
    train_texts = train_df["full_text"].tolist()
    for i, text in enumerate(train_texts):
        toks = tokenize(text, lower=True)
        feats = text_features(text, toks)
        train_tokens.append(toks)
        train_feats.append(feats)
        if (i + 1) % 2000 == 0:
            logging.info(f"Processed {i+1}/{len(train_df)} train rows")

    logging.info("Tokenizing and extracting features for test...")
    test_tokens = []
    test_feats = []
    test_texts = test_df["full_text"].tolist()
    for i, text in enumerate(test_texts):
        toks = tokenize(text, lower=True)
        feats = text_features(text, toks)
        test_tokens.append(toks)
        test_feats.append(feats)
        if (i + 1) % 2000 == 0:
            logging.info(f"Processed {i+1}/{len(test_df)} test rows")

    # Build feature matrix and standardize
    feat_names = list(train_feats[0].keys())
    logging.info(f"Feature names ({len(feat_names)}): {feat_names}")
    Xf_train = np.array([[d[k] for k in feat_names] for d in train_feats], dtype=np.float32)
    Xf_test = np.array([[d[k] for k in feat_names] for d in test_feats], dtype=np.float32)
    feat_mean = Xf_train.mean(axis=0)
    feat_std = Xf_train.std(axis=0)
    feat_std[feat_std == 0.0] = 1.0
    Xf_train = (Xf_train - feat_mean) / feat_std
    Xf_test = (Xf_test - feat_mean) / feat_std
    logging.info(f"Feature means: {feat_mean.tolist()}")
    logging.info(f"Feature stds: {feat_std.tolist()}")

    # Hash tokens to ids
    logging.info("Hashing tokens to ids for train...")
    train_ids = []
    for i, toks in enumerate(train_tokens):
        ids = [stable_hash_token(t, vocab_size) for t in toks]
        train_ids.append(ids)
        if (i + 1) % 2000 == 0:
            logging.info(f"Hashed {i+1}/{len(train_tokens)} train token lists")

    logging.info("Hashing tokens to ids for test...")
    test_ids = []
    for i, toks in enumerate(test_tokens):
        ids = [stable_hash_token(t, vocab_size) for t in toks]
        test_ids.append(ids)
        if (i + 1) % 2000 == 0:
            logging.info(f"Hashed {i+1}/{len(test_tokens)} test token lists")

    # Labels to 0..5
    y = train_df["score"].values.astype(int) - 1

    # Pre-build test dataset/loader
    test_samples = [Sample(input_ids=test_ids[i], features=Xf_test[i], label=-1) for i in range(len(test_df))]
    collator = Collator(max_len=max_len, num_features=Xf_train.shape[1], cls_id=cls_id)
    test_loader = DataLoader(
        EssayDataset(test_samples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=False,
        collate_fn=collator
    )

    # Stratified 5-fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(train_df), dtype=int)
    fold_kappas = []
    test_logits_accum = []

    for fold, (trn_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        logging.info(f"========== Fold {fold}/{n_splits} ==========")
        # Build datasets
        train_samples = [Sample(input_ids=train_ids[i], features=Xf_train[i], label=int(y[i])) for i in trn_idx]
        val_samples = [Sample(input_ids=train_ids[i], features=Xf_train[i], label=int(y[i])) for i in val_idx]

        train_loader = DataLoader(
            EssayDataset(train_samples),
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            persistent_workers=False,
            collate_fn=collator
        )
        val_loader = DataLoader(
            EssayDataset(val_samples),
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=False,
            collate_fn=collator
        )

        # Model
        model = TransfOrdinalClassifier(
            vocab_size=vocab_size, emb_dim=emb_dim, max_len=max_len,
            num_features=Xf_train.shape[1], num_classes=num_classes,
            nheads=nheads, nlayers=nlayers, ff_mult=4, dropout=0.1
        )
        model.to(device)
        logging.info(f"Model initialized for fold {fold} with {sum(p.numel() for p in model.parameters()):,} parameters.")
        logging.info(f"Model first parameter dtype (should be float32 for AMP): {next(model.parameters()).dtype}")

        # Optimizer, Loss weights, Scaler
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        class_weights = compute_class_weights(y[trn_idx], num_classes)
        scaler = torch.amp.GradScaler("cuda", enabled=True)

        best_kappa = -1.0
        best_state = None

        for epoch in range(1, epochs + 1):
            logging.info(f"Fold {fold} | Epoch {epoch}/{epochs} started.")
            train_loss = train_one_epoch(model, train_loader, optimizer, scaler, class_weights, label_smoothing)
            val_kappa, val_preds, val_labels = validate(model, val_loader, num_classes)
            if val_kappa > best_kappa:
                best_kappa = val_kappa
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            logging.info(f"Fold {fold} | Epoch {epoch} done. TrainLoss={train_loss:.4f}, ValQWK={val_kappa:.6f}, BestQWK={best_kappa:.6f}")

        # Load best state
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        final_kappa, val_preds, _ = validate(model, val_loader, num_classes)
        fold_kappas.append(final_kappa)
        oof_preds[val_idx] = val_preds

        # Test logits (CORN K-1 logits)
        logits = predict_logits(model, test_loader)
        test_logits_accum.append(logits)

        # Cleanup
        del model
        torch.cuda.empty_cache()
        logging.info(f"Fold {fold} finished. Best Validation QWK: {best_kappa:.6f}")

    # Overall OOF Kappa
    overall_kappa = quadratic_weighted_kappa(y + 1, oof_preds + 1, 1, num_classes)
    logging.info(f"OOF Quadratic Weighted Kappa across folds: {overall_kappa:.6f}")
    logging.info(f"Per-fold QWK: {[float(k) for k in fold_kappas]}")

    # Test prediction: average logits across folds, then CORN decode
    avg_test_logits = np.mean(np.stack(test_logits_accum, axis=0), axis=0)  # (Ntest, K-1)
    avg_test_logits_t = torch.from_numpy(avg_test_logits)
    test_pred_classes = corn_predict_from_logits(avg_test_logits_t).numpy() + 1  # 1..6

    # Write submission
    submission = pd.DataFrame({
        "essay_id": test_df["essay_id"].values,
        "score": test_pred_classes.astype(int)
    })
    submission.to_csv(SUBMISSION_PATH, index=False)
    logging.info(f"Submission saved to {SUBMISSION_PATH} with shape {submission.shape}.")

    elapsed_total = time.time() - start_total
    logging.info(f"All done in {elapsed_total/60:.2f} minutes. Final OOF QWK: {overall_kappa:.6f}")

if __name__ == "__main__":
    main()