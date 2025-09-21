import os
import re
import math
import time
import json
import random
import hashlib
import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Sampler
from torch.utils.data import WeightedRandomSampler
from torch.nn.utils import clip_grad_norm_
from sklearn.model_selection import StratifiedKFold

# =========================
# Configuration and Logging
# =========================

BASE_DIR = "task/learning-agency-lab-automated-essay-scoring-2"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "6")
LOG_FILE = os.path.join(OUTPUT_DIR, "code_6_v1.txt")
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

logging.info("Starting AES 2.0 training script with CUDA and fp16 mixed precision.")


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
SAMPLE_SUB_CSV = os.path.join(BASE_DIR, "sample_submission.csv")
logging.info(f"Data paths set: train={TRAIN_CSV}, test={TEST_CSV}")


# ============================
# Utility: Quadratic Weighted Kappa
# ============================

def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray, min_rating: int = 0, max_rating: int = 5) -> float:
    assert y_true.shape == y_pred.shape
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

def tokenize(text: str, lower: bool = True) -> List[str]:
    if lower:
        text = text.lower()
    tokens = TOKEN_PATTERN.findall(text)
    return tokens

def stable_hash_token(token: str, vocab_size: int) -> int:
    h = hashlib.md5(token.encode("utf-8")).hexdigest()
    return (int(h, 16) % (vocab_size - 1)) + 1  # reserve 0 for padding

def text_features(text: str, tokens: List[str]) -> Dict[str, float]:
    char_count = len(text)
    word_count = sum(1 for t in tokens if re.match(r"^\w+$", t))
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    sentence_count = len(sentences) if len(sentences) > 0 else 1
    avg_sentence_len = (word_count / sentence_count) if sentence_count > 0 else float(word_count)
    paragraph_count = text.count("\n\n") + 1
    colon_count = text.count(":")
    semicolon_count = text.count(";")
    dash_count = text.count("-")
    words_only = [t for t in tokens if re.match(r"^\w+$", t)]
    unique_words = len(set(words_only))
    ttr = (unique_words / max(1, len(words_only)))
    puncts = set([p for p in re.findall(r"[^\w\s]", text)])
    punctuation_diversity = len(puncts)
    return {
        "char_count": float(char_count),
        "word_count": float(word_count),
        "sentence_count": float(sentence_count),
        "avg_sentence_len": float(avg_sentence_len),
        "paragraph_count": float(paragraph_count),
        "colon_count": float(colon_count),
        "semicolon_count": float(semicolon_count),
        "dash_count": float(dash_count),
        "ttr": float(ttr),
        "punctuation_diversity": float(punctuation_diversity),
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
    def __init__(self, max_len: int, num_features: int):
        self.max_len = max_len
        self.num_features = num_features
        logging.info(f"Collator created with max_len={self.max_len}, num_features={self.num_features}")

    def __call__(self, batch: List[Sample]) -> Dict[str, torch.Tensor]:
        bsz = len(batch)
        ids = torch.zeros((bsz, self.max_len), dtype=torch.long)
        feats = torch.zeros((bsz, self.num_features), dtype=torch.float32)
        labels = torch.full((bsz,), -1, dtype=torch.long)
        for i, s in enumerate(batch):
            seq = s.input_ids
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
# Model
# ============================

class CNNTextClassifier(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, num_features: int, num_classes: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(emb_dim, 128, kernel_size=2, padding=0),
            nn.Conv1d(emb_dim, 128, kernel_size=3, padding=0),
            nn.Conv1d(emb_dim, 128, kernel_size=4, padding=0),
        ])
        self.dropout = nn.Dropout(0.2)
        conv_out_dim = 128 * len(self.convs)
        pooled_dim = conv_out_dim + emb_dim  # cnn pooled + avg embedding
        self.text_proj = nn.Sequential(
            nn.Linear(pooled_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.feat_proj = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.head = nn.Sequential(
            nn.Linear(256 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_ids: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        # input_ids: (B, L), features: (B, F)
        emb = self.embedding(input_ids)  # (B, L, E)
        mask = (input_ids != 0).float()  # (B, L)
        # Avg pooling with mask
        sum_emb = (emb * mask.unsqueeze(-1)).sum(dim=1)  # (B, E)
        len_mask = mask.sum(dim=1).clamp(min=1.0).unsqueeze(-1)  # (B, 1)
        avg_emb = sum_emb / len_mask  # (B, E)
        # CNN paths
        x = emb.transpose(1, 2)  # (B, E, L)
        pooled_list = []
        for conv in self.convs:
            c = conv(x)  # (B, 128, L-k+1)
            c = torch.relu(c)
            p = torch.amax(c, dim=2)  # Global max pool -> (B, 128)
            pooled_list.append(p)
        cnn_pooled = torch.cat(pooled_list, dim=1)  # (B, 128*len)
        text_repr = torch.cat([avg_emb, cnn_pooled], dim=1)  # (B, pooled_dim)
        text_repr = self.text_proj(text_repr)  # (B, 256)
        feat_repr = self.feat_proj(features)  # (B, 64)
        combined = torch.cat([text_repr, feat_repr], dim=1)  # (B, 320)
        logits = self.head(combined)  # (B, C)
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

def train_one_epoch(model, loader, optimizer, scaler, criterion):
    model.train()
    total_loss = 0.0
    total_steps = 0
    start = time.time()
    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        features = batch["features"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            logits = model(input_ids, features)
            loss = criterion(logits, labels)
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
def validate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    start = time.time()
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        features = batch["features"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            logits = model(input_ids, features)
        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    kappa = quadratic_weighted_kappa(all_labels + 1, all_preds + 1, 1, 6)
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
        with torch.cuda.amp.autocast(dtype=torch.float16):
            logits = model(input_ids, features)
        all_logits.append(logits.detach().float().cpu().numpy())
    return np.concatenate(all_logits, axis=0)


# ============================
# Main
# ============================

def main():
    start_total = time.time()
    # Hyperparameters
    vocab_size = 100000
    emb_dim = 256
    max_len = 512  # validated by plan
    num_classes = 6
    n_splits = 5
    batch_size = 16
    epochs = 2
    lr = 2e-3
    weight_decay = 0.01

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
    for i, text in enumerate(train_df["full_text"].tolist()):
        toks = tokenize(text, lower=True)
        feats = text_features(text, toks)
        train_tokens.append(toks)
        train_feats.append(feats)
        if (i + 1) % 2000 == 0:
            logging.info(f"Processed {i+1}/{len(train_df)} train rows")

    logging.info("Tokenizing and extracting features for test...")
    test_tokens = []
    test_feats = []
    for i, text in enumerate(test_df["full_text"].tolist()):
        toks = tokenize(text, lower=True)
        feats = text_features(text, toks)
        test_tokens.append(toks)
        test_feats.append(feats)
        if (i + 1) % 2000 == 0:
            logging.info(f"Processed {i+1}/{len(test_df)} test rows")

    # Build feature matrix and standardize
    feat_names = list(train_feats[0].keys())
    logging.info(f"Feature names: {feat_names}")
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

    # Stratified 5-fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(train_df), dtype=int)
    fold_kappas = []
    test_logits_accum = []

    # Pre-build test dataset/loader
    test_samples = [Sample(input_ids=test_ids[i], features=Xf_test[i], label=-1) for i in range(len(test_df))]
    collator = Collator(max_len=max_len, num_features=Xf_train.shape[1])
    test_loader = DataLoader(
        EssayDataset(test_samples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collator
    )

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
            collate_fn=collator
        )
        val_loader = DataLoader(
            EssayDataset(val_samples),
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=collator
        )

        # Model
        model = CNNTextClassifier(vocab_size=vocab_size, emb_dim=emb_dim, num_features=Xf_train.shape[1], num_classes=num_classes)
        model.to(device)
        model.half()  # move parameters to fp16 as well
        logging.info(f"Model initialized for fold {fold} with {sum(p.numel() for p in model.parameters()):,} parameters.")

        # Optimizer, Loss with class weights, Scaler
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        class_weights = compute_class_weights(y[trn_idx], num_classes)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        scaler = torch.cuda.amp.GradScaler(enabled=True)

        best_kappa = -1.0
        best_state = None

        for epoch in range(1, epochs + 1):
            logging.info(f"Fold {fold} | Epoch {epoch}/{epochs} started.")
            train_loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion)
            val_kappa, val_preds, val_labels = validate(model, val_loader)
            if val_kappa > best_kappa:
                best_kappa = val_kappa
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            logging.info(f"Fold {fold} | Epoch {epoch} done. TrainLoss={train_loss:.4f}, ValQWK={val_kappa:.6f}, BestQWK={best_kappa:.6f}")

        # Load best state
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        # Final validation predictions with best model
        final_kappa, val_preds, _ = validate(model, val_loader)
        fold_kappas.append(final_kappa)
        oof_preds[val_idx] = val_preds

        # Test logits
        logits = predict_logits(model, test_loader)
        test_logits_accum.append(logits)

        # Cleanup
        del model
        torch.cuda.empty_cache()
        logging.info(f"Fold {fold} finished. Best Validation QWK: {best_kappa:.6f}")

    # Overall OOF Kappa
    overall_kappa = quadratic_weighted_kappa(y + 1, oof_preds + 1, 1, 6)
    logging.info(f"OOF Quadratic Weighted Kappa across folds: {overall_kappa:.6f}")
    logging.info(f"Per-fold QWK: {[float(k) for k in fold_kappas]}")

    # Test prediction: average logits across folds
    avg_test_logits = np.mean(np.stack(test_logits_accum, axis=0), axis=0)
    test_pred_classes = avg_test_logits.argmax(axis=1) + 1  # 1..6

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