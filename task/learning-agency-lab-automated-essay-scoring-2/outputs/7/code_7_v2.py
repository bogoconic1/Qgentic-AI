import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import math
import json
import random
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

# ------------------------------------------------------------------
# Logging MUST be configured before any other logging statements.
# ------------------------------------------------------------------
OUTPUT_DIR = "task/learning-agency-lab-automated-essay-scoring-2/outputs/7"
LOG_FILE_V2 = os.path.join(OUTPUT_DIR, "code_7_v2.txt")
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE_V2,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True,
)
logging.info("Initialized logging to %s", LOG_FILE_V2)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
@dataclass
class CFG:
    base_dir: str = "task/learning-agency-lab-automated-essay-scoring-2"
    train_file: str = "train.csv"
    test_file: str = "test.csv"
    persuade_file: str = "persuade_corpus_2.0.csv"  # aggregated from discourse segments
    outputs_dir: str = OUTPUT_DIR
    submission_path: str = os.path.join(OUTPUT_DIR, "submission_2.csv")
    save_model_path: str = os.path.join(OUTPUT_DIR, "deberta_v3_large_fold0.bin")
    seed: int = 42
    model_name: str = "microsoft/deberta-v3-large"
    max_length: int = 1024
    train_batch_size: int = 8
    valid_batch_size: int = 16
    num_labels: int = 6
    lr: float = 5e-5
    weight_decay: float = 0.01
    epochs: int = 3
    warmup_ratio: float = 0.1
    grad_accum_steps: int = 1
    early_stopping_patience: int = 2
    n_splits: int = 5
    fold_index: int = 0
    target_col: str = "score"
    text_col: str = "full_text"
    # Data controls for Persuade 2.0 aggregation
    persuade_group_id_col: str = "essay_id_comp"
    persuade_text_col: str = "discourse_text"
    persuade_target_col: str = "holistic_essay_score"
    persuade_max_samples_per_class: int = None  # set to an int to cap per-class samples from Persuade (None = use all)
    # Cleaning
    lowercase: bool = True

cfg = CFG()
logging.info("Configuration loaded: %s", json.dumps(asdict(cfg), indent=2))

# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------
def set_seed(seed: int):
    logging.info("Setting random seed: %d", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    logging.info("Seeds set and cuDNN benchmark enabled.")

def clean_text(text: str) -> str:
    # Standardized cleaning aligned with the research plan.
    import re
    text = re.sub(r'<[^>]+>', ' ', text)                  # Remove HTML tags
    text = re.sub(r'https?://\S+', ' ', text)             # Remove URLs
    text = re.sub(r'[^a-zA-Z0-9\s.,?!\'"]', ' ', text)    # Keep standard punctuation only
    text = re.sub(r'\s+', ' ', text).strip()              # Normalize whitespace
    if cfg.lowercase:
        text = text.lower()
    return text

def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray, min_rating: int = 1, max_rating: int = 6) -> float:
    logging.info("Computing Quadratic Weighted Kappa (QWK).")
    assert y_true.shape == y_pred.shape
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    N = max_rating - min_rating + 1
    w = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            w[i, j] = ((i - j) ** 2) / ((N - 1) ** 2)
    # histogram
    O = np.zeros((N, N), dtype=float)
    for a, p in zip(y_true, y_pred):
        O[a - min_rating, p - min_rating] += 1
    act_hist = np.bincount(y_true - min_rating, minlength=N).astype(float)
    pred_hist = np.bincount(y_pred - min_rating, minlength=N).astype(float)
    E = np.outer(act_hist, pred_hist)
    E = E / E.sum() * O.sum()
    numerator = (w * O).sum()
    denominator = (w * E).sum()
    kappa = 1.0 - numerator / denominator if denominator != 0 else 0.0
    logging.info("QWK computed: %.6f", kappa)
    return kappa

def compute_class_weights(labels: np.ndarray, num_labels: int) -> torch.Tensor:
    """
    Compute class weights ONLY from training split labels to avoid leakage.
    We normalize weights to have mean ~ 1.0 to keep loss scale stable.
    """
    logging.info("Computing class weights from training split only, labels shape: %s", labels.shape)
    # labels are expected in [1..num_labels]
    counts = np.bincount(labels.astype(int) - 1, minlength=num_labels).astype(float)
    logging.info("Training class counts: %s", counts.tolist())
    total = counts.sum()
    weights = total / (num_labels * counts)
    weights = np.where(np.isfinite(weights), weights, 0.0)
    weights_t = torch.tensor(weights, dtype=torch.float32)
    logging.info("Computed class weights (mean ~ %.6f): %s", float(weights_t.mean().item()), weights_t.tolist())
    return weights_t

# ------------------------------------------------------------------
# Data Loading and Merging
# ------------------------------------------------------------------
def aggregate_persuade(df_p: pd.DataFrame) -> pd.DataFrame:
    logging.info("Aggregating Persuade 2.0 by '%s' -> concatenating '%s'.", cfg.persuade_group_id_col, cfg.persuade_text_col)
    df_p = df_p[[cfg.persuade_group_id_col, cfg.persuade_text_col, cfg.persuade_target_col]].dropna(subset=[cfg.persuade_target_col])
    logging.info("Rows with non-null targets in Persuade: %d", len(df_p))
    # verify per-group target consistency
    tgt_check = df_p.groupby(cfg.persuade_group_id_col)[cfg.persuade_target_col].nunique()
    inconsistent = (tgt_check > 1).sum()
    logging.info("Found %d groups with multiple unique targets; these will be dropped for consistency.", int(inconsistent))
    consistent_ids = tgt_check[tgt_check == 1].index
    df_p = df_p[df_p[cfg.persuade_group_id_col].isin(consistent_ids)]
    agg_texts = df_p.groupby(cfg.persuade_group_id_col)[cfg.persuade_text_col].apply(lambda s: " ".join([str(x) for x in s.tolist()])).reset_index(name=cfg.text_col)
    agg_targets = df_p.groupby(cfg.persuade_group_id_col)[cfg.persuade_target_col].first().reset_index()
    agg = agg_texts.merge(agg_targets, on=cfg.persuade_group_id_col, how="inner")
    agg = agg.rename(columns={cfg.persuade_target_col: cfg.target_col})
    agg["essay_id"] = "p_" + agg[cfg.persuade_group_id_col].astype(str)
    agg = agg[["essay_id", cfg.text_col, cfg.target_col]]
    agg[cfg.target_col] = agg[cfg.target_col].astype(int)
    logging.info("Aggregated Persuade 2.0 shape: %s", agg.shape)
    # Optional per-class cap to control runtime/memory (customizable)
    if isinstance(cfg.persuade_max_samples_per_class, int):
        logging.info("Applying per-class cap to Persuade: %d per class.", cfg.persuade_max_samples_per_class)
        capped = []
        for k in range(1, cfg.num_labels + 1):
            sub = agg[agg[cfg.target_col] == k]
            if len(sub) > cfg.persuade_max_samples_per_class:
                sub = sub.sample(cfg.persuade_max_samples_per_class, random_state=cfg.seed)
            capped.append(sub)
        agg = pd.concat(capped, axis=0).sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
        logging.info("Post-cap Persuade 2.0 shape: %s", agg.shape)
    return agg

def load_and_merge_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    logging.info("Loading competition and Persuade datasets from base dir: %s", cfg.base_dir)
    comp_train_path = os.path.join(cfg.base_dir, cfg.train_file)
    comp_test_path = os.path.join(cfg.base_dir, cfg.test_file)
    persuade_path = os.path.join(cfg.base_dir, cfg.persuade_file)
    train_df = pd.read_csv(comp_train_path)
    test_df = pd.read_csv(comp_test_path)
    persuade_df = pd.read_csv(persuade_path)
    logging.info("Loaded train: %s, test: %s, persuade: %s", train_df.shape, test_df.shape, persuade_df.shape)

    train_df = train_df[["essay_id", cfg.text_col, cfg.target_col]].copy()
    train_df[cfg.target_col] = train_df[cfg.target_col].astype(int)
    logging.info("Competition train class distribution:\n%s", train_df[cfg.target_col].value_counts().sort_index().to_string())

    agg_persuade = aggregate_persuade(persuade_df)

    # Merge competition + persuade
    merged = pd.concat([train_df, agg_persuade], axis=0).reset_index(drop=True)
    logging.info("Merged dataset shape before dedup: %s", merged.shape)

    # Clean text for ALL datasets consistently
    logging.info("Cleaning text for merged train and test datasets.")
    merged[cfg.text_col] = merged[cfg.text_col].astype(str).map(clean_text)
    test_df[cfg.text_col] = test_df[cfg.text_col].astype(str).map(clean_text)

    # Remove exact-duplicate texts; if duplicates have multiple labels, drop them entirely
    logging.info("Checking for duplicate full_text entries with conflicting labels.")
    g = merged.groupby(cfg.text_col)[cfg.target_col].nunique()
    conflict_texts = set(g[g > 1].index)
    logging.info("Number of conflicting duplicate texts: %d", len(conflict_texts))
    if len(conflict_texts) > 0:
        merged = merged[~merged[cfg.text_col].isin(conflict_texts)]
    before = len(merged)
    merged = merged.drop_duplicates(subset=[cfg.text_col]).reset_index(drop=True)
    logging.info("Dropped %d duplicate rows (keeping first occurrence).", before - len(merged))

    logging.info("Final merged dataset shape: %s", merged.shape)
    logging.info("Merged dataset class distribution:\n%s", merged[cfg.target_col].value_counts().sort_index().to_string())
    return merged, test_df

# ------------------------------------------------------------------
# Dataset and Collator
# ------------------------------------------------------------------
class EssayDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int] = None, tokenizer: AutoTokenizer = None, max_length: int = 512):
        logging.info("Initializing EssayDataset with %d samples. Has labels: %s", len(texts), "yes" if labels is not None else "no")
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_attention_mask=True,
        )
        item = {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
        }
        if self.labels is not None:
            # labels expected in [1..6]; convert to [0..5]
            item["labels"] = torch.tensor(int(self.labels[idx]) - 1, dtype=torch.long)
        return item

class EssayCollator:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            features,
            padding=True,
            max_length=None,
            return_tensors="pt",
        )
        return batch

# ------------------------------------------------------------------
# Training / Evaluation
# ------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.cuda.amp.GradScaler,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
) -> float:
    logging.info("Starting training epoch %d.", epoch + 1)
    model.train()
    running_loss = 0.0
    step = 0
    for it, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)

        scaler.scale(loss).backward()
        if (it + 1) % cfg.grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        running_loss += loss.item()
        step += 1
        if (it + 1) % 50 == 0:
            logging.info("Epoch %d | Step %d/%d | Loss: %.6f", epoch + 1, it + 1, len(loader), loss.item())

    avg_loss = running_loss / max(1, step)
    logging.info("Finished epoch %d | Avg train loss: %.6f", epoch + 1, avg_loss)
    return avg_loss

@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, np.ndarray]:
    logging.info("Running validation.")
    model.eval()
    preds = []
    targets = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        pred = torch.argmax(logits, dim=-1).detach().cpu().numpy() + 1  # back to [1..6]
        tgt = labels.detach().cpu().numpy() + 1
        preds.append(pred)
        targets.append(tgt)

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    qwk = quadratic_weighted_kappa(targets, preds, min_rating=1, max_rating=cfg.num_labels)
    logging.info("Validation QWK: %.6f", qwk)
    return qwk, preds

@torch.no_grad()
def infer(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    logging.info("Running inference on test set.")
    model.eval()
    preds = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        pred = torch.argmax(logits, dim=-1).detach().cpu().numpy() + 1
        pred = np.clip(pred, 1, cfg.num_labels)
        preds.append(pred)
    preds = np.concatenate(preds)
    logging.info("Finished inference; predictions shape: %s", preds.shape)
    return preds

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    logging.info("Starting AES 2.0 training script (v2).")
    set_seed(cfg.seed)

    assert torch.cuda.is_available(), "CUDA is required but not available."
    device = torch.device("cuda")
    logging.info("Using device: %s", device)
    logging.info("CUDA device count: %d", torch.cuda.device_count())
    logging.info("CUDA device name: %s", torch.cuda.get_device_name(0))

    merged_df, test_df = load_and_merge_data()

    # Single stratified split (Fold 0 of KFold) for early iteration as requested
    logging.info("Preparing single-fold stratified train/validation split.")
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)

    fold_pairs = list(skf.split(merged_df[cfg.text_col].values, merged_df[cfg.target_col].values))
    tr_idx, val_idx = fold_pairs[cfg.fold_index]
    tr_df = merged_df.iloc[tr_idx].reset_index(drop=True)
    val_df = merged_df.iloc[val_idx].reset_index(drop=True)

    logging.info("Fold %d | Train size: %d | Val size: %d", cfg.fold_index, len(tr_df), len(val_df))
    logging.info("Train class distribution:\n%s", tr_df[cfg.target_col].value_counts().sort_index().to_string())
    logging.info("Val class distribution:\n%s", val_df[cfg.target_col].value_counts().sort_index().to_string())

    # Tokenizer and model
    logging.info("Loading tokenizer and model: %s", cfg.model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=cfg.num_labels)
    model.to(device)
    logging.info("Model loaded and moved to CUDA.")

    # Datasets and loaders
    train_dataset = EssayDataset(
        texts=tr_df[cfg.text_col].tolist(),
        labels=tr_df[cfg.target_col].astype(int).tolist(),
        tokenizer=tokenizer,
        max_length=cfg.max_length,
    )
    val_dataset = EssayDataset(
        texts=val_df[cfg.text_col].tolist(),
        labels=val_df[cfg.target_col].astype(int).tolist(),
        tokenizer=tokenizer,
        max_length=cfg.max_length,
    )
    test_dataset = EssayDataset(
        texts=test_df[cfg.text_col].tolist(),
        labels=None,
        tokenizer=tokenizer,
        max_length=cfg.max_length,
    )

    collator = EssayCollator(tokenizer=tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.valid_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        collate_fn=collator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.valid_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        collate_fn=collator,
    )
    logging.info("DataLoaders constructed.")

    # Optimizer, scheduler, scaler, and loss with class weights computed ONLY on training split
    logging.info("Setting up optimizer, scheduler, scaler, and loss function with training-only class weights.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    num_update_steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum_steps)
    t_total = cfg.epochs * num_update_steps_per_epoch
    warmup_steps = int(cfg.warmup_ratio * t_total)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    class_weights = compute_class_weights(tr_df[cfg.target_col].values.astype(int), num_labels=cfg.num_labels).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # Training with early stopping on validation QWK
    logging.info("Beginning training with early stopping (patience=%d).", cfg.early_stopping_patience)
    best_qwk = -1.0
    best_epoch = -1
    patience = 0

    for epoch in range(cfg.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, loss_fn, device, epoch)
        val_qwk, _ = validate(model, val_loader, device)
        logging.info("Epoch %d completed | Train loss: %.6f | Val QWK: %.6f", epoch + 1, train_loss, val_qwk)

        if val_qwk > best_qwk:
            best_qwk = val_qwk
            best_epoch = epoch
            patience = 0
            torch.save(model.state_dict(), cfg.save_model_path)
            logging.info("New best model saved at epoch %d with QWK %.6f to %s", epoch + 1, best_qwk, cfg.save_model_path)
        else:
            patience += 1
            logging.info("No improvement this epoch. Patience: %d/%d", patience, cfg.early_stopping_patience)
            if patience > cfg.early_stopping_patience:
                logging.info("Early stopping triggered at epoch %d.", epoch + 1)
                break

    # Load best model for validation logging and test inference
    logging.info("Loading best model from %s (best epoch: %d, best val QWK: %.6f).", cfg.save_model_path, best_epoch + 1, best_qwk)
    model.load_state_dict(torch.load(cfg.save_model_path, map_location=device))
    model.to(device)

    # Final validation evaluation and logging of results (guardrail: must log final validation results)
    final_val_qwk, final_val_preds = validate(model, val_loader, device)
    logging.info("FINAL VALIDATION QWK (single fold %d): %.6f", cfg.fold_index, final_val_qwk)

    # Inference on test set and submission
    test_preds = infer(model, test_loader, device)
    submission = pd.DataFrame({
        "essay_id": test_df["essay_id"].astype(str).values,
        "score": test_preds.astype(int),
    })
    submission.to_csv(cfg.submission_path, index=False)
    logging.info("Submission written to %s with shape %s", cfg.submission_path, submission.shape)

    # Persist run configuration and summary
    summary = {
        "best_epoch": int(best_epoch + 1),
        "best_val_qwk": float(best_qwk),
        "final_val_qwk": float(final_val_qwk),
        "train_size": int(len(tr_df)),
        "val_size": int(len(val_df)),
        "merged_size": int(len(merged_df)),
        "model": cfg.model_name,
        "max_length": cfg.max_length,
        "epochs": cfg.epochs,
        "lr": cfg.lr,
        "train_batch_size": cfg.train_batch_size,
        "valid_batch_size": cfg.valid_batch_size,
    }
    summary_path = os.path.join(cfg.outputs_dir, "run_summary_v2.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logging.info("Run summary saved to %s", summary_path)
    logging.info("All done. Logs are at %s and submission at %s", LOG_FILE_V2, cfg.submission_path)

if __name__ == "__main__":
    main()