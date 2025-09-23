import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import math
import time
import random
import logging
import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup, set_seed

# ---------------------------
# Paths and logging setup
# ---------------------------
BASE_DIR = "task/learning-agency-lab-automated-essay-scoring-2"
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs", "7")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
LOG_FILE = os.path.join(OUTPUTS_DIR, "code_7_v1.txt")

# Logging MUST be configured before any logging statements
logging.basicConfig(
    filename=LOG_FILE,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
# After basicConfig, we can log everywhere
logging.info("Logging initialized. Writing logs to %s", LOG_FILE)

# ---------------------------
# Reproducibility & CUDA checks
# ---------------------------
assert torch.cuda.is_available(), "CUDA is required. No fallback methods are implemented."
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
DEVICE = torch.device("cuda")
logging.info("CUDA is available. Using device: %s", DEVICE)

# ---------------------------
# Config
# ---------------------------
@dataclass
class Config:
    model_name: str = os.environ.get("MODEL_NAME", "microsoft/deberta-v3-large")
    max_length: int = int(os.environ.get("MAX_LEN", 512))
    train_batch_size: int = int(os.environ.get("TRAIN_BS", 8))
    eval_batch_size: int = int(os.environ.get("EVAL_BS", 16))
    lr: float = float(os.environ.get("LR", 5e-5))
    weight_decay: float = float(os.environ.get("WEIGHT_DECAY", 0.01))
    epochs: int = int(os.environ.get("EPOCHS", 3))
    grad_accum_steps: int = int(os.environ.get("ACC_STEPS", 4))
    warmup_ratio: float = float(os.environ.get("WARMUP_RATIO", 0.1))
    seed: int = int(os.environ.get("SEED", 42))
    val_fold_index: int = int(os.environ.get("VAL_FOLD_INDEX", 0))  # single fold for early iteration
    num_labels: int = 6
    max_grad_norm: float = float(os.environ.get("MAX_GRAD_NORM", 1.0))
    logging_steps: int = int(os.environ.get("LOG_STEPS", 100))
    early_stopping_patience: int = int(os.environ.get("PATIENCE", 2))
    do_clean_text: bool = os.environ.get("CLEAN_TEXT", "1") == "1"
    text_col: str = "full_text"
    id_col: str = "essay_id"
    target_col: str = "score"

CFG = Config()
set_seed(CFG.seed)
random.seed(CFG.seed)
np.random.seed(CFG.seed)
logging.info("Config: %s", asdict(CFG))

# ---------------------------
# Utilities
# ---------------------------
def clean_text(text: str) -> str:
    # Standardized preprocessing per plan
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s\.\,\?\!\'\"]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.lower()
    return text

def compute_class_weights(labels: np.ndarray, num_labels: int = 6) -> torch.Tensor:
    counts = np.zeros(num_labels, dtype=np.float64)
    for i in range(1, num_labels + 1):
        counts[i - 1] = (labels == i).sum()
    logging.info("Class counts (1..6): %s", counts.tolist())
    inv = 1.0 / (counts + 1e-6)
    weights = (inv / inv.mean()).astype(np.float32)  # Normalize around mean 1.0
    logging.info("Computed class weights: %s", weights.tolist())
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)

def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray, min_rating: int = 1, max_rating: int = 6) -> float:
    assert y_true.ndim == 1 and y_pred.ndim == 1
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    N = max_rating - min_rating + 1
    # O matrix
    O = np.zeros((N, N), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        O[a - min_rating, b - min_rating] += 1.0
    # Histograms
    act_hist = O.sum(axis=1)
    pred_hist = O.sum(axis=0)
    E = np.outer(act_hist, pred_hist) / O.sum()
    W = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            W[i, j] = ((i - j) ** 2) / ((N - 1) ** 2)
    num = (W * O).sum()
    den = (W * E).sum()
    kappa = 1.0 - num / den if den > 0 else 0.0
    return float(kappa)

# ---------------------------
# Data loading and merging
# ---------------------------
def load_competition_data(base_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_path = os.path.join(base_dir, "train.csv")
    test_path = os.path.join(base_dir, "test.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    logging.info("Loaded competition train: %s, test: %s", train_df.shape, test_df.shape)
    return train_df, test_df

def load_persuade_corpus_aggregated(base_dir: str) -> pd.DataFrame:
    path = os.path.join(base_dir, "persuade_corpus_2.0.csv")
    df = pd.read_csv(path)
    logging.info("Loaded persuade_corpus_2.0.csv shape: %s", df.shape)

    # Determine ID column
    id_col = "essay_id_comp" if "essay_id_comp" in df.columns else ("essay_id" if "essay_id" in df.columns else None)
    assert id_col is not None, "Persuade corpus missing essay id column."
    assert "discourse_text" in df.columns, "Persuade corpus requires 'discourse_text'."
    assert "holistic_essay_score" in df.columns, "Persuade corpus requires 'holistic_essay_score'."

    # Keep rows with valid scores 1..6
    df = df[df["holistic_essay_score"].between(1, 6)]
    logging.info("Filtered persuade rows to valid scores 1..6: %s", df.shape)

    # Internal consistency: ensure each essay_id has a single unique score
    consistency = df.groupby(id_col)["holistic_essay_score"].nunique().reset_index(name="n_unique_scores")
    consistent_ids = consistency[consistency["n_unique_scores"] == 1][id_col]
    df = df[df[id_col].isin(consistent_ids)]
    logging.info("After enforcing single unique score per essay: %s", df.shape)

    # Aggregate discourse segments into full essay text
    agg_text = df.groupby(id_col)["discourse_text"].apply(lambda s: " ".join([str(x) for x in s])).reset_index(name="full_text")
    agg_score = df.groupby(id_col)["holistic_essay_score"].first().reset_index()
    merged = agg_text.merge(agg_score, on=id_col, how="inner")
    merged = merged.rename(columns={id_col: "essay_id", "holistic_essay_score": "score"})
    logging.info("Aggregated persuade essays: %s", merged.shape)
    return merged[["essay_id", "full_text", "score"]]

def prepare_and_merge(base_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comp_train, comp_test = load_competition_data(base_dir)
    persu = load_persuade_corpus_aggregated(base_dir)

    # Standardize columns
    comp_train = comp_train[[CFG.id_col, CFG.text_col, CFG.target_col]].copy()
    comp_test = comp_test[[CFG.id_col, CFG.text_col]].copy()

    # Optional cleaning
    if CFG.do_clean_text:
        logging.info("Applying standardized text cleaning to competition train/test and persuade data.")
        comp_train[CFG.text_col] = comp_train[CFG.text_col].astype(str).map(clean_text)
        comp_test[CFG.text_col] = comp_test[CFG.text_col].astype(str).map(clean_text)
        persu[CFG.text_col] = persu[CFG.text_col].astype(str).map(clean_text)

    # Merge: competition train + persuade
    comp_train["source"] = "comp"
    persu["source"] = "persuade"
    merged = pd.concat([comp_train, persu], axis=0, ignore_index=True)

    # Drop duplicates by full_text to mitigate leakage of identical essays across folds
    before = merged.shape[0]
    merged = merged.drop_duplicates(subset=[CFG.text_col]).reset_index(drop=True)
    after = merged.shape[0]
    logging.info("Merged dataset size before/after dropping duplicate texts: %d -> %d", before, after)

    # Keep only valid 1..6 scores
    merged = merged[merged[CFG.target_col].between(1, 6)].reset_index(drop=True)
    logging.info("Final merged dataset shape: %s", merged.shape)

    # Log class distribution
    cls_dist = merged[CFG.target_col].value_counts().sort_index()
    logging.info("Merged class distribution (1..6): %s", cls_dist.to_dict())

    return merged, comp_train, comp_test

# ---------------------------
# Dataset
# ---------------------------
class EssayDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: AutoTokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        if self.labels is not None:
            # Labels expected as 0..5
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# ---------------------------
# Split (single fold validation)
# ---------------------------
def make_single_fold_split(df: pd.DataFrame, fold_index: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    # Stratified split: use sklearn-like behavior without importing heavy modules
    # We will sort by label and do a simple modulo fold assignment to approximate stratification.
    df = df.copy()
    df["_order"] = df.groupby(CFG.target_col).cumcount()
    df = df.sort_values([CFG.target_col, "_order"]).reset_index(drop=True)
    # Create 5 pseudo-folds
    n_folds = 5
    fold_ids = np.arange(len(df)) % n_folds
    val_mask = fold_ids == (fold_index % n_folds)
    train_idx = df.index[~val_mask].to_numpy()
    val_idx = df.index[val_mask].to_numpy()
    logging.info(
        "Single-fold split with %d folds: using fold %d for validation. Train size: %d, Val size: %d",
        n_folds, fold_index, len(train_idx), len(val_idx)
    )
    # Log distribution
    val_dist = df.loc[val_idx, CFG.target_col].value_counts().sort_index()
    train_dist = df.loc[train_idx, CFG.target_col].value_counts().sort_index()
    logging.info("Train distribution (1..6): %s", train_dist.to_dict())
    logging.info("Val distribution (1..6): %s", val_dist.to_dict())
    return train_idx, val_idx

# ---------------------------
# Training & Evaluation
# ---------------------------
def train_one_epoch(
    model: AutoModelForSequenceClassification,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.cuda.amp.GradScaler,
    class_weights: torch.Tensor,
    epoch: int,
    total_epochs: int,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_examples = 0
    start_time = time.time()

    for step, batch in enumerate(loader, 1):
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        labels = batch["labels"].to(DEVICE, non_blocking=True)

        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = F.cross_entropy(logits, labels, weight=class_weights)
            loss = loss / CFG.grad_accum_steps

        scaler.scale(loss).backward()

        if step % CFG.grad_accum_steps == 0:
            scaler.unscale_gradients(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        total_loss += loss.item() * CFG.grad_accum_steps
        total_examples += labels.size(0)

        if step % CFG.logging_steps == 0:
            elapsed = time.time() - start_time
            logging.info(
                "Epoch [%d/%d] Step [%d/%d] - Loss: %.4f - Elapsed: %.1fs",
                epoch, total_epochs, step, len(loader), total_loss / (step), elapsed
            )

    avg_loss = total_loss / len(loader)
    tok_per_step = (CFG.train_batch_size * CFG.grad_accum_steps * CFG.max_length)
    logging.info(
        "Epoch %d finished. Avg train loss: %.4f. Approx tokens/optimizer-step: %d",
        epoch, avg_loss, tok_per_step
    )
    return avg_loss, total_examples

@torch.no_grad()
def evaluate(
    model: AutoModelForSequenceClassification,
    loader: DataLoader,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    preds_all: List[int] = []
    labels_all: List[int] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        labels = batch["labels"].to(DEVICE, non_blocking=True)

        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = F.cross_entropy(logits, labels)
        total_loss += loss.item()

        pred = torch.argmax(logits, dim=-1)  # 0..5
        preds_all.append(pred.detach().cpu().numpy())
        labels_all.append(labels.detach().cpu().numpy())

    preds_all = np.concatenate(preds_all, axis=0)
    labels_all = np.concatenate(labels_all, axis=0)
    # Convert to 1..6 scale
    preds_scores = np.clip(preds_all + 1, 1, 6)
    labels_scores = labels_all + 1
    qwk = quadratic_weighted_kappa(labels_scores, preds_scores, min_rating=1, max_rating=6)
    avg_loss = total_loss / len(loader)
    logging.info("Validation - Avg loss: %.4f | QWK: %.6f", avg_loss, qwk)
    return avg_loss, qwk, labels_scores, preds_scores

@torch.no_grad()
def infer_on_test(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    texts = test_df[CFG.text_col].astype(str).tolist()
    ds = EssayDataset(texts=texts, labels=None, tokenizer=tokenizer, max_length=CFG.max_length)

    loader = DataLoader(
        ds,
        batch_size=CFG.eval_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )
    model.eval()
    preds_all: List[int] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        pred = torch.argmax(logits, dim=-1)  # 0..5
        preds_all.append(pred.detach().cpu().numpy())

    preds_all = np.concatenate(preds_all, axis=0)
    scores = np.clip(preds_all + 1, 1, 6)
    out = pd.DataFrame({CFG.id_col: test_df[CFG.id_col].values, CFG.target_col: scores})
    logging.info("Inference on test complete. Pred shape: %s", out.shape)
    return out

# ---------------------------
# Main execution
# ---------------------------
def main():
    logging.info("Starting pipeline.")
    merged_df, comp_train_df, comp_test_df = prepare_and_merge(BASE_DIR)

    # Prepare tokenizer and model
    logging.info("Loading tokenizer and model: %s", CFG.model_name)
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        CFG.model_name,
        num_labels=CFG.num_labels,
        problem_type="single_label_classification",
    ).to(DEVICE)
    logging.info("Model loaded and moved to CUDA.")

    # Build single-fold split on merged data
    train_idx, val_idx = make_single_fold_split(merged_df, fold_index=CFG.val_fold_index)
    tr_df = merged_df.iloc[train_idx].reset_index(drop=True)
    va_df = merged_df.iloc[val_idx].reset_index(drop=True)

    # Labels transformed to 0..5 for CE
    y_train = (tr_df[CFG.target_col].values.astype(int) - 1).tolist()
    y_val = (va_df[CFG.target_col].values.astype(int) - 1).tolist()

    # Compute class weights from FULL merged dataset (as requested)
    class_weights = compute_class_weights(merged_df[CFG.target_col].values.astype(int), num_labels=CFG.num_labels)

    # Datasets and loaders
    train_texts = tr_df[CFG.text_col].astype(str).tolist()
    val_texts = va_df[CFG.text_col].astype(str).tolist()
    train_ds = EssayDataset(train_texts, y_train, tokenizer, CFG.max_length)
    val_ds = EssayDataset(val_texts, y_val, tokenizer, CFG.max_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.train_batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CFG.eval_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )
    logging.info("DataLoaders ready. Train steps/epoch: %d, Val steps: %d", len(train_loader), len(val_loader))

    # Optimizer, scheduler, scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    total_training_steps = math.ceil(len(train_loader) / CFG.grad_accum_steps) * CFG.epochs
    warmup_steps = int(CFG.warmup_ratio * total_training_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    logging.info("Training setup: total_steps=%d, warmup_steps=%d", total_training_steps, warmup_steps)

    # Training loop with early stopping on QWK
    best_qwk = -1.0
    best_epoch = -1
    patience_counter = 0
    best_ckpt_path = os.path.join(OUTPUTS_DIR, "best_model.pt")

    for epoch in range(1, CFG.epochs + 1):
        train_loss, _ = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, class_weights, epoch, CFG.epochs)
        val_loss, val_qwk, y_true, y_pred = evaluate(model, val_loader)

        logging.info("Epoch %d summary - TrainLoss: %.4f | ValLoss: %.4f | ValQWK: %.6f", epoch, train_loss, val_loss, val_qwk)

        if val_qwk > best_qwk:
            best_qwk = val_qwk
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_ckpt_path)
            logging.info("New best QWK %.6f at epoch %d. Saved checkpoint to %s", best_qwk, best_epoch, best_ckpt_path)
        else:
            patience_counter += 1
            logging.info("No improvement in QWK. Patience counter: %d / %d", patience_counter, CFG.early_stopping_patience)

        if patience_counter > CFG.early_stopping_patience:
            logging.info("Early stopping triggered at epoch %d.", epoch)
            break

    logging.info("Training finished. Best Val QWK: %.6f at epoch %d", best_qwk, best_epoch)
    logging.info("Final validation results logged above.")

    # Load best checkpoint for inference
    model.load_state_dict(torch.load(best_ckpt_path, map_location=DEVICE))
    logging.info("Loaded best model checkpoint for inference.")

    # Inference on competition test
    if CFG.do_clean_text:
        comp_test_df[CFG.text_col] = comp_test_df[CFG.text_col].astype(str).map(clean_text)

    submission = infer_on_test(model, tokenizer, comp_test_df)
    submission_path = os.path.join(OUTPUTS_DIR, "submission_1.csv")
    submission.to_csv(submission_path, index=False)
    logging.info("Saved submission to %s with shape %s", submission_path, submission.shape)

    # Sanity log: show head of submission
    logging.info("Submission head:\n%s", submission.head().to_string(index=False))

if __name__ == "__main__":
    main()