import os
import re
import time
import logging
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)
from torch.optim import AdamW

# =========================
# Configuration and Logging
# =========================

BASE_DIR = "task/learning-agency-lab-automated-essay-scoring-2"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "5")
LOG_FILE = os.path.join(OUTPUT_DIR, "code_5_v5.txt")
SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")

MODEL_NAME = "microsoft/deberta-v3-large"
MAX_LENGTH = 512
NUM_LABELS = 6
N_FOLDS = 5
SEED = 42

BASE_LR = 2e-5
HEAD_LR = 3e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
NUM_EPOCHS_FOLD = 5
PATIENCE = 2  # early stopping patience
FINAL_EPOCHS = 3

PER_DEVICE_TRAIN_BS = 2            # with gradient accumulation to reach effective ~16
GRAD_ACC_STEPS = 8
PER_DEVICE_EVAL_BS = 4
NUM_WORKERS = 2

os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger(__name__)
logging.info("Initialized logging.")


# =========================
# Utilities
# =========================

def set_all_seeds(seed: int):
    logging.info(f"Setting all seeds to {seed}.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def get_device():
    logging.info("Acquiring CUDA device.")
    assert torch.cuda.is_available(), "CUDA is required but not available."
    device = torch.device("cuda")
    logging.info(f"Using device: {device}, name: {torch.cuda.get_device_name(0)}")
    return device


# =========================
# Text Preprocessing
# =========================

_contractions = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there would",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "where'd": "where did",
    "where's": "where is",
    "who'll": "who will",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "won't": "will not",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
}

_contr_re = re.compile("|".join(sorted(map(re.escape, _contractions.keys()), key=len, reverse=True)))
_html_re = re.compile(r"<.*?>")
_url_re = re.compile(r"https?://\S+|www\.\S+")
_email_re = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w{2,4}\b")
_non_alnum_basic_punct_re = re.compile(r"[^a-z0-9\s\.\,\!\?\;\:\'\-\(\)]")
_multi_space_re = re.compile(r"\s+")
_multi_punct_space_re = re.compile(r"\s*([,;:\.\!\?])\s*")


def expand_contractions(text: str) -> str:
    def replace(match):
        return _contractions[match.group(0)]
    return _contr_re.sub(replace, text)


def clean_text_series(series: pd.Series) -> pd.Series:
    logging.info("Starting text cleaning.")
    s = series.astype(str).str.lower()
    s = s.apply(lambda x: _html_re.sub(" ", x))
    s = s.apply(lambda x: _url_re.sub(" ", x))
    s = s.apply(lambda x: _email_re.sub(" ", x))
    s = s.apply(expand_contractions)
    s = s.apply(lambda x: _non_alnum_basic_punct_re.sub(" ", x))
    s = s.apply(lambda x: _multi_punct_space_re.sub(r" \1 ", x))
    s = s.apply(lambda x: _multi_space_re.sub(" ", x).strip())
    logging.info("Completed text cleaning.")
    return s


# =========================
# Dataset
# =========================

class EncodedDataset(Dataset):
    def __init__(self, encodings: dict, indices: np.ndarray, labels: np.ndarray = None):
        self.encodings = encodings
        self.indices = indices
        self.labels = labels
        logging.info(f"Created EncodedDataset with {len(self.indices)} samples. Labels provided: {labels is not None}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])
        item = {
            "input_ids": torch.tensor(self.encodings["input_ids"][real_idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][real_idx], dtype=torch.long),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[real_idx]), dtype=torch.long)
        return item


# =========================
# Trainer with Weighted Loss and Param Groups
# =========================

class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, base_lr: float, head_lr: float, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        self.base_lr = base_lr
        self.head_lr = head_lr
        logging.info(f"Initialized WeightedLossTrainer with base_lr={base_lr}, head_lr={head_lr}")

    def create_optimizer(self):
        logging.info("Creating AdamW optimizer with differential learning rates and weight decay.")
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias", "layer_norm.weight", "layer_norm.bias"]
        head_keywords = ["classifier", "pooler"]
        decay_params_base = []
        nodecay_params_base = []
        decay_params_head = []
        nodecay_params_head = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            use_decay = not any(nd in name for nd in no_decay)
            is_head = any(hk in name for hk in head_keywords)
            if is_head:
                if use_decay:
                    decay_params_head.append(param)
                else:
                    nodecay_params_head.append(param)
            else:
                if use_decay:
                    decay_params_base.append(param)
                else:
                    nodecay_params_base.append(param)

        param_groups = []
        if decay_params_base:
            param_groups.append({"params": decay_params_base, "weight_decay": WEIGHT_DECAY, "lr": self.base_lr})
        if nodecay_params_base:
            param_groups.append({"params": nodecay_params_base, "weight_decay": 0.0, "lr": self.base_lr})
        if decay_params_head:
            param_groups.append({"params": decay_params_head, "weight_decay": WEIGHT_DECAY, "lr": self.head_lr})
        if nodecay_params_head:
            param_groups.append({"params": nodecay_params_head, "weight_decay": 0.0, "lr": self.head_lr})

        self.optimizer = AdamW(param_groups)
        logging.info(f"Optimizer created with {len(param_groups)} parameter groups.")
        return self.optimizer

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
        **kwargs,
    ):
        labels = inputs.get("labels")
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
        weight = self.class_weights.to(device=logits.device, dtype=logits.dtype)
        loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels),
            labels.view(-1),
        )
        return (loss, outputs) if return_outputs else loss


# =========================
# Metrics
# =========================

def expected_rounded_scores_from_logits(logits: np.ndarray) -> np.ndarray:
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    classes = np.arange(1, NUM_LABELS + 1, dtype=np.float32)
    expected = (probs * classes[None, :]).sum(axis=1)
    rounded = np.rint(expected)
    clipped = np.clip(rounded, 1, NUM_LABELS)
    return clipped.astype(int)


def compute_qwk(eval_pred):
    preds = eval_pred.predictions
    labels = eval_pred.label_ids
    preds_int = expected_rounded_scores_from_logits(preds)
    true_int = labels.astype(int) + 1  # labels are 0..5; convert to 1..6
    qwk = cohen_kappa_score(true_int, preds_int, weights="quadratic")
    return {"qwk": float(qwk)}


# =========================
# Main pipeline
# =========================

def main():
    torch.set_float32_matmul_precision("medium")
    start_time = time.time()
    set_all_seeds(SEED)
    device = get_device()

    logging.info("Reading datasets.")
    train_path = os.path.join(BASE_DIR, "train.csv")
    test_path = os.path.join(BASE_DIR, "test.csv")
    sample_sub_path = os.path.join(BASE_DIR, "sample_submission.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample_sub = pd.read_csv(sample_sub_path)

    logging.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}, Sample sub shape: {sample_sub.shape}")

    logging.info("Applying text preprocessing to train and test.")
    train_df["full_text_clean"] = clean_text_series(train_df["full_text"])
    test_df["full_text_clean"] = clean_text_series(test_df["full_text"])

    logging.info(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, local_files_only=True)

    logging.info("Tokenizing full training and test sets.")
    train_texts = train_df["full_text_clean"].tolist()
    test_texts = test_df["full_text_clean"].tolist()

    train_enc = tokenizer(
        train_texts,
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
    )
    test_enc = tokenizer(
        test_texts,
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
    )

    train_enc = {k: np.array(v, dtype=np.int64) for k, v in train_enc.items()}
    test_enc = {k: np.array(v, dtype=np.int64) for k, v in test_enc.items()}

    # Labels: convert 1..6 to 0..5
    y = train_df["score"].values.astype(int) - 1

    # Class weights: total / (6 * count_i)
    logging.info("Computing class weights.")
    counts = np.bincount(y, minlength=NUM_LABELS).astype(np.float32)
    total = float(len(y))
    weights = total / (NUM_LABELS * counts)
    class_weights = torch.tensor(weights, dtype=torch.float32)
    logging.info(f"Class counts: {counts.tolist()}, Class weights: {weights.tolist()}")

    # Cross-validation
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        logging.info(f"===== Fold {fold}/{N_FOLDS} =====")
        fold_dir = os.path.join(OUTPUT_DIR, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        train_ds = EncodedDataset(train_enc, tr_idx, y)
        valid_ds = EncodedDataset(train_enc, va_idx, y)

        logging.info(f"Loading model for fold {fold}: {MODEL_NAME}")
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=NUM_LABELS, local_files_only=True
        )
        model.to(device)
        model.gradient_checkpointing_enable()
        logging.info("Model loaded and moved to CUDA. Gradient checkpointing enabled.")

        args = TrainingArguments(
            output_dir=fold_dir,
            learning_rate=BASE_LR,
            weight_decay=WEIGHT_DECAY,
            per_device_train_batch_size=PER_DEVICE_TRAIN_BS,
            per_device_eval_batch_size=PER_DEVICE_EVAL_BS,
            gradient_accumulation_steps=GRAD_ACC_STEPS,
            num_train_epochs=NUM_EPOCHS_FOLD,
            logging_strategy="steps",
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="qwk",
            greater_is_better=True,
            fp16=True,
            dataloader_num_workers=NUM_WORKERS,
            report_to="none",
            optim="adamw_torch",
            lr_scheduler_type="linear",
            warmup_ratio=WARMUP_RATIO,
            seed=SEED,
            ddp_find_unused_parameters=False
        )

        trainer = WeightedLossTrainer(
            class_weights=class_weights,
            base_lr=BASE_LR,
            head_lr=HEAD_LR,
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=valid_ds,
            tokenizer=tokenizer,
            compute_metrics=compute_qwk,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)]
        )

        logging.info("Starting training for this fold.")
        t0 = time.time()
        trainer.train()
        fold_train_time = time.time() - t0
        logging.info(f"Training completed for fold {fold} in {fold_train_time/60:.2f} minutes.")

        logging.info("Evaluating best model on validation set.")
        eval_metrics = trainer.evaluate()
        fold_qwk = float(eval_metrics["eval_qwk"])
        fold_scores.append(fold_qwk)
        logging.info(f"Fold {fold} Validation QWK: {fold_qwk:.6f}")

        trainer.save_model(fold_dir)
        logging.info(f"Saved best model to {fold_dir}.")

        del trainer
        del model
        torch.cuda.empty_cache()

    mean_qwk = float(np.mean(fold_scores))
    std_qwk = float(np.std(fold_scores))
    logging.info("===== Cross-Validation Results =====")
    logging.info(f"Fold QWKs: {', '.join(f'{s:.6f}' for s in fold_scores)}")
    logging.info(f"Mean QWK: {mean_qwk:.6f}, Std QWK: {std_qwk:.6f}")

    # Final training on full dataset (no early stopping)
    logging.info("===== Final training on full training set =====")
    full_dir = os.path.join(OUTPUT_DIR, "final_full_train")
    os.makedirs(full_dir, exist_ok=True)

    full_indices = np.arange(len(y))
    full_ds = EncodedDataset(train_enc, full_indices, y)

    logging.info(f"Loading model for final training: {MODEL_NAME}")
    final_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS, local_files_only=True
    )
    final_model.to(device)
    final_model.gradient_checkpointing_enable()
    logging.info("Final model loaded and moved to CUDA. Gradient checkpointing enabled.")

    final_args = TrainingArguments(
        output_dir=full_dir,
        learning_rate=BASE_LR,
        weight_decay=WEIGHT_DECAY,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BS,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BS,
        gradient_accumulation_steps=GRAD_ACC_STEPS,
        num_train_epochs=FINAL_EPOCHS,
        logging_strategy="steps",
        logging_steps=100,
        eval_strategy="no",
        save_strategy="no",
        load_best_model_at_end=False,
        fp16=True,
        dataloader_num_workers=NUM_WORKERS,
        report_to="none",
        optim="adamw_torch",
        lr_scheduler_type="linear",
        warmup_ratio=WARMUP_RATIO,
        seed=SEED,
        ddp_find_unused_parameters=False
    )

    final_trainer = WeightedLossTrainer(
        class_weights=class_weights,
        base_lr=BASE_LR,
        head_lr=HEAD_LR,
        model=final_model,
        args=final_args,
        train_dataset=full_ds,
        eval_dataset=None,
        tokenizer=tokenizer,
        compute_metrics=None,
        callbacks=[]
    )

    logging.info("Starting final full-data training.")
    t0_full = time.time()
    final_trainer.train()
    final_train_time = time.time() - t0_full
    logging.info(f"Final training completed in {final_train_time/60:.2f} minutes.")

    # Inference on test
    logging.info("===== Generating predictions for test set =====")
    test_indices = np.arange(test_enc["input_ids"].shape[0])
    test_ds = EncodedDataset(test_enc, test_indices, labels=None)
    test_loader = DataLoader(
        test_ds, batch_size=PER_DEVICE_EVAL_BS * 4, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )

    model_for_infer = final_trainer.model.eval()
    logging.info("Compiling model for inference with torch.compile (if available).")
    if hasattr(torch, "compile"):
        model_for_infer = torch.compile(model_for_infer)

    all_preds = []
    t0_pred = time.time()
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model_for_infer(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                classes = torch.arange(1, NUM_LABELS + 1, device=probs.device, dtype=probs.dtype)
                expected = (probs * classes[None, :]).sum(dim=1)
                rounded = torch.round(expected).clamp(min=1, max=NUM_LABELS).to(torch.int)
                all_preds.append(rounded.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    pred_time = time.time() - t0_pred
    logging.info(f"Inference completed in {pred_time:.2f} seconds.")

    submission = pd.DataFrame({
        "essay_id": test_df["essay_id"].values,
        "score": all_preds.astype(int)
    })
    submission.to_csv(SUBMISSION_PATH, index=False)
    logging.info(f"Submission saved to {SUBMISSION_PATH}")

    total_time = time.time() - start_time
    logging.info("===== Final Validation Results =====")
    logging.info(f"Mean CV QWK: {mean_qwk:.6f}, Std: {std_qwk:.6f}")
    logging.info(f"Total runtime: {total_time/60:.2f} minutes")


if __name__ == "__main__":
    main()