# Filename: code_4_v2.py
import os
import sys
import gc
import math
import time
import random
import logging
import traceback
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
logging.info("Starting script code_4_v2.py")

# Constants and paths
BASE_DIR = "task/learning-agency-lab-automated-essay-scoring-2"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "4")
SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")

TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TEST_CSV = os.path.join(BASE_DIR, "test.csv")
SAMPLE_SUBMISSION_CSV = os.path.join(BASE_DIR, "sample_submission.csv")

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

MODEL_CANDIDATES = [
    # Prefer the target model first
    # "bert-base-uncased",
    # Potential local backups on some Kaggle images
    # "roberta-base",
    # "distilbert-base-uncased",
    "microsoft/deberta-v3-base",
]

NUM_CLASSES = 6  # labels 1..6


def set_torch_seed(seed: int = SEED):
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True  # speedup
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        logging.info(f"Set PyTorch seed to {seed}")
    except Exception as e:
        logging.info(f"Could not set PyTorch seed due to: {e}")


def get_device():
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logging.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            logging.info("CUDA not available. Using CPU.")
        return device
    except Exception as e:
        logging.info(f"Error determining device: {e}")
        return None


def log_cuda_mem(prefix: str = ""):
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            logging.info(f"{prefix} CUDA Memory - Allocated: {allocated:.1f} MB | Reserved: {reserved:.1f} MB")
    except Exception as e:
        logging.info(f"Could not log CUDA memory: {e}")


def compute_class_weights(y: np.ndarray) -> np.ndarray:
    logging.info("Computing class weights for cross-entropy")
    counts = pd.Series(y).value_counts().sort_index()
    max_count = counts.max()
    weights = np.array([max_count / counts.get(i, 1) for i in range(1, NUM_CLASSES + 1)], dtype=np.float32)
    logging.info(f"Class counts: {counts.to_dict()} | Weights: {weights.tolist()}")
    return weights


def qwk_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import cohen_kappa_score
    try:
        score = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    except Exception as e:
        logging.info(f"Error computing QWK: {e}")
        score = 0.0
    return score


def ensure_output_dir():
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logging.info(f"Ensured output directory exists: {OUTPUT_DIR}")
    except Exception as e:
        logging.info(f"Failed to create output directory: {e}")


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    logging.info("Loading train and test data")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    logging.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    return train_df, test_df


def stratified_kfold_split(y: np.ndarray, n_splits: int = 5, random_state: int = SEED):
    logging.info(f"Generating Stratified {n_splits}-Fold indices")
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(skf.split(np.zeros_like(y), y))


def try_import_transformers() -> bool:
    try:
        import torch  # noqa: F401
        from transformers import (  # noqa: F401
            AutoTokenizer,
            AutoModelForSequenceClassification,
            DataCollatorWithPadding,
            get_linear_schedule_with_warmup,
        )
        logging.info("Transformers and torch successfully imported")
        return True
    except Exception as e:
        logging.info(f"Could not import transformers or torch: {e}")
        return False


def try_load_tokenizer_and_model(model_names: List[str], num_labels: int = NUM_CLASSES):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    last_error = None
    for name in model_names:
        try:
            logging.info(f"Attempting to load tokenizer locally: {name}")
            tok = AutoTokenizer.from_pretrained(name, use_fast=True, local_files_only=True)
            logging.info(f"Tokenizer loaded: {name}")
            logging.info(f"Attempting to load model locally: {name}")
            mdl = AutoModelForSequenceClassification.from_pretrained(
                name,
                num_labels=num_labels,
                problem_type="single_label_classification",
                local_files_only=True,
            )
            logging.info(f"Model loaded: {name}")
            return name, tok, mdl
        except Exception as e:
            last_error = e
            logging.info(f"Failed to load {name} locally: {e}")
    raise RuntimeError(f"Could not load any local model from candidates {model_names}. Last error: {last_error}")


class EssayDataset:
    def __init__(self, encodings: Dict[str, List[List[int]]], labels: Optional[List[int]] = None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
        }
        if "token_type_ids" in self.encodings:
            item["token_type_ids"] = self.encodings["token_type_ids"][idx]
        if self.labels is not None:
            item["labels"] = int(self.labels[idx])
        return item


def tokenize_texts(tokenizer, texts: List[str], max_length: int = 512) -> Dict[str, List[List[int]]]:
    logging.info(f"Tokenizing {len(texts)} texts with max_length={max_length}")
    enc = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_attention_mask=True,
        return_token_type_ids="token_type_ids" in tokenizer.model_input_names,
    )
    return {k: enc[k] for k in enc}


def get_autocast_dtype():
    try:
        import torch
        if torch.cuda.is_available():
            # Prefer bfloat16 on newer GPUs if available for stability
            capability = torch.cuda.get_device_capability()
            if capability[0] >= 8:
                logging.info("Using autocast dtype bfloat16")
                return torch.bfloat16
            else:
                logging.info("Using autocast dtype float16")
                return torch.float16
    except Exception as e:
        logging.info(f"Could not determine autocast dtype: {e}")
    return None


def train_one_fold(model, device, train_loader, valid_loader, criterion, optimizer, scheduler,
                   epochs: int, use_cuda: bool, grad_accum_steps: int = 1, patience: int = 2) -> Tuple[Dict, float]:
    import torch

    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)
    best_qwk = -1.0
    best_state_dict = None
    epochs_no_improve = 0
    autocast_dtype = get_autocast_dtype()

    for epoch in range(epochs):
        start_ep = time.time()
        logging.info(f"Epoch {epoch + 1}/{epochs} - Training started")
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            labels = batch.pop("labels").to(device)
            for k in batch:
                batch[k] = batch[k].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda" if use_cuda else "cpu", dtype=autocast_dtype, enabled=True if use_cuda else False):
                outputs = model(**batch)
                loss = criterion(outputs.logits, labels)

            if use_cuda:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % grad_accum_steps == 0:
                if use_cuda:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
            if (step + 1) % 100 == 0:
                logging.info(f"Epoch {epoch+1} | Step {step+1}/{len(train_loader)} | Loss: {running_loss / (step+1):.4f}")
        avg_train_loss = running_loss / max(1, len(train_loader))
        logging.info(f"Epoch {epoch+1} - Training finished | Avg Loss: {avg_train_loss:.4f} | Time: {(time.time() - start_ep):.1f}s")
        log_cuda_mem(prefix=f"After epoch {epoch+1}:")

        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch in valid_loader:
                labels_np = batch.pop("labels").cpu().numpy()
                for k in batch:
                    batch[k] = batch[k].to(device, non_blocking=True)
                with torch.autocast(device_type="cuda" if use_cuda else "cpu", dtype=autocast_dtype, enabled=True if use_cuda else False):
                    outputs = model(**batch)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
                preds = probs.argmax(axis=1)
                val_preds.append(preds)
                val_labels.append(labels_np)
        val_preds = np.concatenate(val_preds)
        val_labels = np.concatenate(val_labels)
        val_qwk = qwk_score(val_labels + 1, val_preds + 1)
        logging.info(f"Epoch {epoch+1} - Validation QWK: {val_qwk:.6f}")

        # Early stopping
        if val_qwk > best_qwk:
            best_qwk = val_qwk
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            logging.info(f"Epoch {epoch+1} - New best QWK: {best_qwk:.6f}")
        else:
            epochs_no_improve += 1
            logging.info(f"Epoch {epoch+1} - No improvement (patience {epochs_no_improve}/{patience})")
            if epochs_no_improve >= patience:
                logging.info("Early stopping triggered")
                break

        # Cleanup
        del val_preds, val_labels
        gc.collect()
        if use_cuda:
            torch.cuda.empty_cache()
    return best_state_dict, best_qwk


def adaptive_build_loaders(dataset_train_subset, dataset_val_subset, dataset_test, data_collator, use_cuda: bool,
                           init_train_bs: int, init_valid_bs: int, max_retries: int = 3):
    """
    Try to build loaders with given batch sizes. If CUDA OOM occurs during a quick dry run, reduce batch sizes and retry.
    """
    import torch
    from torch.utils.data import DataLoader

    train_bs = init_train_bs
    valid_bs = init_valid_bs

    for attempt in range(max_retries):
        try:
            logging.info(f"Building DataLoaders with train_bs={train_bs}, valid/test_bs={valid_bs} (attempt {attempt+1}/{max_retries})")
            train_loader = DataLoader(
                dataset_train_subset, batch_size=train_bs, shuffle=True,
                num_workers=2, pin_memory=use_cuda, collate_fn=data_collator
            )
            valid_loader = DataLoader(
                dataset_val_subset, batch_size=valid_bs, shuffle=False,
                num_workers=2, pin_memory=use_cuda, collate_fn=data_collator
            )
            test_loader = DataLoader(
                dataset_test, batch_size=valid_bs, shuffle=False,
                num_workers=2, pin_memory=use_cuda, collate_fn=data_collator
            )

            # Quick dry run to ensure memory is fine
            if len(train_loader) > 0:
                batch = next(iter(train_loader))
                # just move tensors to device to test memory transfer
                device = get_device()
                for k in batch:
                    if isinstance(batch[k], (np.ndarray, int)):
                        continue
                    try:
                        if hasattr(batch[k], "to"):
                            batch[k] = batch[k].to(device, non_blocking=True)
                    except Exception:
                        pass
                del batch
            logging.info("Successfully built DataLoaders and dry-run passed")
            return train_loader, valid_loader, test_loader, train_bs, valid_bs
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logging.info(f"OOM when building loaders: {e}. Reducing batch sizes and retrying...")
                train_bs = max(1, train_bs // 2)
                valid_bs = max(4, valid_bs // 2)
                gc.collect()
                if use_cuda:
                    torch.cuda.empty_cache()
                continue
            else:
                logging.info(f"RuntimeError when building loaders: {e}")
                raise
        except Exception as e:
            logging.info(f"Error when building loaders: {e}")
            raise
    logging.info("Exceeded maximum retries for DataLoader building due to OOM; proceeding with smallest batch sizes")
    # Final attempt with smallest
    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        dataset_train_subset, batch_size=max(1, train_bs), shuffle=True,
        num_workers=2, pin_memory=use_cuda, collate_fn=data_collator
    )
    valid_loader = DataLoader(
        dataset_val_subset, batch_size=max(4, valid_bs), shuffle=False,
        num_workers=2, pin_memory=use_cuda, collate_fn=data_collator
    )
    test_loader = DataLoader(
        dataset_test, batch_size=max(4, valid_bs), shuffle=False,
        num_workers=2, pin_memory=use_cuda, collate_fn=data_collator
    )
    return train_loader, valid_loader, test_loader, train_bs, valid_bs


def train_and_predict_transformer(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_folds: int = 5,
    max_len: int = 512,
    epochs: int = 3,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    grad_accum_steps: int = 1,
    train_bs: int = 8,
    valid_bs: int = 16,
):
    logging.info("Starting Transformer training and prediction pipeline")

    import torch
    from torch.utils.data import Subset
    from transformers import DataCollatorWithPadding, get_linear_schedule_with_warmup

    device = get_device()
    use_cuda = device is not None and device.type == "cuda"
    if device is None:
        raise RuntimeError("Torch device could not be initialized")

    set_torch_seed(SEED)

    # Load tokenizer and base model (local only)
    model_name, tokenizer, base_model = try_load_tokenizer_and_model(MODEL_CANDIDATES, num_labels=NUM_CLASSES)
    logging.info(f"Using model: {model_name}")

    # Enable gradient checkpointing if available
    try:
        base_model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing")
    except Exception as e:
        logging.info(f"Could not enable gradient checkpointing: {e}")

    # Torch compile if available (PyTorch 2+)
    try:
        if use_cuda and hasattr(torch, "compile"):
            base_model = torch.compile(base_model)
            logging.info("Compiled model with torch.compile for speed")
    except Exception as e:
        logging.info(f"torch.compile not used: {e}")

    # Labels mapping: scores 1..6 -> 0..5
    y_all = train_df["score"].values.astype(int)
    y_all_zero = (y_all - 1).astype(int)

    # Class weights
    class_weights = compute_class_weights(y_all)  # 1..6 weights
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    # Tokenization
    train_texts = train_df["full_text"].astype(str).tolist()
    test_texts = test_df["full_text"].astype(str).tolist()

    train_encodings = tokenize_texts(tokenizer, train_texts, max_length=max_len)
    test_encodings = tokenize_texts(tokenizer, test_texts, max_length=max_len)

    # Datasets
    train_dataset_full = EssayDataset(train_encodings, labels=y_all_zero.tolist())
    test_dataset = EssayDataset(test_encodings, labels=None)

    # Collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    # Folds
    folds = stratified_kfold_split(y_all, n_splits=num_folds, random_state=SEED)

    # OOF and test predictions
    oof_pred = np.zeros(len(train_df), dtype=np.int64)
    test_prob_sum = np.zeros((len(test_df), NUM_CLASSES), dtype=np.float32)

    start_time_all = time.time()
    for fold_idx, (trn_idx, val_idx) in enumerate(folds):
        logging.info(f"===== Fold {fold_idx + 1}/{num_folds} =====")
        # Re-instantiate fresh model per fold to avoid leakage
        _, _, model = try_load_tokenizer_and_model([model_name], num_labels=NUM_CLASSES)
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass
        try:
            if use_cuda and hasattr(torch, "compile"):
                model = torch.compile(model)
        except Exception:
            pass

        model.to(device)
        log_cuda_mem(prefix="After model.to:")

        # Subsets
        trn_subset = Subset(train_dataset_full, trn_idx.tolist())
        val_subset = Subset(train_dataset_full, val_idx.tolist())

        # Build loaders adaptively to avoid OOM
        train_loader, valid_loader, test_loader, eff_train_bs, eff_valid_bs = adaptive_build_loaders(
            trn_subset, val_subset, test_dataset, data_collator, use_cuda, train_bs, valid_bs, max_retries=3
        )
        logging.info(f"Effective batch sizes - train: {eff_train_bs}, valid/test: {eff_valid_bs}")

        # Optimizer and scheduler
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)

        total_steps = math.ceil(len(train_loader) / max(1, grad_accum_steps)) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        # Loss with class weights and mild label smoothing if available
        try:
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.05)
            logging.info("Using CrossEntropyLoss with label_smoothing=0.05")
        except TypeError:
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
            logging.info("Using CrossEntropyLoss without label smoothing (unsupported in this torch version)")

        # Train with early stopping
        best_state_dict, best_qwk = train_one_fold(
            model, device, train_loader, valid_loader, criterion, optimizer, scheduler,
            epochs=epochs, use_cuda=use_cuda, grad_accum_steps=grad_accum_steps, patience=2
        )

        # Load best state
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
            logging.info(f"Fold {fold_idx+1} | Loaded best model state with QWK: {best_qwk:.6f}")
        else:
            logging.info(f"Fold {fold_idx+1} | Warning: No best state found; using last epoch weights")

        # OOF preds
        model.eval()
        fold_val_preds = []
        fold_val_labels = []
        with torch.no_grad():
            for batch in valid_loader:
                labels_np = batch.pop("labels").cpu().numpy()
                for k in batch:
                    batch[k] = batch[k].to(device, non_blocking=True)
                outputs = model(**batch)
                probs = torch.softmax(outputs.logits, dim=-1).detach().cpu().numpy()
                preds = probs.argmax(axis=1)
                fold_val_preds.append(preds)
                fold_val_labels.append(labels_np)
        fold_val_preds = np.concatenate(fold_val_preds)
        fold_val_labels = np.concatenate(fold_val_labels)
        oof_pred[val_idx] = fold_val_preds
        fold_qwk = qwk_score(fold_val_labels + 1, fold_val_preds + 1)
        logging.info(f"Fold {fold_idx+1} | Final Fold QWK: {fold_qwk:.6f}")

        # Test predictions accumulation
        test_probs_fold = []
        with torch.no_grad():
            for batch in test_loader:
                for k in batch:
                    batch[k] = batch[k].to(device, non_blocking=True)
                outputs = model(**batch)
                probs = torch.softmax(outputs.logits, dim=-1).detach().cpu().numpy()
                test_probs_fold.append(probs)
        test_probs_fold = np.vstack(test_probs_fold)
        # Safety check for NaNs/Infs
        if not np.isfinite(test_probs_fold).all():
            logging.info("Non-finite values detected in test probabilities; replacing with uniform distribution")
            test_probs_fold = np.nan_to_num(test_probs_fold, nan=1.0 / NUM_CLASSES, posinf=1.0 / NUM_CLASSES, neginf=1.0 / NUM_CLASSES)
        test_prob_sum += test_probs_fold

        # Cleanup
        del model, optimizer, scheduler, criterion
        del train_loader, valid_loader, test_loader
        gc.collect()
        if use_cuda:
            torch.cuda.empty_cache()
            log_cuda_mem(prefix="After fold cleanup:")

    total_time = time.time() - start_time_all
    logging.info(f"Training complete for all folds in {total_time/60:.2f} minutes")

    # OOF QWK
    oof_qwk = qwk_score(y_all, (oof_pred + 1))
    logging.info(f"OOF QWK over all folds: {oof_qwk:.6f}")

    # Ensemble predictions
    test_probs_avg = test_prob_sum / num_folds
    test_pred_labels = test_probs_avg.argmax(axis=1) + 1  # back to 1..6
    return test_pred_labels


def fallback_tfidf(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    logging.info("Falling back to TF-IDF + Linear model (CPU) due to unavailable transformers or CUDA")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    X = train_df["full_text"].astype(str).values
    y = train_df["score"].values
    X_test = test_df["full_text"].astype(str).values

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=250000,
            ngram_range=(1, 2),
            min_df=2,
            dtype=np.float32,
        )),
        ("clf", LogisticRegression(
            multi_class="multinomial",
            max_iter=400,
            C=2.0,
            n_jobs=4,
            class_weight="balanced",
            solver="lbfgs",
        ))
    ])

    logging.info("Training TF-IDF model")
    pipe.fit(X, y)
    logging.info("Predicting on test data")
    preds = pipe.predict(X_test)
    return preds


def build_and_write_submission(test_df: pd.DataFrame, preds: np.ndarray):
    logging.info("Building submission DataFrame")
    sub = pd.DataFrame({
        "essay_id": test_df["essay_id"].values,
        "score": preds.astype(int),
    })
    # Ensure valid score range 1..6
    sub["score"] = sub["score"].clip(1, NUM_CLASSES)
    sub.to_csv(SUBMISSION_PATH, index=False)
    logging.info(f"Submission written to: {SUBMISSION_PATH}")


def main():
    try:
        ensure_output_dir()
        train_df, test_df = load_data()
        logging.info("Checking availability of transformers and CUDA")
        has_transformers = try_import_transformers()

        if has_transformers:
            try:
                device = get_device()
                use_cuda = device is not None and device.type == "cuda"
                preds_test = train_and_predict_transformer(
                    train_df=train_df,
                    test_df=test_df,
                    num_folds=5,
                    max_len=512,           # per plan
                    epochs=3,              # per plan
                    lr=2e-5,
                    weight_decay=0.01,
                    warmup_ratio=0.1,
                    grad_accum_steps=1,
                    train_bs=8 if use_cuda else 4,
                    valid_bs=16 if use_cuda else 8,
                )
                logging.info("Transformer pipeline finished successfully")
            except Exception as e:
                logging.info(f"Transformer pipeline failed due to: {e}\n{traceback.format_exc()}")
                preds_test = fallback_tfidf(train_df, test_df)
        else:
            preds_test = fallback_tfidf(train_df, test_df)

        build_and_write_submission(test_df, preds_test)
    except Exception as e:
        logging.info(f"Fatal error in main: {e}\n{traceback.format_exc()}")
        # Try to write a safe submission from sample file
        try:
            logging.info("Attempting to write safe fallback submission from sample_submission.csv")
            sample = pd.read_csv(SAMPLE_SUBMISSION_CSV)
            sample.to_csv(SUBMISSION_PATH, index=False)
            logging.info(f"Fallback submission saved to {SUBMISSION_PATH}")
        except Exception as e2:
            logging.info(f"Failed to write fallback submission: {e2}")


if __name__ == "__main__":
    main()