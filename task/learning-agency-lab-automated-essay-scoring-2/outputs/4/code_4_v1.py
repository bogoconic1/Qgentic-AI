# Filename: code_4_v1.py
import os
import sys
import gc
import math
import time
import random
import logging
import traceback
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
logging.info("Starting script code_4_v1.py")

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

def set_torch_seed(seed: int = SEED):
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
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

def compute_class_weights(y: np.ndarray) -> np.ndarray:
    logging.info("Computing class weights")
    # y are scores 1..6
    counts = pd.Series(y).value_counts().sort_index()
    max_count = counts.max()
    weights = np.array([max_count / counts.get(i, 1) for i in range(1, 7)], dtype=np.float32)
    logging.info(f"Class counts: {counts.to_dict()} | Weights: {weights.tolist()}")
    return weights

def qwk_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import cohen_kappa_score
    # both y_true and y_pred are 1..6
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

def load_data():
    logging.info("Loading data")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    logging.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    return train_df, test_df

def stratified_kfold_split(y: np.ndarray, n_splits: int = 5, random_state: int = SEED):
    logging.info(f"Generating Stratified {n_splits}-Fold indices")
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(skf.split(np.zeros_like(y), y))

def try_import_transformers():
    try:
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            DataCollatorWithPadding,
            get_linear_schedule_with_warmup,
        )
        logging.info("Transformers successfully imported")
        return True
    except Exception as e:
        logging.info(f"Could not import transformers or torch: {e}")
        return False

def load_tokenizer(name: str = "bert-base-uncased"):
    logging.info(f"Loading tokenizer: {name}")
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(name, use_fast=True, local_files_only=True)
        logging.info(f"Loaded tokenizer locally: {name}")
        return tok
    except Exception as e:
        logging.info(f"Failed to load tokenizer locally: {e}")
        raise

def load_model(name: str = "bert-base-uncased", num_labels: int = 6):
    logging.info(f"Loading model: {name} with {num_labels} labels")
    from transformers import AutoModelForSequenceClassification
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            name,
            num_labels=num_labels,
            problem_type="single_label_classification",
            local_files_only=True,
        )
        logging.info(f"Loaded model locally: {name}")
        return model
    except Exception as e:
        logging.info(f"Failed to load model locally: {e}")
        raise

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
    # Ensure lists of lists
    return {k: enc[k] for k in enc}

def train_and_predict_bert(
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
    model_name: str = "bert-base-uncased"
):
    logging.info("Starting BERT training and prediction pipeline")

    import torch
    from torch.utils.data import DataLoader, Subset
    from transformers import DataCollatorWithPadding, get_linear_schedule_with_warmup

    device = get_device()
    use_cuda = device is not None and device.type == "cuda"
    if device is None:
        raise RuntimeError("Torch device could not be initialized")

    set_torch_seed(SEED)

    # Label mapping: 1..6 -> 0..5
    y_all = train_df["score"].values
    y_all_zero = y_all - 1

    # Compute class weights
    class_weights = compute_class_weights(y_all)  # for 1..6
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    # Tokenizer
    tokenizer = load_tokenizer(model_name)

    # Pre-tokenize all train and test
    train_texts = train_df["full_text"].astype(str).tolist()
    test_texts = test_df["full_text"].astype(str).tolist()

    train_encodings = tokenize_texts(tokenizer, train_texts, max_length=max_len)
    test_encodings = tokenize_texts(tokenizer, test_texts, max_length=max_len)

    # Build datasets
    train_dataset_full = EssayDataset(train_encodings, labels=y_all_zero.tolist())
    test_dataset = EssayDataset(test_encodings, labels=None)

    # Collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    # Stratified folds
    folds = stratified_kfold_split(y_all, n_splits=num_folds, random_state=SEED)

    # Arrays to collect OOF and test probs
    oof_pred = np.zeros(len(train_df), dtype=np.int64)
    test_prob_sum = np.zeros((len(test_df), 6), dtype=np.float32)

    # Train folds
    start_time_all = time.time()
    for fold_idx, (trn_idx, val_idx) in enumerate(folds):
        logging.info(f"===== Fold {fold_idx+1}/{num_folds} =====")
        trn_subset = Subset(train_dataset_full, trn_idx.tolist())
        val_subset = Subset(train_dataset_full, val_idx.tolist())

        # Adaptive batch size in case of OOM
        _train_bs = train_bs
        _valid_bs = valid_bs

        train_loader = DataLoader(
            trn_subset,
            batch_size=_train_bs,
            shuffle=True,
            num_workers=2,
            pin_memory=use_cuda,
            collate_fn=data_collator,
        )
        valid_loader = DataLoader(
            val_subset,
            batch_size=_valid_bs,
            shuffle=False,
            num_workers=2,
            pin_memory=use_cuda,
            collate_fn=data_collator,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=_valid_bs,
            shuffle=False,
            num_workers=2,
            pin_memory=use_cuda,
            collate_fn=data_collator,
        )

        # Load model
        model = load_model(model_name, num_labels=6)
        model.to(device)

        # Optimizer and Scheduler
        no_decay = ["bias", "LayerNorm.weight"]
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

        # Loss
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)

        # Mixed precision
        scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

        best_qwk = -1.0
        best_state_dict = None
        epochs_no_improve = 0

        for epoch in range(epochs):
            logging.info(f"Fold {fold_idx+1} | Epoch {epoch+1}/{epochs} | Starting training")
            model.train()
            running_loss = 0.0
            optimizer.zero_grad(set_to_none=True)

            for step, batch in enumerate(train_loader):
                labels = batch.pop("labels").to(device)
                for k in batch:
                    batch[k] = batch[k].to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=use_cuda):
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
                    logging.info(f"Fold {fold_idx+1} | Epoch {epoch+1} | Step {step+1}/{len(train_loader)} | Loss: {running_loss / (step+1):.4f}")

            avg_train_loss = running_loss / max(1, len(train_loader))
            logging.info(f"Fold {fold_idx+1} | Epoch {epoch+1} | Training done | Avg Loss: {avg_train_loss:.4f}")

            # Validation
            model.eval()
            val_preds = []
            val_labels = []
            with torch.no_grad():
                for batch in valid_loader:
                    labels = batch.pop("labels").cpu().numpy()
                    for k in batch:
                        batch[k] = batch[k].to(device, non_blocking=True)
                    with torch.cuda.amp.autocast(enabled=use_cuda):
                        outputs = model(**batch)
                        logits = outputs.logits
                        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
                    preds = probs.argmax(axis=1)
                    val_preds.append(preds)
                    val_labels.append(labels)
            val_preds = np.concatenate(val_preds)
            val_labels = np.concatenate(val_labels)

            # Convert to 1..6 for QWK
            val_qwk = qwk_score(val_labels + 1, val_preds + 1)
            logging.info(f"Fold {fold_idx+1} | Epoch {epoch+1} | Validation QWK: {val_qwk:.6f}")

            # Early stopping
            if val_qwk > best_qwk:
                best_qwk = val_qwk
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
                logging.info(f"Fold {fold_idx+1} | Epoch {epoch+1} | New best QWK: {best_qwk:.6f}")
            else:
                epochs_no_improve += 1
                logging.info(f"Fold {fold_idx+1} | Epoch {epoch+1} | No improvement count: {epochs_no_improve}")
                if epochs_no_improve >= 2:
                    logging.info(f"Fold {fold_idx+1} | Early stopping triggered")
                    break

            # Free some memory
            del val_preds, val_labels
            gc.collect()
            if use_cuda:
                torch.cuda.empty_cache()

        # Load best state
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
            logging.info(f"Fold {fold_idx+1} | Loaded best model state with QWK: {best_qwk:.6f}")
        else:
            logging.info(f"Fold {fold_idx+1} | Warning: No best state found; using last epoch weights")

        # OOF predictions for this fold
        model.eval()
        fold_val_preds = []
        fold_val_labels = []
        with torch.no_grad():
            for batch in valid_loader:
                labels = batch.pop("labels").cpu().numpy()
                for k in batch:
                    batch[k] = batch[k].to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_cuda):
                    outputs = model(**batch)
                    probs = torch.softmax(outputs.logits, dim=-1).detach().cpu().numpy()
                preds = probs.argmax(axis=1)
                fold_val_preds.append(preds)
                fold_val_labels.append(labels)
        fold_val_preds = np.concatenate(fold_val_preds)
        fold_val_labels = np.concatenate(fold_val_labels)

        # Store OOF predictions (as labels 0..5)
        oof_pred[val_idx] = fold_val_preds
        fold_qwk = qwk_score(fold_val_labels + 1, fold_val_preds + 1)
        logging.info(f"Fold {fold_idx+1} | Final Fold QWK: {fold_qwk:.6f}")

        # Test predictions for this fold
        test_probs_fold = []
        with torch.no_grad():
            for batch in test_loader:
                for k in batch:
                    batch[k] = batch[k].to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_cuda):
                    outputs = model(**batch)
                    probs = torch.softmax(outputs.logits, dim=-1).detach().cpu().numpy()
                test_probs_fold.append(probs)
        test_probs_fold = np.vstack(test_probs_fold)
        test_prob_sum += test_probs_fold

        # Cleanup per fold
        del model, optimizer, scheduler, criterion, scaler
        del train_loader, valid_loader, test_loader
        gc.collect()
        if use_cuda:
            torch.cuda.empty_cache()

    total_time = time.time() - start_time_all
    logging.info(f"Training complete for all folds in {total_time/60:.2f} minutes")

    # OOF QWK
    oof_qwk = qwk_score(y_all, (oof_pred + 1))
    logging.info(f"OOF QWK over all folds: {oof_qwk:.6f}")

    # Average test probabilities across folds
    test_probs_avg = test_prob_sum / num_folds
    test_pred_labels = test_probs_avg.argmax(axis=1) + 1  # back to 1..6
    return test_pred_labels

def fallback_tfidf(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    logging.info("Falling back to TF-IDF + Linear model (CPU) due to unavailable transformers or CUDA")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import cohen_kappa_score

    X = train_df["full_text"].astype(str).values
    y = train_df["score"].values
    X_test = test_df["full_text"].astype(str).values

    # Pipeline
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=200000,
            ngram_range=(1, 2),
            min_df=2
        )),
        ("clf", LogisticRegression(
            multi_class="multinomial",
            max_iter=200,
            C=2.0,
            n_jobs=4,
            class_weight="balanced"
        ))
    ])

    # Simple training (no folds for speed)
    logging.info("Training TF-IDF model")
    pipe.fit(X, y)
    logging.info("Predicting on test data")
    preds = pipe.predict(X_test)
    return preds

def main():
    try:
        ensure_output_dir()
        train_df, test_df = load_data()
        logging.info("Checking availability of transformers and CUDA")
        has_transformers = try_import_transformers()

        if has_transformers:
            import torch
            device = get_device()
            use_cuda = device is not None and device.type == "cuda"

            try:
                # Try BERT path
                preds_test = train_and_predict_bert(
                    train_df=train_df,
                    test_df=test_df,
                    num_folds=5,
                    max_len=512,
                    epochs=3,
                    lr=2e-5,
                    weight_decay=0.01,
                    warmup_ratio=0.1,
                    grad_accum_steps=1,
                    train_bs=8 if use_cuda else 4,
                    valid_bs=16 if use_cuda else 8,
                    model_name="bert-base-uncased",
                )
                logging.info("BERT pipeline finished successfully")
            except Exception as e:
                logging.info(f"BERT pipeline failed due to: {e}\n{traceback.format_exc()}")
                preds_test = fallback_tfidf(train_df, test_df)
        else:
            preds_test = fallback_tfidf(train_df, test_df)

        # Build submission
        logging.info("Building submission DataFrame")
        submission = pd.DataFrame({
            "essay_id": test_df["essay_id"].values,
            "score": preds_test.astype(int)
        })
        # Ensure valid range 1..6
        submission["score"] = submission["score"].clip(1, 6)

        # Write to required path
        submission.to_csv(SUBMISSION_PATH, index=False)
        logging.info(f"Submission written to: {SUBMISSION_PATH}")
    except Exception as e:
        logging.info(f"Fatal error in main: {e}\n{traceback.format_exc()}")
        # Try to write a safe submission using sample file (default median score = 3)
        try:
            logging.info("Attempting to write safe fallback submission")
            sample = pd.read_csv(SAMPLE_SUBMISSION_CSV)
            sample.to_csv(SUBMISSION_PATH, index=False)
            logging.info(f"Fallback submission saved to {SUBMISSION_PATH}")
        except Exception as e2:
            logging.info(f"Failed to write fallback submission: {e2}")

if __name__ == "__main__":
    main()