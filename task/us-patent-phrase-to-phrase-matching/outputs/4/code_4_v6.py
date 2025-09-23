import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import math
import time
import random
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.isotonic import IsotonicRegression

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
    get_cosine_schedule_with_warmup,
)

# =============================================================================
# Paths, Config, Logging
# =============================================================================
BASE_DIR = "task/us-patent-phrase-to-phrase-matching"
OUT_DIR = os.path.join(BASE_DIR, "outputs", "4")
os.makedirs(OUT_DIR, exist_ok=True)
LOG_FILE = os.path.join(OUT_DIR, "code_4_v6.txt")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),
        logging.StreamHandler()
    ]
)
logging.info("Initialized logging system.")
logging.info(f"Logs will be written to: {LOG_FILE}")

@dataclass
class Config:
    # General
    seed: int = 42
    n_splits: int = 5
    num_workers: int = 4
    fp16: bool = True
    # Backbone
    model_name: str = "microsoft/deberta-v3-large"
    max_length: int = 256
    # Supervised finetune
    train_bs: int = 16
    valid_bs: int = 32
    epochs: int = 3
    lr: float = 3e-5
    weight_decay: float = 0.01
    optimizer_eps: float = 1e-8
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    dropout: float = 0.2
    n_msd: int = 5
    last_n_layers: int = 4
    layer_decay: float = 0.95
    gradient_checkpointing: bool = False
    # Multi-task loss weights
    lambda_reg: float = 0.5
    lambda_exp: float = 0.25
    lambda_cls: float = 0.25
    # R-Drop and Pearson regularizer
    rdrop_coef: float = 0.3
    lambda_np: float = 0.05
    # Inference blend (reg vs expected value)
    blend_alpha_init: float = 0.5
    # Adversarial (FGM)
    adv_epsilon: float = 1e-2
    adv_weight: float = 0.5
    adv_start_ratio: float = 0.1
    # EMA
    ema_decay: float = 0.999
    # DAPT (Domain-Adaptive PreTraining) for MLM
    dapt_enable: bool = True
    dapt_steps: int = 1200
    dapt_bs: int = 64
    dapt_lr: float = 5e-5
    dapt_wd: float = 0.01
    dapt_warmup_ratio: float = 0.06
    dapt_mask_prob: float = 0.15
    dapt_corpus_cap: int = 100000
    # Variants (two input templates, two seeds, ensemble)
    model_variants: int = 2
    variant_seeds: List[int] = None  # will set later
    # Files
    train_file: str = os.path.join(BASE_DIR, "train.csv")
    test_file: str = os.path.join(BASE_DIR, "test.csv")
    titles_file: str = os.path.join(BASE_DIR, "titles.csv")
    submission_file: str = os.path.join(OUT_DIR, "submission_6.csv")
    log_interval: int = 100

CFG = Config()
if CFG.variant_seeds is None:
    CFG.variant_seeds = [42, 2025]  # one per variant
logging.info(f"Config: {asdict(CFG)}")

# =============================================================================
# Reproducibility and CUDA
# =============================================================================
def seed_all(seed: int):
    logging.info(f"Seeding all RNGs with seed={seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

assert torch.cuda.is_available(), "CUDA must be available."
device = torch.device("cuda")
logging.info(f"Using device: {device}, CUDA device count: {torch.cuda.device_count()}")
logging.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")

seed_all(CFG.seed)

# =============================================================================
# Load data and enrich with CPC titles
# =============================================================================
logging.info("Loading CSV files.")
df_train = pd.read_csv(CFG.train_file)
df_test = pd.read_csv(CFG.test_file)
df_titles = pd.read_csv(CFG.titles_file)

logging.info(f"Train shape: {df_train.shape}, columns: {list(df_train.columns)}")
logging.info(f"Test shape: {df_test.shape}, columns: {list(df_test.columns)}")
logging.info(f"Titles shape: {df_titles.shape}, columns: {list(df_titles.columns)}")

expected_title_cols = ["code", "title", "section", "class", "subclass", "group", "subgroup"]
for col in expected_title_cols:
    if col not in df_titles.columns:
        logging.info(f"Column '{col}' not found in titles; creating empty column.")
        df_titles[col] = ""

logging.info("Merging CPC titles into train and test on context->code.")
df_train = df_train.merge(df_titles[expected_title_cols], left_on="context", right_on="code", how="left")
df_test = df_test.merge(df_titles[expected_title_cols], left_on="context", right_on="code", how="left")

for col in expected_title_cols:
    if col != "code":
        df_train[col] = df_train[col].fillna("")
        df_test[col] = df_test[col].fillna("")

logging.info(f"Post-merge train shape: {df_train.shape}, columns now: {list(df_train.columns)}")
logging.info(f"Post-merge test shape: {df_test.shape}, columns now: {list(df_test.columns)}")

# =============================================================================
# EDA checks for research plan
# =============================================================================
logging.info("Checking for train-test leakage (overlap of (anchor, target, context)).")
train_keys = df_train[["anchor", "target", "context"]].copy()
test_keys = df_test[["anchor", "target", "context"]].copy()
train_keys["triple"] = train_keys["anchor"].astype(str) + " | " + train_keys["target"].astype(str) + " | " + train_keys["context"].astype(str)
test_keys["triple"] = test_keys["anchor"].astype(str) + " | " + test_keys["target"].astype(str) + " | " + test_keys["context"].astype(str)
overlap = pd.Series(np.intersect1d(train_keys["triple"].values, test_keys["triple"].values))
logging.info(f"Number of overlapping (anchor, target, context) triplets: {len(overlap)}")

logging.info("Computing test context distribution (top 10).")
test_context_counts = df_test["context"].value_counts()
test_total = len(df_test)
for ctx, cnt in test_context_counts.head(10).items():
    pct = 100.0 * cnt / test_total
    logging.info(f"Test context {ctx}: count={cnt}, pct={pct:.2f}%")

logging.info("Score distribution per context for H04, G01, A61.")
for ctx in ["H04", "G01", "A61"]:
    sub = df_train[df_train["context"] == ctx]
    if len(sub) == 0:
        logging.info(f"Context {ctx}: no rows in train.")
    else:
        counts = sub["score"].value_counts().sort_index()
        logging.info(f"Context {ctx}: total={len(sub)}")
        for score_val, cnt in counts.items():
            logging.info(f"  score={score_val}: count={cnt}")

max_target_len_test = df_test["target"].astype(str).str.len().max()
logging.info(f"Maximum target character length in test.csv: {max_target_len_test}")

logging.info("Checking mapping coverage of contexts to titles codes.")
train_contexts = set(df_train["context"].unique().tolist())
test_contexts = set(df_test["context"].unique().tolist())
title_codes = set(df_titles["code"].astype(str).unique().tolist())
train_covered = len(train_contexts - title_codes) == 0
test_covered = len(test_contexts - title_codes) == 0
logging.info(f"All train contexts covered by titles: {train_covered}")
logging.info(f"All test contexts covered by titles: {test_covered}")

# =============================================================================
# Build two input variants (templates)
# Variant 0: rich CPC fields + anchor vs target (as v5)
# Variant 1: concise CPC title prompt appended after pair
# =============================================================================
def build_pair_variant0(row: pd.Series) -> Tuple[str, str]:
    cpc_parts = [
        f"code: {str(row['context'])}",
        f"title: {str(row['title'])}" if 'title' in row and pd.notna(row['title']) else "title: ",
        f"section: {str(row['section'])}" if 'section' in row and pd.notna(row['section']) else "section: ",
        f"class: {str(row['class'])}" if 'class' in row and pd.notna(row['class']) else "class: ",
        f"subclass: {str(row['subclass'])}" if 'subclass' in row and pd.notna(row['subclass']) else "subclass: ",
    ]
    cpc_str = " | ".join(cpc_parts)
    first_seq = f"context: {cpc_str} [CTX] anchor: {str(row['anchor'])}"
    second_seq = f"target: {str(row['target'])}"
    return first_seq, second_seq

def build_pair_variant1(row: pd.Series) -> Tuple[str, str]:
    # concise CPC prompt at the end, encourages focus on pair with light domain cue
    first_seq = f"anchor: {str(row['anchor'])}"
    title = str(row['title']) if 'title' in row and pd.notna(row['title']) else ""
    second_seq = f"target: {str(row['target'])} [DOMAIN {row['context']}: {title}]"
    return first_seq, second_seq

logging.info("Building sentence pairs for both variants.")
pairs0_train = df_train.apply(build_pair_variant0, axis=1)
pairs0_test = df_test.apply(build_pair_variant0, axis=1)
pairs1_train = df_train.apply(build_pair_variant1, axis=1)
pairs1_test = df_test.apply(build_pair_variant1, axis=1)

df_train_var0 = df_train.copy()
df_test_var0 = df_test.copy()
df_train_var1 = df_train.copy()
df_test_var1 = df_test.copy()

df_train_var0["first_seq"] = pairs0_train.map(lambda x: x[0])
df_train_var0["second_seq"] = pairs0_train.map(lambda x: x[1])
df_test_var0["first_seq"] = pairs0_test.map(lambda x: x[0])
df_test_var0["second_seq"] = pairs0_test.map(lambda x: x[1])

df_train_var1["first_seq"] = pairs1_train.map(lambda x: x[0])
df_train_var1["second_seq"] = pairs1_train.map(lambda x: x[1])
df_test_var1["first_seq"] = pairs1_test.map(lambda x: x[0])
df_test_var1["second_seq"] = pairs1_test.map(lambda x: x[1])

logging.info("Sample Variant 0 pairs:")
for i in range(min(2, len(df_train_var0))):
    logging.info(f"V0 FIRST: {df_train_var0['first_seq'].iloc[i]}")
    logging.info(f"V0 SECOND: {df_train_var0['second_seq'].iloc[i]}")
logging.info("Sample Variant 1 pairs:")
for i in range(min(2, len(df_train_var1))):
    logging.info(f"V1 FIRST: {df_train_var1['first_seq'].iloc[i]}")
    logging.info(f"V1 SECOND: {df_train_var1['second_seq'].iloc[i]}")

# =============================================================================
# Stratified 5-fold split (common across variants)
# =============================================================================
def make_bins(values: np.ndarray, n_bins: int = 5) -> np.ndarray:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    inds = np.digitize(values, bins, right=True)
    inds = np.clip(inds, 1, n_bins) - 1
    return inds

logging.info("Preparing stratified K-Fold splits.")
y_all = df_train["score"].astype(float).values
bins = make_bins(y_all, n_bins=5)
skf = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)
folds = list(skf.split(np.zeros(len(df_train)), bins))
logging.info(f"Total folds: {CFG.n_splits}")

# =============================================================================
# Tokenizer
# =============================================================================
logging.info("Loading tokenizer.")
tokenizer = AutoTokenizer.from_pretrained(CFG.model_name, use_fast=True)
tokenizer.model_max_length = CFG.max_length

# =============================================================================
# DAPT: Domain-Adaptive Pretraining (MLM)
# =============================================================================
class SimpleTextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer: AutoTokenizer, max_length: int):
        self.texts = texts
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        enc = self.tok(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True
        )
        return enc

def build_dapt_corpus(df_train_var0: pd.DataFrame, df_test_var0: pd.DataFrame, df_titles: pd.DataFrame, cap: int) -> List[str]:
    logging.info("Building DAPT corpus from CPC titles and pair templates.")
    title_texts = []
    code_col = df_titles["code"].astype(str).tolist()
    title_col = df_titles["title"].astype(str).tolist()
    section_col = df_titles["section"].astype(str).tolist()
    class_col = df_titles["class"].astype(str).tolist()
    subclass_col = df_titles["subclass"].astype(str).tolist()
    for c, t, s, cl, sc in zip(code_col, title_col, section_col, class_col, subclass_col):
        piece = f"CPC {c}: {t} | section: {s} | class: {cl} | subclass: {sc}"
        title_texts.append(piece)

    pair_texts_train = (df_train_var0["first_seq"] + " [SEP] " + df_train_var0["second_seq"]).astype(str).tolist()
    pair_texts_test = (df_test_var0["first_seq"] + " [SEP] " + df_test_var0["second_seq"]).astype(str).tolist()

    corpus = title_texts + pair_texts_train + pair_texts_test
    random.shuffle(corpus)
    if len(corpus) > cap:
        corpus = corpus[:cap]
    logging.info(f"DAPT corpus size: {len(corpus)}")
    return corpus

dapt_state_dict = None
if CFG.dapt_enable:
    logging.info("Starting Domain-Adaptive Pretraining (MLM).")
    seed_all(CFG.seed + 10)
    corpus = build_dapt_corpus(df_train_var0, df_test_var0, df_titles, CFG.dapt_corpus_cap)
    dapt_dataset = SimpleTextDataset(corpus, tokenizer, CFG.max_length)
    mlm_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=CFG.dapt_mask_prob)
    dapt_loader = DataLoader(
        dapt_dataset,
        batch_size=CFG.dapt_bs,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=mlm_collator
    )
    mlm_model = AutoModelForMaskedLM.from_pretrained(CFG.model_name).to(device)
    mlm_optimizer = torch.optim.AdamW(mlm_model.parameters(), lr=CFG.dapt_lr, weight_decay=CFG.dapt_wd, eps=CFG.optimizer_eps)
    total_steps = CFG.dapt_steps
    warmup_steps = int(CFG.dapt_warmup_ratio * total_steps)
    mlm_scheduler = get_cosine_schedule_with_warmup(mlm_optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = torch.amp.GradScaler('cuda', enabled=CFG.fp16)

    step = 0
    iter_loader = iter(dapt_loader)
    t0 = time.time()
    while step < total_steps:
        try:
            batch = next(iter_loader)
        except StopIteration:
            iter_loader = iter(dapt_loader)
            batch = next(iter_loader)
        for k in batch:
            batch[k] = batch[k].to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=CFG.fp16):
            outputs = mlm_model(**batch)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.unscale_(mlm_optimizer)
        torch.nn.utils.clip_grad_norm_(mlm_model.parameters(), CFG.max_grad_norm)
        scaler.step(mlm_optimizer)
        scaler.update()
        mlm_optimizer.zero_grad(set_to_none=True)
        mlm_scheduler.step()

        step += 1
        if step % 50 == 0:
            logging.info(f"DAPT step {step}/{total_steps} | MLM loss: {loss.item():.6f} | LR: {mlm_scheduler.get_last_lr()[0]:.8f}")

    elapsed = time.time() - t0
    logging.info(f"DAPT completed in {elapsed/60.0:.2f} min. Extracting backbone state dict.")
    # Save backbone state dict to CPU for later loading into supervised models
    base = mlm_model.base_model
    dapt_state_dict = {k: v.detach().cpu() for k, v in base.state_dict().items()}
    # Free MLM model
    del mlm_model
    del mlm_optimizer
    del mlm_scheduler
    torch.cuda.empty_cache()
else:
    logging.info("DAPT is disabled by config.")
    dapt_state_dict = None

# =============================================================================
# Dataset and DataLoader for supervised finetuning
# =============================================================================
class PatentPairDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: AutoTokenizer, max_length: int, with_labels: bool = True, swap: bool = False):
        if swap:
            self.first_seq = df["second_seq"].astype(str).tolist()
            self.second_seq = df["first_seq"].astype(str).tolist()
        else:
            self.first_seq = df["first_seq"].astype(str).tolist()
            self.second_seq = df["second_seq"].astype(str).tolist()
        self.with_labels = with_labels
        self.labels = df["score"].astype(np.float32).to_numpy() if with_labels and "score" in df.columns else None
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.first_seq)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.first_seq[idx],
            self.second_seq[idx],
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True
        )
        if self.with_labels:
            enc["labels"] = float(self.labels[idx])
        return enc

# =============================================================================
# Model with Weighted Layer Pooling + Mean Pooling + Multi-Sample Dropout
# Multi-task: regression head + 5-class classification head
# =============================================================================
class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers: int, last_n_layers: int = 4):
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        self.last_n_layers = last_n_layers
        self.layer_weights = nn.Parameter(torch.ones(last_n_layers, dtype=torch.float32))

    def forward(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        selected = hidden_states[-self.last_n_layers:]  # list of tensors [B, T, H]
        stack = torch.stack(selected, dim=0)  # [L, B, T, H]
        norm_w = torch.softmax(self.layer_weights, dim=0).view(self.last_n_layers, 1, 1, 1)
        weighted = (norm_w * stack).sum(dim=0)  # [B, T, H]
        return weighted

def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    input_mask_expanded = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)  # [B, T, 1]
    sum_embeddings = (token_embeddings * input_mask_expanded).sum(dim=1)  # [B, H]
    sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-6)  # [B, 1]
    return sum_embeddings / sum_mask

class CrossEncoderMTL(nn.Module):
    def __init__(self, model_name: str, dropout: float, last_n_layers: int, enable_checkpointing: bool):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        if hasattr(self.config, "use_cache"):
            self.config.use_cache = False
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config)
        if enable_checkpointing and hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()
        self.pooler = WeightedLayerPooling(num_hidden_layers=self.config.num_hidden_layers, last_n_layers=last_n_layers)
        self.hidden_size = self.config.hidden_size
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(CFG.n_msd)])
        self.regressor = nn.Linear(self.hidden_size, 1)
        self.classifier = nn.Linear(self.hidden_size, 5)

    def load_backbone_state(self, state_dict: Dict[str, torch.Tensor]):
        logging.info("Loading DAPT backbone weights into cross-encoder backbone.")
        self.backbone.load_state_dict(state_dict, strict=True)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_layers = self.pooler(outputs.hidden_states)  # [B, T, H]
        features = mean_pooling(pooled_layers, attention_mask)  # [B, H]

        reg_logits_accum = 0.0
        cls_logits_accum = 0.0
        for d in self.dropouts:
            x = d(features)
            reg_logits_accum = reg_logits_accum + self.regressor(x)  # [B, 1]
            cls_logits_accum = cls_logits_accum + self.classifier(x)  # [B, 5]
        reg_logits = reg_logits_accum / len(self.dropouts)  # [B, 1]
        cls_logits = cls_logits_accum / len(self.dropouts)  # [B, 5]
        return {"regression": reg_logits.squeeze(-1), "class_logits": cls_logits}

# =============================================================================
# Optimizer groups with Layer-wise LR Decay
# =============================================================================
def get_layer_id_for_deberta(name: str, num_layers: int) -> int:
    if name.startswith("backbone.embeddings"):
        return 0
    if name.startswith("backbone.encoder.layer."):
        parts = name.split(".")
        idx = parts.index("layer") + 1
        layer_num = int(parts[idx])
        return layer_num + 1
    return num_layers + 1

def get_optimizer_grouped_parameters(model: nn.Module, base_lr: float, weight_decay: float, layer_decay: float) -> List[Dict]:
    logging.info("Creating optimizer parameter groups with layer-wise LR decay.")
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "layernorm.weight"]
    num_layers = model.config.num_hidden_layers
    param_groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        layer_id = get_layer_id_for_deberta(name, num_layers)
        decay_flag = not any(nd in name for nd in no_decay)
        group_key = (layer_id, decay_flag)
        if group_key not in param_groups:
            lr = base_lr * (layer_decay ** (num_layers + 1 - layer_id))
            wd = weight_decay if decay_flag else 0.0
            param_groups[group_key] = {"params": [], "lr": lr, "weight_decay": wd}
        param_groups[group_key]["params"].append(param)
    groups = list(param_groups.values())
    for i, g in enumerate(groups):
        logging.info(f"Group {i}: params={len(g['params'])}, lr={g['lr']:.8f}, wd={g['weight_decay']}")
    return groups

# =============================================================================
# EMA and FGM
# =============================================================================
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._registered = False
        self._param_names = []

    def register(self, model: nn.Module):
        logging.info("Registering EMA shadow parameters.")
        self.shadow = {}
        self._param_names = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                self._param_names.append(name)
        self._registered = True

    def update(self, model: nn.Module):
        if not self._registered:
            self.register(model)
        for name, param in model.named_parameters():
            if name in self.shadow:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model: nn.Module):
        logging.info("Applying EMA weights for evaluation.")
        self.backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        logging.info("Restoring original weights after EMA evaluation.")
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

class FGM:
    def __init__(self, model: nn.Module, emb_name: str = "word_embeddings", epsilon: float = 1e-2):
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name and param.grad is not None:
                self.backup[name] = param.data.clone()
                grad = param.grad
                norm = torch.norm(grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

# =============================================================================
# Loss utils
# =============================================================================
mse_loss = nn.MSELoss(reduction="mean")
kldiv_loss = nn.KLDivLoss(reduction="batchmean")

CLASS_BINS_CPU = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)

def compute_pearsonr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])

def labels_to_soft_targets(labels: torch.Tensor, class_bins: torch.Tensor) -> torch.Tensor:
    diffs = torch.abs(labels.unsqueeze(1) - class_bins.unsqueeze(0))  # [B, 5]
    weights = torch.clamp(1.0 - (diffs / 0.25), min=0.0)
    norm = torch.sum(weights, dim=1, keepdim=True).clamp(min=1e-6)
    probs = weights / norm
    return probs

def pearson_loss_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_c = pred - pred.mean()
    target_c = target - target.mean()
    cov = (pred_c * target_c).mean()
    pred_std = pred_c.pow(2).mean().sqrt()
    target_std = target_c.pow(2).mean().sqrt()
    rho = cov / (pred_std * target_std + 1e-6)
    return 1.0 - rho

# =============================================================================
# Train/eval helpers
# =============================================================================
def to_device(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out

def run_valid(model: nn.Module, loader: DataLoader, class_bins: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    logging.info("Running validation inference (EMA-applied).")
    model.eval()
    preds_reg = []
    preds_exp = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch)
            labels = batch.pop("labels").float()
            with torch.amp.autocast('cuda', enabled=CFG.fp16):
                outputs = model(**batch)
                pred_reg = outputs["regression"]
                logits_cls = outputs["class_logits"]
                probs_cls = torch.softmax(logits_cls, dim=-1)
                expected_val = torch.sum(probs_cls * class_bins, dim=-1)
            preds_reg.append(pred_reg.detach().float().cpu().numpy())
            preds_exp.append(expected_val.detach().float().cpu().numpy())
            targets.append(labels.detach().float().cpu().numpy())
    return np.concatenate(preds_reg), np.concatenate(preds_exp), np.concatenate(targets)

def run_test(model: nn.Module, loader: DataLoader, loader_swapped: DataLoader, class_bins: torch.Tensor, alpha_blend: float) -> np.ndarray:
    logging.info("Running test inference (TTA with swapped pairs).")
    def infer(dl: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        model.eval()
        pr, pe = [], []
        with torch.no_grad():
            for batch in dl:
                batch = to_device(batch)
                with torch.amp.autocast('cuda', enabled=CFG.fp16):
                    outputs = model(**batch)
                    pred_reg = outputs["regression"]
                    logits_cls = outputs["class_logits"]
                    probs_cls = torch.softmax(logits_cls, dim=-1)
                    expected_val = torch.sum(probs_cls * class_bins, dim=-1)
                pr.append(pred_reg.detach().float().cpu().numpy())
                pe.append(expected_val.detach().float().cpu().numpy())
        return np.concatenate(pr), np.concatenate(pe)

    r1, e1 = infer(loader)
    r2, e2 = infer(loader_swapped)
    blend1 = alpha_blend * r1 + (1.0 - alpha_blend) * e1
    blend2 = alpha_blend * r2 + (1.0 - alpha_blend) * e2
    final = 0.5 * (blend1 + blend2)
    final = np.clip(final, 0.0, 1.0)
    return final

# =============================================================================
# Supervised CV training for a given variant dataset and seed
# =============================================================================
def train_variant(df_train_variant: pd.DataFrame, df_test_variant: pd.DataFrame, seed: int) -> Dict[str, np.ndarray]:
    logging.info(f"Starting supervised training for variant with seed={seed}.")
    seed_all(seed)

    # Pre-build test datasets and loaders
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    test_dataset = PatentPairDataset(df_test_variant, tokenizer, CFG.max_length, with_labels=False, swap=False)
    test_dataset_swapped = PatentPairDataset(df_test_variant, tokenizer, CFG.max_length, with_labels=False, swap=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.valid_bs,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collator
    )
    test_loader_swapped = DataLoader(
        test_dataset_swapped,
        batch_size=CFG.valid_bs,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collator
    )
    logging.info(f"Test batches (normal/swapped): {len(test_loader)}/{len(test_loader_swapped)}")

    class_bins_torch = torch.tensor(CLASS_BINS_CPU, dtype=torch.float32, device=device)

    # Storage
    oof_reg = np.zeros(len(df_train_variant), dtype=np.float32)
    oof_exp = np.zeros(len(df_train_variant), dtype=np.float32)
    oof_targets = df_train_variant["score"].astype(np.float32).values
    test_preds_folds_reg = []
    test_preds_folds_exp = []
    fold_alphas = []

    for fold_id, (train_idx, valid_idx) in enumerate(folds):
        logging.info(f"===== Fold {fold_id+1}/{CFG.n_splits} (seed={seed}) =====")
        fold_train = df_train_variant.iloc[train_idx].reset_index(drop=True)
        fold_valid = df_train_variant.iloc[valid_idx].reset_index(drop=True)

        train_dataset = PatentPairDataset(fold_train, tokenizer, CFG.max_length, with_labels=True, swap=False)
        valid_dataset = PatentPairDataset(fold_valid, tokenizer, CFG.max_length, with_labels=True, swap=False)

        train_loader = DataLoader(
            train_dataset,
            batch_size=CFG.train_bs,
            shuffle=True,
            num_workers=CFG.num_workers,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=collator
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=CFG.valid_bs,
            shuffle=False,
            num_workers=CFG.num_workers,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=collator
        )
        logging.info(f"Fold {fold_id} -> Train batches: {len(train_loader)}, Valid batches: {len(valid_loader)}")

        model = CrossEncoderMTL(
            CFG.model_name,
            dropout=CFG.dropout,
            last_n_layers=CFG.last_n_layers,
            enable_checkpointing=CFG.gradient_checkpointing
        ).to(device)

        if dapt_state_dict is not None:
            model.load_backbone_state(dapt_state_dict)

        optimizer = torch.optim.AdamW(
            get_optimizer_grouped_parameters(model, CFG.lr, CFG.weight_decay, CFG.layer_decay),
            lr=CFG.lr,
            eps=CFG.optimizer_eps,
        )
        num_update_steps_per_epoch = len(train_loader)
        num_training_steps = CFG.epochs * num_update_steps_per_epoch
        num_warmup_steps = int(CFG.warmup_ratio * num_training_steps)
        adv_start_step = int(CFG.adv_start_ratio * num_training_steps)
        logging.info(f"Fold {fold_id} -> Training steps: {num_training_steps}, Warmup steps: {num_warmup_steps}, Adv start step: {adv_start_step}")

        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        scaler = torch.amp.GradScaler('cuda', enabled=CFG.fp16)
        ema = EMA(model, decay=CFG.ema_decay)
        ema.register(model)
        fgm = FGM(model, emb_name="word_embeddings", epsilon=CFG.adv_epsilon)

        best_val_pearson = -1.0
        best_state_dict = None
        global_step = 0

        for epoch in range(CFG.epochs):
            logging.info(f"Starting epoch {epoch+1}/{CFG.epochs} for fold {fold_id}.")
            model.train()
            epoch_loss = 0.0
            t0 = time.time()

            for step, batch in enumerate(train_loader):
                batch = to_device(batch)
                labels = batch.pop("labels").float()
                soft_targets = labels_to_soft_targets(labels, class_bins_torch)

                with torch.amp.autocast('cuda', enabled=CFG.fp16):
                    # R-Drop: two stochastic forwards
                    out1 = model(**batch)
                    out2 = model(**batch)

                    pred1_reg = out1["regression"]
                    logits1 = out1["class_logits"]
                    probs1 = torch.softmax(logits1, dim=-1)
                    exp1 = torch.sum(probs1 * class_bins_torch, dim=-1)

                    pred2_reg = out2["regression"]
                    logits2 = out2["class_logits"]
                    probs2 = torch.softmax(logits2, dim=-1)
                    exp2 = torch.sum(probs2 * class_bins_torch, dim=-1)

                    # main losses (both passes)
                    loss1_reg = mse_loss(pred1_reg, labels)
                    loss1_exp = mse_loss(exp1, labels)
                    loss1_cls = kldiv_loss(torch.log_softmax(logits1, dim=-1), soft_targets)

                    loss2_reg = mse_loss(pred2_reg, labels)
                    loss2_exp = mse_loss(exp2, labels)
                    loss2_cls = kldiv_loss(torch.log_softmax(logits2, dim=-1), soft_targets)

                    main1 = CFG.lambda_reg * loss1_reg + CFG.lambda_exp * loss1_exp + CFG.lambda_cls * loss1_cls
                    main2 = CFG.lambda_reg * loss2_reg + CFG.lambda_exp * loss2_exp + CFG.lambda_cls * loss2_cls

                    # R-Drop consistency
                    kl12 = kldiv_loss(torch.log_softmax(logits1, dim=-1), torch.softmax(logits2, dim=-1))
                    kl21 = kldiv_loss(torch.log_softmax(logits2, dim=-1), torch.softmax(logits1, dim=-1))
                    reg_cons = mse_loss(pred1_reg, pred2_reg)

                    blended_pred = CFG.blend_alpha_init * (0.5 * (pred1_reg + pred2_reg)) + (1.0 - CFG.blend_alpha_init) * (0.5 * (exp1 + exp2))
                    np_loss = pearson_loss_torch(blended_pred, labels)

                    total_loss = 0.5 * (main1 + main2) + CFG.rdrop_coef * (0.5 * (kl12 + kl21) + reg_cons) + CFG.lambda_np * np_loss

                scaler.scale(total_loss).backward()

                use_adv = (global_step >= adv_start_step)
                if use_adv:
                    fgm.attack()
                    with torch.amp.autocast('cuda', enabled=CFG.fp16):
                        out_adv = model(**batch)
                        pred_adv_reg = out_adv["regression"]
                        logits_adv = out_adv["class_logits"]
                        probs_adv = torch.softmax(logits_adv, dim=-1)
                        exp_adv = torch.sum(probs_adv * class_bins_torch, dim=-1)

                        adv_reg = mse_loss(pred_adv_reg, labels)
                        adv_exp = mse_loss(exp_adv, labels)
                        adv_cls = kldiv_loss(torch.log_softmax(logits_adv, dim=-1), soft_targets)
                        adv_loss = CFG.lambda_reg * adv_reg + CFG.lambda_exp * adv_exp + CFG.lambda_cls * adv_cls
                        adv_loss = CFG.adv_weight * adv_loss
                    scaler.scale(adv_loss).backward()
                    fgm.restore()

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                ema.update(model)
                global_step += 1

                epoch_loss += total_loss.item()

                if (step + 1) % CFG.log_interval == 0:
                    logging.info(
                        f"Fold {fold_id} | Epoch {epoch+1}/{CFG.epochs} | Step {step+1}/{len(train_loader)} | "
                        f"Loss: {total_loss.item():.6f} | "
                        f"LR: {scheduler.get_last_lr()[0]:.8f} | "
                        f"Adv:{use_adv}"
                    )

            avg_train_loss = epoch_loss / len(train_loader)
            elapsed = time.time() - t0
            logging.info(f"Fold {fold_id} | Epoch {epoch+1} done. Avg train loss: {avg_train_loss:.6f} | Time: {elapsed:.2f}s")

            # Validation with EMA
            ema.apply_shadow(model)
            v_reg, v_exp, v_tgt = run_valid(model, valid_loader, class_bins_torch)
            ema.restore(model)

            # use initial blend for model selection
            v_blend = CFG.blend_alpha_init * v_reg + (1.0 - CFG.blend_alpha_init) * v_exp
            v_mse_reg = float(np.mean((v_reg - v_tgt) ** 2))
            v_mse_exp = float(np.mean((v_exp - v_tgt) ** 2))
            v_mse_blend = float(np.mean((v_blend - v_tgt) ** 2))
            v_pear_reg = compute_pearsonr(v_tgt, v_reg)
            v_pear_exp = compute_pearsonr(v_tgt, v_exp)
            v_pear_blend = compute_pearsonr(v_tgt, v_blend)
            logging.info(
                f"Fold {fold_id} | Validation epoch {epoch+1}: "
                f"MSE(reg)={v_mse_reg:.6f}, MSE(exp)={v_mse_exp:.6f}, MSE(blend)={v_mse_blend:.6f} | "
                f"Pearson(reg)={v_pear_reg:.6f}, Pearson(exp)={v_pear_exp:.6f}, Pearson(blend)={v_pear_blend:.6f}"
            )

            if v_pear_blend > best_val_pearson:
                best_val_pearson = v_pear_blend
                logging.info(f"Fold {fold_id} | New best validation Pearson (blend): {best_val_pearson:.6f} at epoch {epoch+1}")
                # Save EMA weights as best
                ema.apply_shadow(model)
                best_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                ema.restore(model)

        # Load best
        model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})
        # OOF predictions for this fold (store separate reg/exp to tune alpha later)
        logging.info(f"Fold {fold_id} | Generating OOF predictions with best EMA weights.")
        v_reg, v_exp, v_tgt = run_valid(model, valid_loader, class_bins_torch)
        oof_reg[valid_idx] = np.clip(v_reg, 0.0, 1.0)
        oof_exp[valid_idx] = np.clip(v_exp, 0.0, 1.0)
        fold_oof_pear_reg = compute_pearsonr(v_tgt, v_reg)
        fold_oof_pear_exp = compute_pearsonr(v_tgt, v_exp)
        logging.info(f"Fold {fold_id} | OOF Pearson reg={fold_oof_pear_reg:.6f}, exp={fold_oof_pear_exp:.6f}")

        # For test, we will run later after alpha tuning per model; store both reg/exp predictions per fold
        # Use initial alpha for now to log
        alpha_tmp = CFG.blend_alpha_init
        fold_alphas.append(alpha_tmp)
        test_reg = run_test(model, test_loader, test_loader_swapped, class_bins_torch, alpha_blend=1.0)  # pure reg
        test_exp = run_test(model, test_loader, test_loader_swapped, class_bins_torch, alpha_blend=0.0)  # pure exp
        test_preds_folds_reg.append(test_reg)
        test_preds_folds_exp.append(test_exp)

        # Free model gpu weights for next fold
        del model
        torch.cuda.empty_cache()

    return {
        "oof_reg": oof_reg,
        "oof_exp": oof_exp,
        "oof_targets": oof_targets,
        "test_folds_reg": np.vstack(test_preds_folds_reg),  # [folds, N_test]
        "test_folds_exp": np.vstack(test_preds_folds_exp),  # [folds, N_test]
    }

# =============================================================================
# Train both variants
# =============================================================================
results_variants = []
for vidx in range(CFG.model_variants):
    if vidx == 0:
        df_train_variant = df_train_var0
        df_test_variant = df_test_var0
    else:
        df_train_variant = df_train_var1
        df_test_variant = df_test_var1
    results = train_variant(df_train_variant, df_test_variant, seed=CFG.variant_seeds[vidx])
    results_variants.append(results)

# =============================================================================
# Post-processing: Tune alpha per model, isotonic calibration per fold, ensemble weight search
# =============================================================================
logging.info("Post-processing: tuning internal blend alpha per model using OOF.")
alphas = []
calibrated_oof_models = []
calibrated_test_models = []

for vidx, res in enumerate(results_variants):
    oof_reg = res["oof_reg"]
    oof_exp = res["oof_exp"]
    y = res["oof_targets"]

    # Grid search alpha in [0,1] maximize Pearson on OOF
    best_alpha = CFG.blend_alpha_init
    best_pear = -1.0
    for a in np.linspace(0.0, 1.0, 101):
        oof_blend = a * oof_reg + (1.0 - a) * oof_exp
        pear = compute_pearsonr(y, oof_blend)
        if pear > best_pear:
            best_pear = pear
            best_alpha = a
    alphas.append(best_alpha)
    logging.info(f"Variant {vidx}: best alpha={best_alpha:.3f} with OOF Pearson={best_pear:.6f}")

    # Per-fold isotonic calibration: fit on each fold val preds, apply to that fold's test preds
    logging.info(f"Variant {vidx}: applying per-fold isotonic calibration.")
    fold_calibrators = []
    calibrated_oof = np.zeros_like(y, dtype=np.float32)
    test_calibrated_folds = []

    test_folds_reg = res["test_folds_reg"]  # [folds, N_test]
    test_folds_exp = res["test_folds_exp"]  # [folds, N_test]

    for fold_id, (train_idx, valid_idx) in enumerate(folds):
        # OOF fold predictions (blended)
        val_pred_reg = oof_reg[valid_idx]
        val_pred_exp = oof_exp[valid_idx]
        val_blend = best_alpha * val_pred_reg + (1.0 - best_alpha) * val_pred_exp
        val_true = y[valid_idx]

        # Fit isotonic on this fold
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
        iso.fit(val_blend, val_true)
        fold_calibrators.append(iso)

        # Calibrated OOF for this fold
        calibrated_oof[valid_idx] = iso.transform(val_blend).astype(np.float32)

        # Apply to this fold's test preds (blended for this model and fold)
        test_fold_reg = test_folds_reg[fold_id]
        test_fold_exp = test_folds_exp[fold_id]
        test_fold_blend = best_alpha * test_fold_reg + (1.0 - best_alpha) * test_fold_exp
        test_fold_calibrated = iso.transform(test_fold_blend).astype(np.float32)
        test_calibrated_folds.append(test_fold_calibrated)

    calibrated_oof_pearson = compute_pearsonr(y, calibrated_oof)
    logging.info(f"Variant {vidx}: Calibrated OOF Pearson={calibrated_oof_pearson:.6f}")

    # Average calibrated test across folds for this variant
    test_variant_mean = np.mean(np.vstack(test_calibrated_folds), axis=0)
    calibrated_oof_models.append(calibrated_oof)
    calibrated_test_models.append(test_variant_mean)

# Ensemble two models with non-negative blend weight search on OOF
logging.info("Ensembling variants with non-negative weight search on OOF.")
m1_oof = calibrated_oof_models[0]
m2_oof = calibrated_oof_models[1]
y_all = results_variants[0]["oof_targets"]  # same across variants

best_w = 0.5
best_pear = -1.0
for w in np.linspace(0.0, 1.0, 101):
    mix = w * m1_oof + (1.0 - w) * m2_oof
    pear = compute_pearsonr(y_all, mix)
    if pear > best_pear:
        best_pear = pear
        best_w = w
logging.info(f"Ensemble weight w (model0)={best_w:.3f}, (model1)={1.0-best_w:.3f} with OOF Pearson={best_pear:.6f}")

# Final test predictions
test_final = best_w * calibrated_test_models[0] + (1.0 - best_w) * calibrated_test_models[1]
test_final = np.clip(test_final, 0.0, 1.0)

# Also log per-variant OOF and overall OOF before saving
for vidx, coof in enumerate(calibrated_oof_models):
    pear = compute_pearsonr(y_all, coof)
    logging.info(f"Variant {vidx}: Final calibrated OOF Pearson={pear:.6f}")
mix_oof = best_w * m1_oof + (1.0 - best_w) * m2_oof
mix_pear = compute_pearsonr(y_all, mix_oof)
mix_mse = float(np.mean((mix_oof - y_all) ** 2))
logging.info(f"Final Ensemble Calibrated OOF -> MSE: {mix_mse:.6f}, Pearson: {mix_pear:.6f}")

# =============================================================================
# Save submission
# =============================================================================
logging.info("Preparing and saving submission.")
submission = pd.DataFrame({
    "id": df_test["id"],
    "score": test_final
})
submission.to_csv(CFG.submission_file, index=False)
logging.info(f"Submission saved to: {CFG.submission_file}")

# =============================================================================
# Extra analytics
# =============================================================================
logging.info("Extra analytics and diagnostics.")
train_score_counts = df_train["score"].value_counts().sort_index()
for s, c in train_score_counts.items():
    logging.info(f"Train score={s}: count={c}, pct={100.0*c/len(df_train):.2f}%")

logging.info("Token length diagnostics (first 100 train samples, variant 0).")
lens = []
for i in range(min(100, len(df_train_var0))):
    enc = tokenizer(
        df_train_var0["first_seq"].iloc[i],
        df_train_var0["second_seq"].iloc[i],
        truncation=True,
        max_length=CFG.max_length,
        add_special_tokens=True
    )
    lens.append(len(enc["input_ids"]))
lens = np.array(lens)
logging.info(f"Token length stats -> mean={lens.mean():.2f}, max={lens.max()}, min={lens.min()}")

logging.info("Pipeline completed with CUDA fp16, DAPT-MLM, dual-variant 5-fold cross-encoders, MTL head, LLRD, cosine schedule, EMA, FGM, R-Drop, per-fold isotonic calibration, and OOF-tuned ensembling. Submission ready.")