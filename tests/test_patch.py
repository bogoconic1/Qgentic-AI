import ast
import difflib
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from agents.developer import DeveloperAgent

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logging.getLogger("agents.developer").setLevel(logging.DEBUG)



WRONG_PATCH = """--- code_16_v2.py
+++ code_16_v2.py
@@ -20,5 +20,5 @@
 BASE_DIR = "task/us-patent-phrase-to-phrase-matching"
 OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "16")
-SUB_PATH = os.path.join(OUTPUT_DIR, "submission_2.csv")
-LOG_PATH = os.path.join(OUTPUT_DIR, "code_16_v2.txt")
+SUB_PATH = os.path.join(OUTPUT_DIR, "submission_3.csv")
+LOG_PATH = os.path.join(OUTPUT_DIR, "code_16_v3.txt")
 os.makedirs(OUTPUT_DIR, exist_ok=True)
@@ -90,10 +90,16 @@
     titles = titles.rename(columns={"code": "context"})
     train = train.merge(titles[["context", "title"]], on="context", how="left")
     test = test.merge(titles[["context", "title"]], on="context", how="left")
     train["title"] = train["title"].fillna("")
     test["title"] = test["title"].fillna("")
-    train["text_input"] = train.apply(lambda r: create_text_input(r["anchor"], r["target"], r["title"]), axis=1)
-    test["text_input"] = test.apply(lambda r: create_text_input(r["anchor"], r["target"], r["title"]), axis=1)
+    # Lowercased fields for two-segment tokenization
+    train["anchor_l"] = train["anchor"].astype(str).str.lower()
+    train["target_l"] = train["target"].astype(str).str.lower()
+    train["title_l"] = train["title"].fillna("").astype(str).str.lower()
+    test["anchor_l"] = test["anchor"].astype(str).str.lower()
+    test["target_l"] = test["target"].astype(str).str.lower()
+    test["title_l"] = test["title"].fillna("").astype(str).str.lower()
     # Group key to prevent leakage
     train["group"] = (train["anchor"].astype(str) + "||" + train["target"].astype(str))
     if debug_mode:
         # Take a head sample; maintain group integrity
         # Select groups until approximately debug_max_samples reached
@@ -151,7 +157,50 @@
     if "label" in batch[0]:
         labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32)
         result["labels"] = labels
     return result
 
+class PatentPairDataset(Dataset):
+    def __init__(self, anchors: List[str], targets: List[str], titles: List[str], labels: Optional[List[float]], tokenizer, max_len: int):
+        self.anchors = anchors
+        self.targets = targets
+        self.titles = titles
+        self.labels = labels
+        self.tokenizer = tokenizer
+        self.max_len = max_len
+
+    def __len__(self):
+        return len(self.anchors)
+
+    def __getitem__(self, idx):
+        item = {
+            "a": self.anchors[idx],
+            "t": self.targets[idx],
+            "c": self.titles[idx],
+        }
+        if self.labels is not None:
+            item["label"] = float(self.labels[idx])
+        return item
+
+def collate_pair(batch, tokenizer, max_len: int):
+    # Two-segment packing:
+    # text_a = [ANCHOR] anchor + [CONTEXT] title
+    # text_b = [TARGET] target
+    text_a = [f"[ANCHOR] {b['a']} [CONTEXT] {b['c']}".strip() for b in batch]
+    text_b = [f"[TARGET] {b['t']}".strip() for b in batch]
+
+    enc = tokenizer(
+        text_a,
+        text_b,
+        max_length=max_len,
+        padding=True,
+        truncation=True,
+        return_tensors="pt",
+    )
+    out = {
+        "input_ids": enc["input_ids"],
+        "attention_mask": enc["attention_mask"],
+    }
+    if "token_type_ids" in enc:
+        out["token_type_ids"] = enc["token_type_ids"]
+    if "label" in batch[0]:
+        out["labels"] = torch.tensor([b["label"] for b in batch], dtype=torch.float32)
+    return out
+
 class MeanPooler(nn.Module):
     def __init__(self):
         super().__init__()
     def forward(self, last_hidden_state, attention_mask):
         mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
@@ -363,21 +412,26 @@
     # Return the best model for test inference and the oof predictions for this split will be set by caller
     return val_pred, val_tgt, model
 
-def infer_on_dataset(model: nn.Module, texts: List[str], tokenizer, cfg: Config, device: torch.device) -> np.ndarray:
-    ds = PatentDataset(texts, labels=None, tokenizer=tokenizer, max_len=cfg.max_len)
+def infer_on_dataset(model: nn.Module, anchors: List[str], targets: List[str], titles: List[str], tokenizer, cfg: Config, device: torch.device) -> np.ndarray:
+    ds = PatentPairDataset(
+        anchors=anchors,
+        targets=targets,
+        titles=titles,
+        labels=None,
+        tokenizer=tokenizer,
+        max_len=cfg.max_len
+    )
     loader = DataLoader(
-        ds, batch_size=cfg.valid_bs, shuffle=False, num_workers=cfg.num_workers, pin_memory=True,
-        collate_fn=lambda b: collate_fn(b, tokenizer, cfg.max_len)
+        ds, batch_size=cfg.valid_bs, shuffle=False, num_workers=cfg.num_workers, pin_memory=True,
+        collate_fn=lambda b: collate_pair(b, tokenizer, cfg.max_len)
     )
     model.eval()
     all_preds = []
     with torch.no_grad():
         for batch in loader:
             input_ids = batch["input_ids"].to(device, non_blocking=True)
             attention_mask = batch["attention_mask"].to(device, non_blocking=True)
             token_type_ids = batch.get("token_type_ids", None)
             if token_type_ids is not None:
                 token_type_ids = token_type_ids.to(device, non_blocking=True)
             with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                 preds = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
             all_preds.append(preds.detach().float().cpu().numpy())
     preds = np.concatenate(all_preds)
     preds = np.clip(preds, 0.0, 1.0)
     return preds
@@ -398,4 +452,8 @@
     # Load data
     train_df, test_df, _ = load_and_prepare_data(cfg, debug_mode)
     tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
+    # Add special role tokens and keep tokenizer/model in sync
+    special_tokens = {"additional_special_tokens": ["[ANCHOR]", "[TARGET]", "[CONTEXT]"]}
+    num_added = tokenizer.add_special_tokens(special_tokens)
 
@@ -416,33 +474,31 @@
         trn_df = train_df.iloc[trn_idx].reset_index(drop=True).copy()
         val_df = train_df.iloc[val_idx].reset_index(drop=True).copy()
 
-        # Build datasets for fold specifically
-        trn_dataset = PatentDataset(trn_df["text_input"].tolist(), trn_df["score"].tolist(), tokenizer, cfg.max_len)
-        val_dataset = PatentDataset(val_df["text_input"].tolist(), val_df["score"].tolist(), tokenizer, cfg.max_len)
-
-        # Wrap into loaders inside the fold-specific training function
-        # Reuse train_one_fold but pass in sliced datasets through DataLoaders constructed there
-        # Instead, to use the same routine while keeping code manageable, temporarily override train_df inside the function:
-        # We'll pass a holder dataframe with the same API used in train_one_fold
-        fold_train_wrapper = pd.DataFrame({"text_input": trn_df["text_input"], "score": trn_df["score"]})
-        fold_valid_wrapper = pd.DataFrame({"text_input": val_df["text_input"], "score": val_df["score"]})
+        # Build datasets for fold specifically (two-segment inputs)
+        trn_dataset = PatentPairDataset(
+            anchors=trn_df["anchor_l"].tolist(),
+            targets=trn_df["target_l"].tolist(),
+            titles=trn_df["title_l"].tolist(),
+            labels=trn_df["score"].tolist(),
+            tokenizer=tokenizer,
+            max_len=cfg.max_len
+        )
+        val_dataset = PatentPairDataset(
+            anchors=val_df["anchor_l"].tolist(),
+            targets=val_df["target_l"].tolist(),
+            titles=val_df["title_l"].tolist(),
+            labels=val_df["score"].tolist(),
+            tokenizer=tokenizer,
+            max_len=cfg.max_len
+        )
 
         # Build model and loaders directly here to maintain customization and test-time usage
         model = DebertaRegressor(cfg.model_name)
         model.to(device)
+        # Ensure embeddings resized to accommodate added special tokens
+        model.backbone.resize_token_embeddings(len(tokenizer))
 
         train_loader = DataLoader(
             trn_dataset,
             batch_size=cfg.train_bs,
             shuffle=True,
             num_workers=cfg.num_workers,
             pin_memory=True,
-            collate_fn=lambda b: collate_fn(b, tokenizer, cfg.max_len)
+            collate_fn=lambda b: collate_pair(b, tokenizer, cfg.max_len)
         )
         valid_loader = DataLoader(
             val_dataset,
             batch_size=cfg.valid_bs,
             shuffle=False,
             num_workers=cfg.num_workers,
             pin_memory=True,
-            collate_fn=lambda b: collate_fn(b, tokenizer, cfg.max_len)
+            collate_fn=lambda b: collate_pair(b, tokenizer, cfg.max_len)
         )
 
@@ -541,7 +597,9 @@
         logging.info(f"MODE={'DEBUG' if debug_mode else 'FULL'} | FOLD={fold+1}/{total_folds} | FINAL_FOLD_PEARSON={fold_corr:.6f}")
 
         # Test inference for this fold
-        test_preds_fold = infer_on_dataset(model, test_df["text_input"].tolist(), tokenizer, cfg, device)
+        test_preds_fold = infer_on_dataset(
+            model, test_df["anchor_l"].tolist(), test_df["target_l"].tolist(), test_df["title_l"].tolist(),
+            tokenizer, cfg, device
+        )
         test_fold_preds.append(test_preds_fold)
 
         # Free memory
         del model
         torch.cuda.empty_cache()
"""


def _restore_original_from_diff(diff_text: str) -> str:
    original_lines = []
    for line in diff_text.splitlines():
        if not line or line.startswith("diff ") or line.startswith("index "):
            continue
        if line.startswith("--- ") or line.startswith("+++ ") or line.startswith("@@ "):
            continue
        prefix = line[0]
        content = line[1:]
        if prefix in {"-", " "}:
            original_lines.append(content)
    return "\n".join(original_lines) + "\n"


@pytest.fixture
def developer_agent(monkeypatch):
    monkeypatch.setattr(DeveloperAgent, "_load_benchmark_info", lambda self: None)
    agent = DeveloperAgent("us-patent-phrase-to-phrase-matching", iteration=16)
    agent.outputs_dir = Path("/workspace/gstar-project")
    agent.developer_log_path = agent.outputs_dir / "developer_patch.log"
    agent.plan_path = agent.outputs_dir / "developer_patch_plan.md"
    return agent


def test_apply_wrong_then_correct_patch_on_code_16_v2(developer_agent, tmp_path):
    base_source = Path("task/us-patent-phrase-to-phrase-matching/outputs/16/code_16_v2.py")
    base_source_content = base_source.read_text()

    work_dir = tmp_path / "outputs"
    work_dir.mkdir()

    copied_base = work_dir / base_source.name
    copied_base.write_text(base_source_content)

    developer_agent.outputs_dir = work_dir
    developer_agent.developer_log_path = work_dir / "developer_patch.log"
    developer_agent.plan_path = work_dir / "developer_patch_plan.md"

    normalized = DeveloperAgent._normalize_diff_payload(copied_base, WRONG_PATCH)
    assert normalized is not None

    new_file = developer_agent.outputs_dir / developer_agent._code_filename(3)
    if new_file.exists():
        new_file.unlink()

    result = developer_agent._apply_patch(base_version=2, diff_payload=WRONG_PATCH, target_version=3)
    assert result is not None
    assert "PatentPairDataset" in result
    assert "submission_3.csv" in result

    expected_lines = DeveloperAgent._apply_unified_diff(base_source_content.splitlines(), normalized.splitlines())
    assert expected_lines is not None
    assert result.splitlines() == expected_lines

    assert copied_base.read_text() == base_source_content

    try:
        assert new_file.exists()
    finally:
        if new_file.exists():
            new_file.unlink()
