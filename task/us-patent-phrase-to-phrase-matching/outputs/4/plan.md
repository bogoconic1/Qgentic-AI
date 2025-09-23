

### Step-by-Step Technical Plan for Developer

#### **1. Merge titles.csv with training/test data to enrich context descriptions**
- **Evidence from EDA**: 
  - `titles.csv` has 260,476 rows with columns `code`, `title`, `section`, etc. 
  - All 106 unique context codes in `train.csv` are covered in `titles.csv`'s `code` column (100% match).
  - Example: Context "H04" in the dataset maps to a row in `titles.csv` where `code = "H04"` and `title = "TELECOMMUNICATIONS"` (exact title varies per code).
- **Action**: 
  - Merge `train.csv` and `test.csv` with `titles.csv` on `context` ↔ `code` to add `title`, `section`, and `subclass` fields.
  - **Why?** The CPC title provides critical domain context (e.g., "H04" → "TELECOMMUNICATIONS"). Baseline solutions (10+ submissions) use this enrichment to disambiguate synonyms (e.g., "battery" in context "H01M" [Batteries] vs. "H02J" [Electrical circuits]).
- **Risk**: 
  - Some codes in `titles.csv` may have hierarchical formats (e.g., "A01B" vs. "A01"). Confirm `context` in train/test exactly matches `code` structure in `titles.csv` (already validated, but double-check for edge cases like 2-digit vs. 4-digit codes).

#### **2. Address score distribution skew in model training strategy**
- **Evidence from EDA**: 
  - Score distribution: 0.25 (31.4%), 0.5 (33.7%), 0.0 (20.6%), 0.75 (11.1%), 1.0 (3.2%). 
  - Class 1.0 is severely underrepresented (only 1,043 samples vs. 11,068 for 0.5).
- **Action**: 
  - Use **stratified 5-fold cross-validation** with bins of scores (e.g., [0.0-0.2], [0.2-0.4], ..., [0.8-1.0]) to preserve the distribution in each fold.
  - **Why?** High scores (e.g., 1.0) are too rare to reliably train a model if not stratified. Baselines using 4-5 fold stratified CV (14+ submissions) achieve stable interpolation for rare scores.
- **Alternate solution**: 
  - Use loss weighting (higher weights for rare scores) or oversample high-score examples, but CV stratification is less error-prone for regression tasks.
- **Risk**: 
  - 1.0-score examples are only ~3% of the dataset. If test set has even fewer, the model may underperform on this category. Monitor high-score predictions in CV results.

#### **3. Check for train-test data leakage by (anchor, target, context) overlap**
- **Evidence from EDA**: 
  - Train dataset has 0 duplicate (anchor, target, context) rows.
  - **No prior check** for overlapping pairs between train and test sets yet.
- **Action**: 
  - Run `ask_eda("How many (anchor, target, context) triplets exist in both train.csv and test.csv?")`.
  - **Why?** If duplicates exist, the model could "cheat" by memorizing training pairs in test. Even if scores are missing in test, overlap would violate competition rules.
- **Hypothesis**: 
  - Given the number of unique anchors/targets (e.g., "LED" appears in multiple contexts), leakage may occur if the pair is reused in test for a new context. But data size (32k train + 3.6k test) suggests low risk.
- **Risk**: 
  - If leakage exists, model performance may be unrealistically high. This must be verified before proceeding.

#### **4. Validate train-test context distribution shift**
- **Evidence from EDA**: 
  - Train top contexts: H04 (5.98%), H01 (5.96%), G01 (4.97%).
  - Test top context distribution **not explicitly checked yet** (only total rows: 3,648 with 106 unique codes).
- **Action**: 
  - Run `ask_eda("What is the count and percentage of each context code in test.csv? Show top 10 context codes by count")`.
  - Compare with train distribution (e.g., if "G06" is 20% of test but 3% of train, model may fail on underrepresented domains).
- **Why?** Contexts like "H04" (telecom) may have different semantic patterns than "A61" (medical devices). A shift could degrade generalization.
- **Risk**: 
  - If test has more rare contexts than train, the model may overfit to common contexts. Baselines use CPC titles (see Step 1) to disambiguate context-specific nuances.

#### **5. Handle character/word length distributions carefully**
- **Evidence from EDA**: 
  - Anchor/target lengths are nearly identical across train/test:
    - Anchor: avg 16 chars, 2.18 words in train; 16 chars, 2.18 words in test.
    - Target: avg 15.75 chars, 2.16 words in test (vs. 15.76 chars, 2.17 words in train).
  - Max target length: 98 chars in train vs. 47 in test → test sequences may be shorter.
- **Action**: 
  - Use **DYNAMIC PADDING** with max length = 64 tokens (empirically safe for anchor+target+context_title) to avoid truncating longer instances.
  - **Why?** Current models (DeBERTa-v3-small) use max 128 tokens but this is wasteful for short phrases. Truncation could lose information for targets up to 98 chars in train.
- **Risk**: 
  - If test targets have higher max length than train (unlikely based on current data, but verify via Step 4), the model may fail on unseen lengths.

#### **6. Investigate score-per-context patterns**
- **Evidence from EDA**: 
  - Top contexts like H04 (telecom) have high counts (5.98% of train), but score distribution per context is not yet analyzed.
- **Action**: 
  - Run `ask_eda("For context codes H04, G01, and A61, show the score distribution (count of each 0.0/0.25/0.5/0.75/1.0)")`.
  - **Why?** In patent contexts, "TV set" and "television set" might have high scores in "G06" (computing) context but low scores in "H04" (telecom) context. Domain knowledge affects semantics.
- **Hypothesis**: 
  - High-similarity scores (0.75-1.0) may concentrate in specific contexts (e.g., technical synonyms in "C07" chemicals), while 0.25/0.0 scores dominate in others. This justifies context-based feature engineering.

#### **Remaining Critical Unverified Questions**
- 🔍 **Leakage check**: Are any (anchor, target, context) triplets shared between train and test? (Step 3)
- 🔍 **Test context distribution**: Is it consistent with train? (Step 4)
- 🔍 **Longest sequences in test**: Does target length exceed train's max (98 chars)? (Step 5)
- 🔍 **CPC code structure**: Do all context codes directly map to `titles.csv`'s `code` (or do some codes require hierarchical lookup)? (Step 1)

---

### Next Tool Calls Required
1. `ask_eda("How many (anchor, target, context) triplets exist in both train.csv and test.csv?")`
2. `ask_eda("What is the count and percentage of each context code in test.csv? Show top 10 context codes by count")`
3. `ask_eda("For context codes H04, G01, and A61, show the score distribution (count of each 0.0/0.25/0.5/0.75/1.0)")`
4. `ask_eda("What is the maximum target character count in test.csv?")`

**Only after these tools return results, finalize the modeling architecture (e.g., full input envelope including CPC title, max sequence length, and CV strategy).**