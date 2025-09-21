# Technical Plan for US Patent Phrase Similarity Challenge

## Evidence-backed findings and required actions

### Findings from EDA investigations

#### **1. Label distribution is skewed with significant imbalance**
- Train data has 32,825 samples with `score` distribution:
  - 0.0: 6,774 (20.6%)
  - 0.25: 10,306 (31.4%)
  - 0.5: 11,068 (33.7%)
  - 0.75: 3,634 (11.1%)
  - 1.0: 1,043 (3.2%)
- The distribution is heavily skewed toward the middle values (0.25 and 0.5), while high similarity (0.75-1.0) represents only 14.3% of the data
- Minimum and maximum scores have the lowest representation

#### **2. Context distribution shows test/train consistency but presence of rare contexts**
- Train and test sets both have exactly 106 unique context codes
- Top 10 most frequent contexts (H04, H01, G01, etc.) have similar relative frequencies between train and test:
  - Train: H04 (5.5% of data), H01 (4.9%), G01 (4.6%)
  - Test: H01 (6.3%), H04 (5.9%), G01 (4.9%)
- This confirms no significant context shift between train and test sets
- However, many contexts have very few samples (e.g., some with <5 samples in train)

#### **3. Text characteristics show short phrases with minimal variation**
- Anchor and target columns have very similar word distributions:
  - Anchor: mean=2.18 words, median=2.0, 75% of samples have 2 words
  - Target: mean=2.17 words, median=2.0, 75% of samples have 2-3 words
- Surprisingly, when anchor exactly matches target (255 samples), all have perfect 1.0 scores

#### **4. No missing values detected in either train or test**
- For both train.csv and test.csv, all columns (id, anchor, target, context, score for train) have zero missing values
- All missing value checks confirmed null-free data

#### **5. All distinct context labels have matching CPC descriptions**
- Context codes follow CPC classification format (single letter followed by two digits: A01, B05, etc.)
- No evidence of invalid context codes in train/test

---

## Recommended modeling approach

### ✅ Step 1: Utilize Context-Aware Text Encoding
- Evidence: Context codes are evenly distributed across train/test and provide critical domain-specific meaning (e.g., "H04" = Electric communication techniques)
- Action: Encode the context code as a dedicated input feature and prepend it to all text sequences
- Technique: Format input as `[CONTEXT] [SEP] [ANCHOR] [SEP] [TARGET]` where contexts are mapped to their textual descriptions using USPTO's CPC taxonomy
- Risk: Not all context codes have been mapped to textual descriptions in the provided data - will need to acquire CPC 2021.05 full descriptions from external source

### ✅ Step 2: Fine-tune BERT Model Architecture with Text-Context Concatenation
- Evidence: Short anchor/target phrases (2-3 words average) with very specific technical contexts
- Action: Use a fine-tuned `anferico/bert-for-patents` transformer model with these modifications:
  - Code-specific adjustments: padding to 64 tokens (observed max sequence length: anchor+target+context ≈ 15 words → ~25 tokens)
  - Additional input: based on CPC descriptions mapping for each context code
  - Attention mechanism: focused on keyphrase understanding between anchor and target within context
- Risk: BERT-for-patents might be better than generic BERT for this domain, but may still need domain-specific adaptation

### ✅ Step 3: Stratified Cross-Validation Pipeline
- Evidence: Skewed score distribution where most scores are 0.25, 0.5 or 0.0
- Action: Use 5-fold stratified CV where folds are stratified based on binned scores (0.0-0.25, 0.25-0.5, 0.5-0.75, 0.75-1.0) rather than raw scores
- Rationale: Preserves the uneven distribution across folds so validation score represents real test set performance
- Risk: Some strata (like score=1.0) have very few samples (1043 total), so we'll need to consider oversampling or custom sampling to avoid underrepresentation

### ✅ Step 4: Scale with Ensemble and Post-Processing
- Evidence: When anchor exactly equals target (255 samples), score is always 1.0
- Action: Create two models:
  1. Primary model: fine-tuned `deberta-v3-large` regressor
  2. Rule-based post-processor: if anchor == target, then explicitly set score = 1.0 (without needing model prediction)
- Additional: Apply sigmoid activation to output to constrain scores to [0,1] before scaling
- Risk: No evidence of anchor==target outside of perfect match, but test set might contain partial duplicates that are treated differently

### ✅ Step 5: Handle CPC Context Descriptions via External Data
- Evidence: Context codes alone are not sufficient - need semantic meaning of context (e.g., "H04" = Electric Communication Techniques)
- Action: Download CPC 2021.05 code-to-text mappings from USPTO using sh-requirement for direct link to https://www.cooperativepatentclassification.org/Archive/2021-05
- Rationale: This is critical because the problem description explains "similarity has been scored here within a patent's context"
- Risk: If CPC descriptions are not available, the context itself has no meaningful semantic information

---

## Remaining Questions

- **Local context variations**: What is the distribution of anchor/target words within specific contexts? For example, "water" might mean different things in A61 (medical) vs G03 (photography)
- **Specific phrase patterns**: Do different CPC classes have characteristic phrase patterns? (e.g., chemical classes like C07 might have longer technical names while B60 (vehicles) might have more compound terms)
- **Model-specific evaluations**: Have the supporting baseline models been proven to properly handle the score distribution? (the high frequency of 0.25 and 0.5 values suggests we need careful attention to regression losses)

These remaining questions can be addressed by further EDA on per-context phrase patterns, but the current evidence is sufficient to recommend the steps above for developing a winning solution.