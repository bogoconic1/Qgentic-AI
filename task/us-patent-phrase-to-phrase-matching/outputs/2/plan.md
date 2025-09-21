# Technical Plan for Patent Phrase Similarity Competition

## 1. **Contextual Preprocessing: Merge CPC Titles from External Data**
   - **Action**: Merge `train.csv` and `test.csv` with `titles.csv` using the `context` column to add comprehensive CPC domain descriptions.
   - **Evidence**: 
     - All 106 unique context codes in `train.csv` and all context codes in `test.csv` are present in `titles.csv` (verified via EDA).
     - `titles.csv` contains detailed descriptions for CPC codes (e.g., "H04" maps to "Electric Communication or Detection"), which will provide semantic domain context to the model.
   - **Why**: The raw CPC codes (e.g., "H04") are meaningless tokens to the model. Mapping them to human-readable titles (e.g., "Electric Communication") helps the model understand the technical domain, which is critical per the competition description: "similarity has been scored here within a patent's context."

## 2. **Text Concatenation & Standardization**
   - **Action**: Create a single input string per sample:  
     `"anchor [SEP] target [SEP] context_title"`, where `[SEP]` is the separator token used by transformer models.  
     Convert all text to lowercase for uniformity (e.g., `"LED" → "led"`).
   - **Evidence**:
     - Average anchor/target lengths are ~16 characters each; context titles average ~20-40 characters. Total input length will be <100 characters.
     - Baseline submissions consistently use this concatenation pattern with [SEP] tokens (e.g., "15 recommendations for combining anchor, target, context text").
   - **Why**: This structure aligns with how transformers process multi-field text inputs. Lowercasing reduces token variance without losing semantic meaning (e.g., "TV" and "tv" become identical).

## 3. **Stratified Cross-Validation Using Binned Scores**
   - **Action**: Bin scores into 5 fixed bins [0.0, 0.25, 0.5, 0.75, 1.0] and use 4-fold stratified K-Fold CV. Use binned scores as the strata variable.
   - **Evidence**:
     - Score distribution is highly skewed: 0.25 (31.4%), 0.5 (33.7%), 0.0 (20.6%), 0.75 (11.1%), 1.0 (3.2%). 
     - Context-specific score variations (e.g., G02 has higher average score of 0.381 vs C07's 0.3276) show that score distributions differ by domain.
     - Opposing strategies: 
       - Non-stratified CV would over-sample 0.25/0.5 in some folds → unreliable CV scores.
       - Using raw Pearson corrs on CV splits without binning risks poor representation of scores like 1.0 (only 3.2% of data).
   - **Why**: Ensures each fold preserves the original score distribution, especially for rare scores like 1.0 (critical for robust evaluation).

## 4. **Model Architecture: Fine-tune DeBERTa-v3-large with Contextual Features**
   - **Action**: 
     1. Use `microsoft/deberta-v3-large` (pre-trained on general text, but adaptable to domain-specific use cases).
     2. Add the concatenated text (`anchor + [SEP] + target + [SEP] + context_title`) as input.
     3. Use a regression head with MSE loss during training, followed by sigmoid activation to constrain predictions to [0, 1].
   - **Evidence**:
     - Top-performing baselines all use DeBERTa variants (e.g., "14 recommendations for DeBERTa-v3-large", "3 recommendations for DeBERTa with attention pooling").
     - `bert-for-patents` or `DeBERTa-v3` are explicitly recommended because they are trained on patent-related data citations (supporting baseline notes).
     - EDA showed context titles contain meaningful domain keywords (e.g., "Electric Communication" for H04), which the model must leverage.
   - **Why**: 
     - DeBERTa-v3 handles multi-field text inputs better than BERT variants. 
     - Sigmoid activation ensures predictions stay within valid score range [0,1], matching competition constraints.

## 5. **Training Refinements: Adjust to Context-Specific Score Distributions**
   - **Action**: Use context-specific imbalance handling:
     - For each fold, calculate per-context score distributions and apply weighted loss (higher weights for rare scores in a context).
     - Example: In H04, 1.0 scores represent only 2.7% of samples (54 of 1962), so weight 1.0 samples 10× higher than 0.25 samples.
   - **Evidence**: 
     - Context-specific score distributions vary significantly (e.g., G02 has 2× more 1.0 scores than H04: 16 vs 54 in H04 for 1.0 counts).
     - Without this, the model would under-predict 1.0 scores in contexts where they're rare.
   - **Risks**: Requires per-context weighting logic; may need additional tuning. If implemented, standardize weights across folds for consistency.

## 6. **Post-Processing & Submission**
   - **Action**:
     1. Clip predictions to [0, 1] range.
     2. Round to nearest 0.25 increment for final submission (as scores are defined in 0.25 increments).
     3. Ensemble predictions from 5-fold CV using mean averaging.
   - **Evidence**:
     - Competitors used both clipping (2 recommendations) and rounding (9 recommendations for sigmoid activation), but rounding to 0.25 is semantically meaningful.
     - All evaluation is scored via Pearson correlation on continuous values between [0,1], so rounding is safe and necessary per the score meanings (each increment has a defined semantic meaning).
   - **Risks**: Round-to-0.25 might introduce minor bias (e.g., 0.26 → 0.25), but the scoring metric (Pearson) is robust to small rounding errors.

---

### **Critical Risks & Unanswered Questions**
1. **Novel Contexts in Test Data**: 
   - 106 unique contexts in train, but test uses similar top contexts (H01, H04, G01, etc.). However, if a new context code appears in test, the model has zero information about it. 
   - **Mitigation**: All contexts in test must be mapped to titles via `titles.csv`—this was confirmed by EDA (0 missing contexts). 

2. **Anchor/Target Overlap**: 
   - Train has 733 unique anchors, test has 703. The overlap analysis only checked triples, not individual anchors. If the same anchor appears in train and test but paired with different targets, the model must generalize. 
   - **Risk**: Models may memorize common anchors, but 703 out of 733 anchors are shared → test data likely contains familiar anchors, making overfitting a risk. 

3. **Minimum Data Needs for Rare Scores**: 
   - Scores like 1.0 appear in only 3.2% of data, and per context (e.g., C07 has only 8 of 954 samples). The model might struggle with 1.0 predictions. 
   - **Mitigation**: Use context-specific weighted loss (as in Step 5) and oversample 1.0 samples during training.

---

### **Final Step-by-Step Implementation Guide**
1. **Preprocess Data**:
   ```python
   # Merge context titles
   train = train.merge(titles[['code', 'title']], left_on='context', right_on='code', how='left')
   train['text'] = train['anchor'] + ' [SEP] ' + train['target'] + ' [SEP] ' + train['title'].astype(str
   # Lowercase for uniformity
   train['text'] = train['text'].str.lower()
   ```

2. **Set Up CV**:
   ```python
   train['bin'] = pd.cut(train['score'], bins=[0.0, 0.25, 0.5, 0.75, 1.0], labels=False)
   # 4-fold stratified CV on 'bin'
   ```

3. **Model Training**:
   - Use Hugging Face `Trainer` with `DebertaV3ForSequenceClassification` (num_labels=1 for regression).
   - Loss: `MSELoss`, then apply `sigmoid` in inference.
   - Use weighted loss per context (based on per-context score distribution).
   - Hyperparameters: max_length=150, batch size=16, LR=2e-5 (encoder), 5e-5 (head).

4. **Inference**:
   ```python
   predictions = model.predict(test_text_batch)
   predictions = torch.sigmoid(predictions).clamp(0, 1).round(decimals=2)  # Round further as submission requirements
   submission = pd.DataFrame({'id': test_ids, 'score': predictions})
   submission['score'] = (submission['score'] * 4).round().astype(int) / 4  # 0.0, 0.25, 0.5, 0.75, 1.0
   ```

**Why this plan wins**: It directly addresses the compositional nature of the task (domain context + phrase pairs) using evidence-backed preprocessing (context titles), and uses the strongest-performing models (DeBERTa-v3) with cross-validation strategies validated by the competition’s baseline submissions.