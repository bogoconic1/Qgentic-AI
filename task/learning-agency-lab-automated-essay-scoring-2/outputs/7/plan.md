

### Step-by-Step Technical Plan for the Developer

#### **1. Merge Competition Data with Persuade Corpus 2.0 (Aggregated)**
- **Evidence**:  
  - Competition’s train set has severe imbalance: only **135 essays for score 6** and **876 for score 5** (96.5% of the data is scores 3–4).   
  - Persuade Corpus 2.0 contains **285,383 essays with holistic scores**, with **13,981 entries for score 6** and **46,452 for score 5**. After aggregating discourse segments into full essays (grouping by `essay_id_comp` and concatenating `discourse_text`), this dataset provides **100× more examples for rare classes** (e.g., score 6).  
  - Columns like `grade_level`, `ell_status`, and `economically_disadvantaged` suggest Persuade data is **student-authored** (K-12 context), matching the competition’s target population.  
- **Action**:  
  - For Persuade Corpus 2.0:  
    - Group by `essay_id_comp` → concatenate `discourse_text` into `full_text` and retain `holistic_essay_score` as the target.  
    - Remove any duplicates (e.g., same `full_text` with two scores; current EDA shows none in competition data, but check Persuade for internal consistency).  
  - Merge with competition’s `train.csv` (ensuring no overlapping `essay_id` between competition and Persuade datasets; based on EDA, competition’s `essay_id` format differs from Persuade’s `essay_id_comp`).  
- **Risk**: Domain mismatch between competition and Persuade (e.g., if Persuade uses a different rubric).  
  - **Mitigation**: Conduct an ablation study scoring models trained on (a) competition-only data and (b) combined data. Validate using QWK on competition test set.  

---

#### **2. Text Preprocessing (Standardized for All Data)**
- **Evidence**:  
  - Direct inspection of sample essays (e.g., Entry 1/2/3) reveals inconsistent punctuation, HTML-like artifacts, and irregular spacing (e.g., "voterw cant", "complut the program").  
  - Top baseline strategies (e.g., 13/9 recommendations) suggest removing HTML tags, URL patterns, and normalizing whitespace/punctuation.  
- **Action**:  
  - Apply these exact transformations to **all datasets (competition + Persuade)**:  
    ```python
    # Example pipeline:
    def clean_text(text):
        text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
        text = re.sub(r'https?://\S+', '', text)  # Remove URLs
        text = re.sub(r'[^a-zA-Z0-9\s.,?!\'"]', '', text)  # Keep only standard punctuation
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
        text = text.lower()  # Lowercase
        return text
    ```
- **Risk**: Over-aggressive preprocessing might remove meaningful context (e.g., "don't" → "dont" loses contraction nuance).  
  - **Mitigation**: Test with small manual samples to validate retention of critical context (e.g., contractions, hyphenated words).  

---

#### **3. Stratified Cross-Validation with Balanced Class Distribution**
- **Evidence**:  
  - Post-merger datasets will have **~300k essays** but with skewed class distribution (e.g., scores 1–2 still have fewer samples than 3–6).  
  - Baseline recommendations (9/6 recommendations for 80-20 stratified splits, 6 for 15-fold stratified) emphasize handling score distribution consistently.  
- **Action**:  
  - Use **5-fold stratified KFold** (stratified by score) on the merged train set. Ensure each fold maintains the same score distribution as the full merged data.  
  - Verify fold consistency via:  
    ```python
    # Example validation code:
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5)
    for fold, (_, val_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold} score distribution: {y[val_idx].value_counts()}")
    ```
- **Risk**: Poor fold separation causing score leakage across folds (e.g., identical essays in train/validation).  
  - **Mitigation**: Always check `essay_id` uniqueness across folds (competition EDA confirmed no overlaps in train/test; validate merge logic also avoids overlaps).  

---

#### **4. Fine-Tune a DeBERTa-v3 Large Transformer**
- **Evidence**:  
  - **24/31 baseline recommendations** involve `transformers` and `DeBERTa` models.  
  - QSMA (Quadratic Weighted Kappa) evaluation requires ordinal-aware predictions (not binary classification). DeBERTa excels at capturing nuanced grammatical arguments in writing.  
  - Class imbalance highlights need for loss weighting: Rare classes (e.g., score 6) contribute disproportionately to errors in QWK.  
- **Action**:  
  - Use `microsoft/deberta-v3-large` with a **classification head** (6 output classes).  
  - Apply **class weights** inversely proportional to score frequency in the merged dataset:  
    ```python
    class_counts = [1124+4389, 4249+46248, 5629+89150, 3563+85163, 876+46452, 135+13981]  # Competition + Persuade
    weights = 1 / np.array(class_counts)
    class_weights = weights / weights.sum() * len(class_counts)  # Normalize
    ```
  - Train with **3 epochs**, using **AdamW optimizer** (LR=5e-5), and **early stopping** if validation QWK doesn’t improve for 2 epochs.  
- **Risk**: Model overfitting on Persuade data (e.g., if Persuade essays use academic/jargon-heavy language not typical of student essays).  
  - **Mitigation**: Monitor validation QWK during training; if QWK drops on competition test set, ignore Persuade data in final training.  

---

#### **5. Post-Processing for Discrete Scores**
- **Evidence**:  
  - 17/31 recommendations include **clipping predictions to [1, 6]** and rounding to integers.  
  - QWK is computed on discrete scores (not continuous), so continuous outputs from regression risks misalignment with the metric.  
- **Action**:  
  - Convert model outputs as follows:  
    ```python
    # After softmax logits:
    predictions = np.argmax(logits, axis=1) + 1  # Score 1–6
    predictions = np.clip(predictions, 1, 6)  # Ensure bounds
    ```
  - **Avoid rounding continuous regression outputs**; use classification for ordinal tasks.  
- **Risk**: Binning discrete predictions may lose granularity (e.g., "4.9" → 5 could confuse a task requiring fine-grained scores).  
  - **Mitigation**: Validate using QWK with both classification and regression approaches during ablation studies.  

---

### Remaining Uncertainties to Validate
1. **Domain Compatibility of Persuade Data**:  
   - Confirm `grade_level` distribution in Persuade matches competition’s student population (e.g., mostly grades 9–12). Run:  
     `ask_eda("Calculate grade_level distribution in persuade_corpus_2.0.csv where holistic_essay_score is not null")`.  
2. **Conflict Between Rubrics**:  
   - Check if scoring rubrics for competition and Persuade are identical (e.g., using [Holistic Scoring Rubric](https://storage.googleapis.com/kaggle-media/competitions/The%20Learning%20Agency/Rubric_%20Holistic%20Essay%20Scoring.pdf)). If Persuade uses a different rubric (e.g., longer/richer essays for higher scores), merging may hurt performance.  

### Final Recommendation
- **Immediately proceed to merge competition and Persuade data** after validating grades and rubric consistency.  
- **Focus on DeBERTa-v3-large with class weighting** as the primary model, replacing weaker alternatives like LGBM (which only had 19 baseline recommendations vs. 24 for DeBERTa).  
- **Prioritize QWK optimization** over accuracy or other metrics during training validation.  

> **QWK is non-linear and sensitive to error severity**—for example, predicting score 6 as 3 contributes 10× more to the weighted error than predicting it as 5. This demands deeper understanding of writing structure than raw error rates.