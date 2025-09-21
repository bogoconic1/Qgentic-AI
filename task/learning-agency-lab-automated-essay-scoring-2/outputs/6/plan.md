### Step-by-Step Technical Plan for Developer

#### **Foundation: Data Distribution & Quality Insights (Confirmed by EDA)**
1. **Class Imbalance Analysis (Critical Risk)**  
   - **Evidence**: Score distribution in train.csv is highly skewed: score 1 (7.2%), score 2 (27.3%), score 3 (36.1%), score 4 (22.9%), score 5 (5.6%), score 6 (0.87%).  
   - **Action**:  
     - Use **stratified cross-validation (at least 5 folds)** to ensure each fold contains representative examples of all scores (especially rarer scores 5/6).  
     - Implement **class-weighted loss functions** during model training to upweight rare classes. For example, in transformers, set `class_weight` for CrossEntropyLoss or use focal loss to reduce focus on dominant classes.  
   - **Risk**: Without correction, models will likely default to predicting score 3 for all essays (baseline approach), resulting in severe underprediction of higher scores and poor quadratic weighted kappa.  

2. **Text Length Correlation with Score (Key Signal)**  
   - **Evidence**: Average word count increases with score (1: 267, 6: 767 words); character counts follow the same trend (1: 1,511, 6: 4,491). Standard deviations also widen for higher scores (e.g., score 6 word count SD: 157 vs. score 1: 106).  
   - **Action**:  
     - **Do not remove/trim text** – length is a valid signal (longer essays = higher quality in argumentative writing).  
     - Include **explicit text-length features** (word count, character count) in hybrid models alongside contextual features.  
   - **Risk**: If the model ignores length correlations (e.g., when using fixed-token-length truncation), it will lose a strong predictive signal. Ensure `max_length=512` for tokenizers (validated by max word count 1,656 in training, fitting within BERT/DeBERTa limits).

3. **Error Analysis from Text Samples (Content Patterns)**  
   - **Evidence**:  
     - Low scores (1): Frequent grammar/spelling errors ("diffrent", "tempertures"), fragmented thoughts, lack of structure.  
     - High scores (6): Coherent arguments with evidence-based reasoning ("Dr. Huang... states"), structured paragraphs, proper citations.  
   - **Action**:  
     - Design **grammatical correctness features** (e.g., POS tagging error rates, spelling accuracy via spellchecker) and **rhetorical features** (e.g., transition word counts, sentence complexity).  
     - Use **transformer-based feature extraction** (e.g., DeBERTa embeddings) to capture semantic coherence beyond surface-level grammar.  
   - **Risk**: Models trained solely on bag-of-words (e.g., TF-IDF) may miss structural improvements present in high-score essays.

---

#### **Validation Strategy (Balancing Performance and Robustness)**  
4. **Cross-Validation Setup**  
   - **Evidence**: No duplicate essays, consistent statistics across train/test sets (e.g., word count ranges overlap).  
   - **Action**:  
     - Use **5-fold stratified KFold** (by score) for training to ensure fold-level score distribution matches the dataset skew.  
     - For transformer fine-tuning, **freeze lower layers** early (e.g., first 6 of 24 DeBERTa layers) to prevent overfitting on small score 5/6 examples.  
   - **Risk**: Non-stratified splits (e.g., random folds) would under-sample score 6 in some folds, leading to unstable training for higher scores.

---

#### **Feature Engineering Priorities (Confirmed by Trends)**  
5. **Multilevel Feature Extraction**  
   - **Evidence**: Word count and length increase with score, but sample essays show differences in structure beyond simple length (e.g., score 6 essays cite specific sources, use subordinate clauses).  
   - **Action**:  
     - **Paragraph-level features**: Count of paragraphs with >20 words (score 6 essays averaged 5+ paragraphs in samples).  
     - **Sentence-level features**: Average sentence length (score 6 had longer sentences in samples), sentence count, and punctuation diversity (e.g., semicolon/colon usage).  
     - **Vocabulary features**: Type-token ratio (TTR = unique words / total words; score 6 likely higher), and rare-word counts (e.g., words from academic word lists).  
   - **Risk**: Models relying only on raw text without structural features may fail to capture nuanced writing quality indicators.

---

#### **Model Architecture Guidance**  
6. **Transformer Fine-Tuning Over LGBM/XGBoost**  
   - **Evidence**: Sample essays require deep contextual understanding (e.g., recognizing "FACS" as a proper noun in score 6 essays versus generic phrases in score 1).  
   - **Action**:  
     - **Primary model**: Fine-tune `microsoft/deberta-v3-large` (supports 512-token limit, fits max essay length seen) with these adjustments:  
       - **Custom loss**: Weighted CrossEntropyLoss with weights inverse to score frequency (e.g., score 6 weight = 15576 / 135 ≈ 115x, score 1 weight = 15576 / 1124 ≈ 14x).  
       - **Threshold post-processing**: For regression-based models, clip to [1,6] and round to integer; **do not apply softmax** since scores are ordinal integers (classification would misrepresent scoring scale).  
     - **Secondary model**: If LGBM is used, train on hand-crafted features (e.g., POS error rates, sentence complexity) + DeBERTa embeddings as additional inputs.  
   - **Risk**: Pure LGBM models on raw text will underperform transformers for nuanced language understanding; hybrid models (LGBM + transformer embeddings) should be tried but prioritize transformer-only finesse.

---

#### **Unanswered Questions & Next EDA Steps**  
- **Critical uncertainty**: Do the essays share common prompts? If all essays share the same prompt (e.g., "Discuss Venus"), topics won’t confound scoring. But different prompts for different essays would mean scores depend on relevance to the prompt.  
  - **Next EDA**: Prove prompt consistency: `ask_eda("Do all train.csv essays share the same prompt text? If not, show prompt variations and their score distributions.")`  

- **Hidden leakage risk**: Test set contains "rerun" test (8k samples) – was it split from train? If train/test were split prior to scoring (e.g., pre-existing system), papers may have similar patterns.  
  - **Next EDA**: Check overlap between test and train perplexity scores: `ask_eda("For a random 1000 essays from train and test, calculate the average perplexity of a pre-trained GPT-2 small model. If test perplexity is statistically similar to train, it implies minimal distribution shift.")`  

- **Structural feature validation**: Are punctuation patterns (e.g., colons, semicolons) predictive of score? Current sample shows score 6 uses formal punctuation.  
  - **Next EDA**: `ask_eda("Calculate average count of colons, semicolons, and dashes per essay by score (1-6) in train.csv.")`  

> **Summary**: The path forward prioritizes transformer fine-tuning with weighted loss and structural features. Immediate next EDA must confirm prompt consistency and test/train distribution similarity before proceeding to model tuning. Avoid rule-based length normalization (e.g., truncation) – essay length is a key signal.