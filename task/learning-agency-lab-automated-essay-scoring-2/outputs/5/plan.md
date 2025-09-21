**Technical Plan for Developer Implementation**

1. **Data Preprocessing**:
   - Apply standard text cleaning: lowercasing, remove HTML tags, URLs, email patterns, and special characters; normalize consecutive spaces/punctuation; expand contractions.
   - For speed, use `pandas` and `re` for efficient cleaning; avoid NLTK/spacy.

2. **Tokenizer & Dataset Setup**:
   - Use `DeBERTa-v3-large` tokenizer (`microsoft/deberta-v3-large`) with `max_length=512`, `truncation=True`, and padding to maximum length.
   - Convert labels: shift scores (1-6) to index (0-5) for model output layer.

3. **Class-Weighted Loss**:
   - Compute class weights inversely proportional to frequency:  
     `weight_i = total_essays / (6 * count_i)` for score class `i`.
   - Apply weights to `CrossEntropyLoss` during training.

4. **Stratified Cross-Validation & Training**:
   - Split data into **5-stratified folds** (based on score levels).
   - For each fold:
     - Train with `AdamW` optimizer: backbone LR=2e-5, head LR=3e-4, weight decay=0.01.
     - Gradient checkpointing enabled; batch size=16 (adjust to fit GPU memory).
     - Early stopping: stop if validation QWK doesn’t improve for 2 consecutive epochs.
     - Save best model per fold based on validation QWK.
   - **Critical**: During training, compute QWK on validation set using **rounded expected scores** (`round(sum(prob[i] * (i+1) for i in 0..5))`).

5. **Final Model Training**:
   - Train final model on **entire training set** with hyperparameters validated across folds.
   - Disable early stopping; train for 3-5 epochs.

6. **Submission Generation**:
   - For each test essay:
     - Tokenize → feed to final model → get logits → apply softmax.
     - Compute expected score: `sum((i+1) * prob[i])` for `i` in [0-5], then round to integer.
   - Clip values to [1,6] after rounding to ensure validity.

7. **Optimizations**:
   - Use mixed-precision training (`fp16=True` in Hugging Face Trainer) for speed.
   - Enable `torch.compile` for compiled model during inference (if compatible).
   - Log training time to stay under 9-hour GPU limit.

**Model Selection**: Only `DeBERTa-v3-large` fine-tuned for classification. No hand-crafted features or ensembles (preliminary analysis shows pure transformers outperform feature-based approaches in this dataset).

**Validation Metric**: Strictly use quadratic weighted kappa on validation folds to guide hyperparameters. Ensure code computes QWK accurately (import `sklearn.metrics.cohen_kappa_score` with `weights='quadratic'`).