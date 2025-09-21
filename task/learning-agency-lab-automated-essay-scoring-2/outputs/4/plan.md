

Technical Plan for Developer:

1. **Data Preprocessing**:
   - Load `train.csv` and split into 5 stratified folds based on `score` to maintain class distribution.
   - Tokenize `full_text` using `bert-base-uncased` tokenizer with 512 token maximum length (truncate longer essays).
   - Compute class weights as `max_class_count / class_count` for cross-entropy loss (e.g., score 6 has highest weight ~41.7).

2. **Model Training**:
   - For each fold, train a `bert-base-uncased` model as follows:
     - Use PyTorch with mixed precision for speed.
     - Apply class-weighted cross-entropy loss during training.
     - Track quadratic weighted kappa (QWK) on validation set every epoch.
     - Stop training early if validation QWK doesn't improve for 2 consecutive epochs.
     - Save the model with highest validation QWK.

3. **Ensemble Prediction**:
   - For each test essay, run all 5 trained models and average their softmax probability outputs.
   - Assign final score as **argmax** of averaged probabilities (converting to integer 1-6).

4. **Efficiency & Validation**:
   - Use GPU acceleration for 9-hour runtime allowance.
   - Validate model ensemble performance on a held-out train subset before final submission.
   - Ensure submission file format: `essay_id,score` with no extra columns.

**Note**: Focus solely on maximizing QWK for Leaderboard Prize. Do not optimize for CPU runtime or model size.