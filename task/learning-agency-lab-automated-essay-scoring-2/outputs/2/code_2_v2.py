# task/learning-agency-lab-automated-essay-scoring-2/code_2_v1.py

import os
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import cohen_kappa_score
import lightgbm as lgb
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import warnings
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths
DATA_DIR = "task/learning-agency-lab-automated-essay-scoring-2"
OUTPUTS_DIR = os.path.join(DATA_DIR, "outputs", "2")
SUBMISSION_PATH = os.path.join(OUTPUTS_DIR, "submission.csv")

# Ensure output directory exists
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logging.info("Downloading punkt tokenizer...")
    nltk.download('punkt')

# -------------------------------
# 1. Preprocessing Functions
# -------------------------------

def clean_text(text):
    """Clean and normalize text."""
    if not isinstance(text, str):
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Normalize multiple spaces and punctuation
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\.{2,}', '.', text)
    
    # Convert to lowercase and strip
    text = text.lower().strip()
    
    return text

# -------------------------------
# 2. Feature Engineering
# -------------------------------

def extract_features(df):
    """Extract hand-crafted features for LightGBM."""
    logging.info("Extracting features...")
    features = pd.DataFrame()
    
    features['word_count'] = df['full_text'].apply(lambda x: len(word_tokenize(x)))
    features['sentence_count'] = df['full_text'].apply(lambda x: len(sent_tokenize(x)))
    features['char_count'] = df['full_text'].apply(len)
    features['avg_word_length'] = features['char_count'] / (features['word_count'] + 1e-8)
    
    # Punctuation ratios
    features['comma_per_sentence'] = df['full_text'].apply(lambda x: x.count(',')) / (features['sentence_count'] + 1e-8)
    features['period_per_sentence'] = df['full_text'].apply(lambda x: x.count('.')) / (features['sentence_count'] + 1e-8)
    features['question_mark_per_sentence'] = df['full_text'].apply(lambda x: x.count('?')) / (features['sentence_count'] + 1e-8)
    
    # Complex word ratio
    features['complex_word_ratio'] = df['full_text'].apply(
        lambda x: len([w for w in word_tokenize(x) if len(w) > 6]) / (len(word_tokenize(x)) + 1e-8)
    )
    
    # First and last sentence lengths
    def get_first_sent_len(text):
        sents = sent_tokenize(text)
        return len(sents[0]) if sents else 0
    
    def get_last_sent_len(text):
        sents = sent_tokenize(text)
        return len(sents[-1]) if sents else 0
    
    features['first_sentence_length'] = df['full_text'].apply(get_first_sent_len)
    features['last_sentence_length'] = df['full_text'].apply(get_last_sent_len)
    
    return features

# -------------------------------
# 3. Load and Process Data (Limited Sample)
# -------------------------------

def load_data():
    """Load a small sample of training and test data."""
    logging.info("Loading data...")
    
    # Load train data (first 100 rows)
    train_path = os.path.join(DATA_DIR, "train.csv")
    train_df = pd.read_csv(train_path)
    
    # Load test data (first 100 rows)
    test_path = os.path.join(DATA_DIR, "test.csv")
    test_df = pd.read_csv(test_path)
    
    # Clean text
    train_df['full_text'] = train_df['full_text'].apply(clean_text)
    test_df['full_text'] = test_df['full_text'].apply(clean_text)
    
    return train_df, test_df

# -------------------------------
# 4. LightGBM Model with TF-IDF
# -------------------------------

def train_lightgbm_model(train_df, test_df):
    """Train a LightGBM model using TF-IDF and hand-crafted features."""
    logging.info("Training LightGBM model...")
    
    # Extract features
    train_features = extract_features(train_df)
    test_features = extract_features(test_df)
    
    # TF-IDF features
    tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=1000, stop_words='english')
    train_tfidf = tfidf.fit_transform(train_df['full_text']).toarray()
    test_tfidf = tfidf.transform(test_df['full_text']).toarray()
    
    # Combine features
    train_features_combined = np.hstack([train_features.values, train_tfidf])
    test_features_combined = np.hstack([test_features.values, test_tfidf])
    
    # Prepare for training
    X = train_features_combined
    y = train_df['score'].values - 1  # Convert to 0-5 for LGBM
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Reduced folds for small sample
    oof_preds = np.zeros((len(train_df), 6))
    test_preds = np.zeros((len(test_df), 6))
    
    params = {
        'objective': 'multiclass',
        'num_class': 6,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logging.info(f"Training fold {fold + 1}...")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=100,  # Reduced for speed
            callbacks=[lgb.log_evaluation(period=0)]
        )
        
        oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        test_preds += model.predict(test_features_combined, num_iteration=model.best_iteration) / skf.n_splits
    
    # Convert back to 1-6 scale
    oof_scores = np.argmax(oof_preds, axis=1) + 1
    test_scores = np.argmax(test_preds, axis=1) + 1
    
    # Calculate QWK
    qwk = cohen_kappa_score(train_df['score'].values, oof_scores, weights='quadratic')
    logging.info(f"LightGBM QWK (OOF): {qwk:.4f}")
    
    return test_scores, test_preds

# -------------------------------
# 5. Simple Heuristic Baseline (Fallback)
# -------------------------------

def simple_baseline_model(train_df, test_df):
    """A simple baseline model based on essay length."""
    logging.info("Running simple baseline model...")
    
    # Calculate mean score for each word count bin
    train_df['word_count'] = train_df['full_text'].apply(lambda x: len(word_tokenize(x)))
    
    # Create bins and calculate mean score per bin
    train_df['word_bin'] = pd.cut(train_df['word_count'], bins=10, labels=False)
    score_map = train_df.groupby('word_bin')['score'].mean().to_dict()
    
    # Predict on test set
    test_df['word_count'] = test_df['full_text'].apply(lambda x: len(word_tokenize(x)))
    test_df['word_bin'] = pd.cut(test_df['word_count'], bins=10, labels=False)
    test_df['word_bin'] = test_df['word_bin'].fillna(train_df['word_bin'].median())  # Fill NaN with median bin
    
    predictions = test_df['word_bin'].map(score_map).fillna(train_df['score'].mean())  # Fallback to mean
    
    # Round to nearest integer and clip to 1-6
    predictions = np.round(predictions).clip(1, 6).astype(int)
    
    return predictions.values

# -------------------------------
# 6. Main Execution
# -------------------------------

def main():
    """Main function to run the complete pipeline."""
    logging.info("Starting Automated Essay Scoring pipeline...")
    
    # Load data
    train_df, test_df = load_data()
    logging.info(f"Loaded {len(train_df)} training samples and {len(test_df)} test samples")
    
    try:
        # Train LightGBM model
        test_scores_lgb, test_probs = train_lightgbm_model(train_df, test_df)
        logging.info("LightGBM model training completed")
        
        # Use LightGBM predictions
        final_predictions = test_scores_lgb
        logging.info("Using LightGBM predictions")
        
    except Exception as e:
        logging.warning(f"LightGBM model failed: {e}. Falling back to simple baseline.")
        # Fallback to simple baseline
        final_predictions = simple_baseline_model(train_df, test_df)
        logging.info("Using simple baseline predictions")
    
    # Create submission
    submission = pd.DataFrame({
        'essay_id': test_df['essay_id'],
        'score': final_predictions
    })
    
    # Save submission
    submission.to_csv(SUBMISSION_PATH, index=False)
    logging.info(f"Submission saved to {SUBMISSION_PATH}")
    
    # Show submission head
    logging.info("\nSubmission preview:")
    logging.info(submission.head())

if __name__ == "__main__":
    main()