# code_1_v1.py
import os
import pandas as pd
import numpy as np
import logging
import re
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Define paths
DATA_DIR = "task/learning-agency-lab-automated-essay-scoring-2"
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs", "1")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def preprocess_text(text):
    """Clean and preprocess text"""
    logger.info("Preprocessing text...")
    # Convert to string if not already
    text = str(text)
    
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

def extract_features(df):
    """Extract hand-crafted features for traditional ML model"""
    logger.info("Extracting features...")
    features = pd.DataFrame()
    
    # Basic counts
    features['word_count'] = df['full_text'].apply(lambda x: len(x.split()))
    features['sentence_count'] = df['full_text'].apply(lambda x: len(re.split(r'[.!?]+', x)))
    features['char_count'] = df['full_text'].apply(len)
    
    # Averages
    features['avg_word_length'] = features['char_count'] / features['word_count']
    features['avg_sentence_length'] = features['word_count'] / features['sentence_count']
    
    # Punctuation ratios
    features['comma_per_sentence'] = df['full_text'].apply(lambda x: x.count(',')) / features['sentence_count']
    features['period_per_sentence'] = df['full_text'].apply(lambda x: x.count('.')) / features['sentence_count']
    features['question_per_sentence'] = df['full_text'].apply(lambda x: x.count('?')) / features['sentence_count']
    
    # Complex word ratio
    features['complex_word_ratio'] = df['full_text'].apply(
        lambda x: len([w for w in x.split() if len(w) > 6]) / len(x.split())
    )
    
    # First and last sentence length
    features['first_sentence_length'] = df['full_text'].apply(
        lambda x: len(re.split(r'[.!?]+', x)[0].split()) if re.split(r'[.!?]+', x)[0] else 0
    )
    features['last_sentence_length'] = df['full_text'].apply(
        lambda x: len(re.split(r'[.!?]+', x)[-1].split()) if re.split(r'[.!?]+', x)[-1] else 0
    )
    
    return features

def quadratic_weighted_kappa(y_true, y_pred):
    """Calculate quadratic weighted kappa"""
    logger.info("Calculating QWK...")
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

class OptimizedRounder:
    """Optimize rounding thresholds to maximize QWK"""
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 1
            elif pred < coef[1]:
                X_p[i] = 2
            elif pred < coef[2]:
                X_p[i] = 3
            elif pred < coef[3]:
                X_p[i] = 4
            elif pred < coef[4]:
                X_p[i] = 5
            else:
                X_p[i] = 6

        ll = quadratic_weighted_kappa(y, X_p)
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [1.5, 2.5, 3.5, 4.5, 5.5]
        constraints = (
            {'type': 'ineq', 'fun': lambda x: x[1] - x[0] - 0.1},
            {'type': 'ineq', 'fun': lambda x: x[2] - x[1] - 0.1},
            {'type': 'ineq', 'fun': lambda x: x[3] - x[2] - 0.1},
            {'type': 'ineq', 'fun': lambda x: x[4] - x[3] - 0.1},
        )
        result = minimize(loss_partial, initial_coef, method='SLSQP', constraints=constraints)
        self.coef_ = result.x

    def predict(self, X, coef=None):
        if coef is None:
            coef = self.coef_
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 1
            elif pred < coef[1]:
                X_p[i] = 2
            elif pred < coef[2]:
                X_p[i] = 3
            elif pred < coef[3]:
                X_p[i] = 4
            elif pred < coef[4]:
                X_p[i] = 5
            else:
                X_p[i] = 6
        return X_p

def main():
    logger.info("Starting Automated Essay Scoring...")
    
    # Load subset of training data
    logger.info("Loading training data...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    # train_df = train_df.head(100)  # Use only first 100 rows
    train_df['full_text'] = train_df['full_text'].apply(preprocess_text)
    
    # Load test data
    logger.info("Loading test data...")
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    # test_df = test_df.head(100)  # Use only first 100 rows
    test_df['full_text'] = test_df['full_text'].apply(preprocess_text)
    
    # Extract features
    train_features = extract_features(train_df)
    test_features = extract_features(test_df)
    
    # TF-IDF features
    logger.info("Extracting TF-IDF features...")
    tfidf = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
    train_tfidf = tfidf.fit_transform(train_df['full_text']).toarray()
    test_tfidf = tfidf.transform(test_df['full_text']).toarray()
    
    # Combine features
    X_train = np.hstack([train_features.values, train_tfidf])
    X_test = np.hstack([test_features.values, test_tfidf])
    y_train = train_df['score'].values
    
    # Model training with cross-validation
    logger.info("Training model with cross-validation...")
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    oof_predictions = np.zeros(len(train_df))
    test_predictions = np.zeros(len(test_df))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        logger.info(f"Training fold {fold + 1}...")
        
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        # Train logistic regression model
        model = LogisticRegression(max_iter=1000, class_weight='balanced')
        model.fit(X_tr, y_tr)
        
        # Predict on validation set
        val_pred_proba = model.predict_proba(X_val)
        val_pred = np.argmax(val_pred_proba, axis=1) + 1
        oof_predictions[val_idx] = val_pred
        
        # Predict on test set
        test_pred_proba = model.predict_proba(X_test)
        test_predictions += np.argmax(test_pred_proba, axis=1) + 1
    
    # Average test predictions
    test_predictions /= skf.n_splits
    
    # Optimize rounding thresholds
    logger.info("Optimizing rounding thresholds...")
    optR = OptimizedRounder()
    optR.fit(oof_predictions, y_train)
    optimized_predictions = optR.predict(test_predictions)
    
    # Create submission
    logger.info("Creating submission...")
    submission_df = pd.DataFrame({
        'essay_id': test_df['essay_id'],
        'score': np.clip(optimized_predictions, 1, 6).astype(int)
    })
    
    # Save submission
    submission_path = os.path.join(OUTPUT_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    logger.info(f"Submission saved to {submission_path}")

if __name__ == "__main__":
    from functools import partial
    main()