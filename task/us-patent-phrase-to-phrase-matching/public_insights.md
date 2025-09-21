### Overall Approach
- Fine-tuning a pre-trained BERT model for patents (anferico/bert-for-patents) to predict semantic similarity scores between patent phrases using a concatenation of context and anchor with target text as input. (2 recommendations)
- Ensemble of DeBERTa-v3-large models using attention-based pooling and cross-validation to predict patent phrase similarity scores. (14 recommendations)

### Data Preprocessing
- Extract CPC context text from external files using regex patterns and map them to the train and test datasets based on context codes. (2 recommendations)
- Convert all text to lowercase for uniformity. (3 recommendations)
- Combine anchor, target, and context text into a single 'text' field separated by [SEP] tokens. (15 recommendations)
- Merge test dataframe with CPC titles using the 'context' column to enrich context with corresponding titles. (6 recommendations)
- Load test dataset and CPC title codes from external CSV files. (2 recommendations)
- Load test data and sample submission from CSV files. (12 recommendations)
- Map CPC context codes to their corresponding text descriptions using a preloaded dictionary. (12 recommendations)

### Feature Engineering
- Use tokenizer to encode input pairs ('input' and 'target') into tokenized sequences suitable for transformer models. (2 recommendations)
- Created 'num_anchor' binary feature indicating presence of digits in 'anchor'. (2 recommendations)
- Tokenize anchor, target, and context text to determine maximum sequence length for padding. (8 recommendations)
- Tokenizing the concatenated text (anchor + '[SEP]' + target + '[SEP]' + context_text) using a pre-trained DeBERTa-v3-large tokenizer with max length 133 and padding. (10 recommendations)

### Validation Strategy
- Using out-of-fold predictions from a 4-fold stratified cross-validation setup to compute CV score via Pearson correlation. (2 recommendations)
- Out-of-fold predictions from 4-fold and 5-fold cross-validation were used to estimate model performance, with final submission based on ensemble inference on test data. (2 recommendations)
- Use 4-fold stratified k-fold cross-validation based on binned score values to evaluate model performance. (2 recommendations)

### Modeling
- Uses a fine-tuned DeBERTa-v3-large transformer with an attention mechanism and a linear head to predict continuous similarity scores. (3 recommendations)
- Two transformer-based models (DeBERTa-v3-large and RoBERTa-large) with attention and dropout-based pooling, trained via cross-validation and ensembled via model averaging. (2 recommendations)
- Fine-tune a pre-trained DeBERTa-v3-large transformer with an additional attention layer and a linear head, using BCEWithLogitsLoss and specified learning rates for encoder and decoder layers. (2 recommendations)
- Fine-tuned the microsoft/deberta-v3-small model for regression (num_labels=1) using Hugging Face Transformers Trainer with Pearson correlation as the evaluation metric. (2 recommendations)
- Using a fine-tuned BERT-for-patents model with a dropout layer and linear head for regression, trained with mean squared error loss and optimized via AdamW with linear schedule warmup. (2 recommendations)
- Uses a fine-tuned DeBERTa-v3-large model with an additional attention mechanism over the hidden states and a linear head for regression, trained across 5 folds with ensemble prediction. (3 recommendations)

### Post Processing
- Writing predicted scores to the submission file with 'id' and 'score' columns. (2 recommendations)
- Apply MinMaxScaler to each fold's predictions to normalize scores between 0 and 1. (2 recommendations)
- Apply sigmoid activation to model outputs to constrain predictions to [0,1] range. (9 recommendations)
- Applied MinMaxScaler to normalize predictions from both models to [0,1] range. (3 recommendations)
- Clip predicted scores to the range [0, 1] to ensure they adhere to the required submission bounds. (2 recommendations)
- Ensemble predictions across 4 folds by averaging their outputs. (6 recommendations)

### Technical Stack
- matplotlib (3 recommendations)
- torch.utils.data (2 recommendations)
- scipy (12 recommendations)
- logging (10 recommendations)
- pickle (4 recommendations)
- pandas (24 recommendations)
- tensorflow (2 recommendations)
- os (5 recommendations)
- tez (2 recommendations)
- numpy (23 recommendations)
- scikit-learn (6 recommendations)
- json (4 recommendations)
- tqdm (14 recommendations)
- re (4 recommendations)
- tokenizers (10 recommendations)
- warnings (4 recommendations)
- pathlib (3 recommendations)
- wandb (2 recommendations)
- seaborn (4 recommendations)
- datasets (6 recommendations)
- torch (23 recommendations)
- sklearn (13 recommendations)
- joblib (8 recommendations)
- transformers (22 recommendations)
- random (3 recommendations)

