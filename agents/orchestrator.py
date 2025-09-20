from typing import Tuple
import subprocess

# from agents.eda import EDAAgent
# from agents.researcher import ResearcherAgent
from agents.developer import DeveloperAgent
class Orchestrator:
    def __init__(self, slug: str, iteration: int):
        self.slug = slug
        self.iteration = iteration
        # self.eda = EDAAgent(slug, data_path=f"data/{slug}")
        # self.researcher = ResearcherAgent(slug)
        self.developer = DeveloperAgent(slug, iteration)

    def run(self, max_code_tries: int = 3) -> Tuple[bool, str]:
        # self.eda.build_summary()
        # plan = self.researcher.build_plan()
        plan = """
Based on the EDA results, I now understand the data structure and key challenges:

1. **Severe class imbalance**: Scores 5 and 6 are extremely rare (5.6% and 0.87% respectively), which will cause models to underpredict high scores without targeted handling.
2. **Strong length-score correlation**: Essay length (words, sentences, characters) is a strong predictor of score (correlation ~0.65–0.69), suggesting lexical and structural features are highly informative.
3. **No duplicates**: All essays are unique, so no deduplication needed.

### Final Technical Plan for Developer:

**Goal**: Maximize Quadratic Weighted Kappa (QWK) on this imbalanced 6-class scoring task.

**Strategy**: Ensembe of two complementary models — a Transformer (DeBERTa-v3-large) to capture semantic and discourse patterns, and a LightGBM model to leverage explicit text statistics — with careful post-processing to address class imbalance.

---

### ✅ Step-by-Step Implementation Plan:

#### 1. **Preprocessing**  
- Clean text:  
  - Remove HTML tags and URLs  
  - Normalize multiple spaces, punctuation (e.g., `...` → `.`)  
  - Convert to lowercase  
  - Strip leading/trailing whitespace  
- **Do NOT remove numbers or expand contractions** — they are signal in student writing.

#### 2. **Feature Engineering**  
- Extract **hand-crafted features** for LightGBM:  
  - `word_count`, `sentence_count`, `char_count`  
  - `avg_word_length`, `comma_per_sentence`, `period_per_sentence`, `question_mark_per_sentence`  
  - `complex_word_ratio` (% words >6 chars)  
  - `first_sentence_length`, `last_sentence_length`  
  - TF-IDF (ngram_range=(1,3), max_features=1000) on full text  
  - Part-of-speech ratios (using spaCy: NOUN, VERB, ADJ, ADV)  
- **Do NOT truncate essays** — use full text for both models.

#### 3. **Transformer Model (DeBERTa-v3-large)**  
- Use Hugging Face `transformers` for sequence classification (6 classes: 1-6)  
- Tokenizer: `microsoft/deberta-v3-large` with `max_length=512`, padding='max_length'  
- Architecture:  
  - Classifier head: Linear layer (768 → 6) + softmax  
  - Use **weighted cross-entropy** loss with weights inversely proportional to class frequency  
  - Optimize for QWK via **custom loss + label smoothing**  
- Training:  
  - 10 epochs, batch_size=8, warmup_steps=500, AdamW  
  - Use **5-fold stratified CV** — preserve score distribution in each fold  
  - Save out-of-fold logits for ensemble  
- **Critical**: Output probabilities → round to nearest integer **ONLY after** post-processing (see below).

#### 4. **LightGBM Model**  
- Train on structured features (hand-crafted + TF-IDF)  
- Use **custom QWK objective** and metric (available in LightGBM via `objective='multiclass'` + `num_class=6` and custom eval)  
- Hyperparameter tuning with Optuna: focus on `num_leaves`, `min_child_samples`, `learning_rate`, `subsample`  
- Use **5-fold stratified CV** — ensure each fold has all 6 score levels  
- Aggregate out-of-fold predictions

#### 5. **Ensemble & Post-Processing**  
- Combine predictions:  
  - **Weighted average**: DeBERTa (55%) + LightGBM (45%)  
    - Tuned on validation set using grid search over [0.4–0.7] weights  
- **Post-processing to fix imbalance**:  
  - Use **pred_dist_adjustment**:  
    - Calculate actual distribution of predicted scores on train set  
    - Apply **optimal thresholding** to map continuous prediction to discrete scores:  
      Use method: `OptimizedRounder` (from GitHub) or `scipy.optimize` to minimize QWK loss  
  - **Clip** predictions to [1,6] integer range  
- Final submission: **integers only**

#### 6. **Validation & Monitoring**  
- Use **stratified 5-fold CV** for all model selection  
- Log per-class F1 and QWK at each fold — prioritize score 5 and 6 F1  
- Track both QWK and **individual class recall** — do not let score 6 be ignored

---

### ✅ Why This Wins:  
- DeBERTa captures complex writing style/signals  
- LightGBM exploits strong length/statistical signals missed by transformers  
- Post-processing corrects for extreme rarity of top scores  
- Ensemble > any single model on this high-stakes, imbalanced task  
- No GPU dependency — fully CPU-compatible for reproducibility  

**Deploy on CPU only**, validate against public leaderboard, submit final ensemble.
"""
        success = self.developer.run(plan, max_tries=max_code_tries)
        if success:
            # mlebench grade-sample /workspace/gstar-project/task/learning-agency-lab-automated-essay-scoring-2/outputs/3/submission.csv learning-agency-lab-automated-essay-scoring-2
            subprocess.run([
                "mlebench", "grade-sample",
                f"/workspace/gstar-project/task/{self.slug}/outputs/{self.iteration}/submission.csv",
                self.slug
            ])
  
        return success, plan
    

