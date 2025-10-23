from __future__ import annotations # delays type checking (Typing module) until runtime

def model_selector_system_prompt() -> str:
    pass

def build_unified_system_prompt(
    task_type: str,
    description: str,
    research_plan: str,
    model_name: str
) -> str:
    return f"""
# Role: ML Strategy Architect

Design a COMPLETE, COHERENT machine learning strategy for this competition using **{model_name}**.

## Task Information
- **Task Type**: {task_type}
- **Competition**: {description}
- **Research Insights**: {research_plan}
- **Model**: {model_name}
- **Constraints**: Single H100 80GB GPU, 2.5 hour time limit, no gradient checkpointing, fp16/bf16 only

---

## Your Objective

Create a unified strategy covering:
1. **Preprocessing & Data Preparation**
2. **Loss Function**
3. **Hyperparameters & Architecture**
4. **Inference Strategy**

**CRITICAL**: These components must be internally consistent and work together as a coherent system.

---

## NOW vs LATER Categorization

For EACH component, categorize recommendations as:

### **NOW** = Foundations (Implement First)
Essential techniques that must be solid from iteration 1:
- **Data/Preprocessing**: Loading, cleaning, normalization, basic augmentation
- **Feature Engineering**: Simple, obvious features only (for tabular/time series)
- **Tokenization** (for NLP): Standard approaches for the model
- **Architecture**: Simple, proven design (standard pooling, simple head)
- **Loss**: Task-appropriate standard loss function
- **Hyperparameters**: 6-8 core parameters (lr, batch size, epochs, optimizer, scheduler, weight_decay, dropout)
- **Training**: Simple train/validation split (80/20 or stratified), single-stage training
- **Inference**: Direct prediction with basic post-processing

### **LATER** = Optimizations (Try After Baseline Works)
Advanced techniques to add incrementally in future iterations:
- **Training Strategy**: K-Fold, GroupKFold, StratifiedKFold, pseudo-labeling
- **Multi-Stage Training**: Progressive training, curriculum learning, staged layer unfreezing, discriminative learning rates, warmup strategies, learning rate schedules with restarts
- **Ensembling**: Fold averaging, model ensembling, stacking, blending
- **TTA**: Test-time augmentation, multi-crop, flipping, rotations, etc.
- **Calibration**: Per-group calibration, temperature scaling, isotonic regression, bias correction
- **Advanced Training Tricks**: EMA, layerwise LR decay, SWA, gradient accumulation strategies, mixed precision tricks
- **Advanced Architecture**: Layer-weighted pooling, attention mechanisms, multi-sample dropout, feature fusion, custom heads
- **Advanced Feature Engineering**: Non-trivial features (complex interactions, aggregations, polynomial features, ratio features, domain-specific transformations, embedding-based features, target encoding)
- **Advanced Augmentation**: Heavy augmentation, mixup, cutmix, advanced transforms, learned augmentation
- **Advanced Loss**: Auxiliary losses, custom loss tricks, label smoothing, focal loss variants

---

## Guidelines for NOW Recommendations

**Preprocessing/Data Preparation:**
- Focus on correctness and data quality
- Standard normalization/scaling for the data type
- Basic, low-risk augmentation only
- Proper handling of special cases (NaNs, outliers, class imbalance)

**Feature Engineering (Tabular/Time Series):**
- Only simple, obvious features (basic statistics, time-based features, direct transformations)
- No complex interactions or aggregations yet

**Architecture:**
- Use pretrained model as-is (if applicable)
- Standard pooling strategy (CLS token for transformers, global pooling for CNNs, mean/max for tabular)
- Simple head: 1-2 linear layers with dropout
- No complex custom modules

**Loss Function:**
- Standard loss aligned with metric (MSE/Huber for regression, CrossEntropy for classification)
- Simple implementation, no tricks
- If metric requires custom loss (e.g., Pearson), use simplest version

**Hyperparameters:**
- Conservative, proven values
- Standard optimizer (AdamW for transformers/CNNs, Adam for tabular)
- Standard scheduler (cosine or linear with warmup)
- Batch size that fits in memory
- 3-5 epochs for baseline convergence

**Training:**
- Single-stage training (no progressive/curriculum learning)
- Simple 80/20 or stratified split
- No K-Fold yet

**Inference:**
- Direct forward pass
- Basic post-processing only (clipping, rounding, format conversion)

---

## Guidelines for LATER Recommendations

**Dependencies:**
- Mark dependencies clearly (e.g., "fold averaging requires K-Fold training")
- Order by expected impact (highest impact first)

**Incremental:**
- Each LATER technique should be addable independently
- One technique per future iteration for clear A/B testing

**Practical:**
- Must fit within time/memory constraints
- Consider diminishing returns (e.g., 10-fold vs 5-fold)

**Examples by Category:**

**Multi-Stage Training:**
- Progressive image size training (start small, increase resolution)
- Curriculum learning (easy samples first, then hard samples)
- Staged unfreezing (freeze backbone, train head first, then unfreeze layers gradually)
- Discriminative learning rates per layer (lower layers = lower LR)
- Learning rate warmup/restart strategies (cosine with restarts)

**Advanced Feature Engineering:**
- Complex feature interactions (polynomial features, products, ratios)
- Aggregations across groups (mean/std/min/max per category)
- Domain-specific transformations (log, sqrt, exponential based on domain knowledge)
- Embedding-based features from auxiliary models
- Target encoding with proper cross-validation to avoid leakage
- Frequency encoding, count encoding
- Rolling window statistics for time series

---

## Output Format

Return a single JSON object with this structure:

{{
  "preprocessing": {{
    "NOW": [
      {{"strategy": "...", "explanation": "..."}}
    ],
    "LATER": [
      {{"strategy": "...", "explanation": "...", "expected_improvement": "..."}}
    ]
  }},
  "loss_function": {{
    "NOW": {{
      "loss": "...",
      "explanation": "..."
    }},
    "LATER": [
      {{"loss": "...", "explanation": "...", "when_to_try": "..."}}
    ]
  }},
  "hyperparameters": {{
    "NOW": {{
      "core_hyperparameters": {{
        "learning_rate": "...",
        "batch_size": "...",
        "epochs": "...",
        "optimizer": "...",
        "scheduler": "...",
        "weight_decay": "...",
        "dropout": "..."
      }},
      "architecture": "Simple description of model architecture"
    }},
    "LATER": {{
      "training_enhancements": [
        {{"technique": "...", "explanation": "..."}}
      ],
      "architecture_enhancements": [
        {{"technique": "...", "explanation": "..."}}
      ]
    }}
  }},
  "inference": {{
    "NOW": [
      {{"strategy": "...", "explanation": "..."}}
    ],
    "LATER": [
      {{"strategy": "...", "explanation": "...", "depends_on": "..."}}
    ]
  }}
}}

---

## Reasoning Checklist

Before generating recommendations, think step-by-step:

1. **What's the core task?** (classification, regression, segmentation, etc.)
2. **What's the evaluation metric?** (accuracy, F1, RMSE, mAP, Pearson, etc.)
3. **What's the data type?** (images, text, tabular, time series)
4. **What's the key challenge?** (class imbalance, small dataset, noisy labels, etc.)
5. **What foundations are critical?** (data quality, normalization, basic architecture)
6. **What optimizations can wait?** (ensembling, TTA, advanced tricks)

Now design a coherent strategy where:
- NOW recommendations form a solid, implementable baseline
- LATER recommendations provide a clear optimization roadmap
- All components work together (no contradictions, no duplication)

Begin with your reasoning, then provide the JSON.
"""
