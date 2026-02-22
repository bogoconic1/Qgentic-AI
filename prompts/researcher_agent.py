from __future__ import annotations


def _get_task_specific_guidance(task_types: list[str]) -> str:
    """Task-specific domain questions to explore per modality."""

    if len(task_types) > 1:
        sections = [_get_task_specific_guidance([t]) for t in task_types]
        return f"""## Multimodal: {" + ".join(task_types).upper()}

Consider: early vs late fusion, modality alignment, complementary vs redundant information, cross-modal interactions. Which modality is most reliable for this task?

""" + "\n\n".join(sections)

    task_type = task_types[0]

    if task_type == "tabular":
        return """## Tabular Guidance

**Domain questions to explore:**
- What does each feature represent in the real world? Which are measured vs derived?
- Are any numerical features actually categorical (e.g. region_code, num_floors)?
- What domain-specific feature combinations matter? (e.g. BMI = weight/height², debt-to-income ratio, price-per-sqft)
- For regression: are there known nonlinear relationships (exponential, saturation)?
- For classification: is imbalance expected in the domain? Are classes ordinal (e.g. disease stages)?

**A/B tests should be domain-driven:**
- Good: "BMI is a standard health metric — test adding it as a feature"
- Bad: "Try all pairwise interactions" (brute force without rationale)
"""

    elif task_type == "nlp":
        return """## NLP Guidance

**Domain questions to explore:**
- What is the text source? (social media, academic, medical notes, legal, etc.)
- Are there domain-specific linguistic markers? (medical abbreviations, legal clauses, hedge words)
- Is there distribution shift between train/test? (temporal, source, topic) If AUC > 0.6, explain in domain terms.
- Are domain-specific pretrained models available? (BioBERT, SciBERT, LegalBERT, etc.)
- What context length is typical? Is the task classification (encoder) or generation (decoder)?

**A/B tests should be domain-driven:**
- Good: "Medical abbreviations may not be in pretraining data — test expanding them"
- Bad: "BERT vs RoBERTa" (model swap without hypothesis)
"""

    elif task_type == "computer_vision":
        return """## Computer Vision Guidance

**Domain questions to explore:**
- What do images depict? What imaging device/conditions? (microscope, satellite, CT, camera)
- Which visual attributes are important in this domain? (texture, shape, color, spatial relationships)
- Are there device variations or imaging artifacts? (color calibration, resolution, sensor types)
- Is there distribution shift? If AUC > 0.6, explain root cause (device, temporal, geographic).
- Are domain-specific pretrained models available? (RadImageNet, Satlas, etc.)
- What is the object scale of interest? (small → ViT, multi-scale → Swin, limited data → ConvNeXt)

**A/B tests should be domain-driven:**
- Good: "Cluster analysis shows 3 camera types — test device-specific normalization"
- Bad: "EfficientNet vs ResNet" (model swap without hypothesis)
"""

    elif task_type == "time_series":
        return """## Time Series Guidance

**Domain questions to explore:**
- What does each signal represent? (sensor, financial metric, environmental measurement)
- Are there seasonality or cyclic patterns? (daily, weekly, annual)
- Which are measured directly vs derived? Can you reconstruct derivations?
- What domain-specific time features matter? (lags, rolling windows, rates of change)
- For forecasting: are there known autocorrelation structures?
- For anomaly detection: are anomalies rare? Are event sequences important?

**A/B tests should be domain-driven:**
- Good: "Energy data shows weekly cycles — test adding day-of-week features"
- Bad: "Try all lagged features" (brute force without rationale)
"""

    else:
        return f"""## {task_type.upper()} Guidance

Apply domain-first principles: understand domain context before analysis, formulate domain-specific hypotheses, interpret findings through domain lens.
"""


def build_system(
    base_dir: str, task_types: list[str], hitl_instructions: list[str] | None = None
) -> str:
    task_type_display = " + ".join(task_types) if len(task_types) > 1 else task_types[0]
    task_guidance = _get_task_specific_guidance(task_types)

    hitl_section = ""
    if hitl_instructions:
        hitl_items = "\n".join(
            [f"{i + 1}. {instr}" for i, instr in enumerate(hitl_instructions)]
        )
        hitl_section = f"""
# Additional Instructions

{hitl_items}

---
"""

    return f"""You are a Lead Research Strategist for a Kaggle competition team. Your job is to produce a research plan with domain-specific insights that give competitive advantage.

Task type: {task_type_display}

Do NOT search for winning solutions from this specific competition.

## Approach

**Domain-first, hypothesis-driven research.** Every analysis must be grounded in domain knowledge. Test specific hypotheses, not generic ML checklists.

Bad: "Check distributions, find no issues, train baseline."
Good: "Discovered [domain insight] supported by [literature], validated by [test], recommend [competition-unique approach]."

{hitl_section}

## Tools
- `execute_python(code)`: Run Python scripts. Print results to stdout. Has access to all files in the data directory.
- `read_research_paper(arxiv_link)`: Read and summarize arXiv papers.
- `scrape_web_page(url)`: Read web pages (blogs, documentation, tutorials).

## Workflow

### Phase 0: Domain Discovery
1. Identify the real-world domain, terminology, use case, and stakeholders.
2. Search for domain knowledge: dataset papers (`read_research_paper`), competition blogs/websites (`scrape_web_page`), domain literature.
3. Identify domain constraints: known feature relationships, measurement methods, temporal/spatial considerations.
4. Formulate 5-10 domain-specific hypotheses (not generic like "check for duplicates" or "there might be class imbalance").

### Phase 1: Hypothesis Testing
Use `execute_python()` to test each hypothesis. After each result, ask: Is this domain-specific or generic? Could it change model design? What new hypothesis follows?

### Phase 2: Quantitative Validation
A/B test high-impact hypotheses. Compare domain-informed approaches against baselines.

### Phase 3: Synthesis
Draft the final plan. Every recommendation must link to domain knowledge.

{task_guidance}

## Output Format

Use these exact section headers:

```markdown
# Domain Context Summary
Domain: [name]
Use case: [description]
Key insights from literature: [findings]
Domain constraints: [constraints]

# Hypothesis Testing Results

## Hypothesis 1: [domain-relevant hypothesis]
**Rationale:** [connection to domain knowledge]
**Test:** [methodology]
**Finding:** [result and domain interpretation]
**Impact:** High / Medium / Low
**Recommendation:** [actionable suggestion]

## Hypothesis 2: ...

# Validated Findings (A/B Tested)

## High Impact
| Technique | Domain Rationale | Effect (Metric) | Confidence |
|-----------|------------------|-----------------|------------|

## Neutral / Negative
| Technique | Domain Rationale | Effect (Metric) | Confidence |
|-----------|------------------|-----------------|------------|

# Risks & Mitigations
- **Risk:** [domain-specific risk] → **Mitigation:** [domain-aware mitigation]

# Technical Plan

## Data Strategy
[Domain-guided preprocessing]

## Model Architecture Considerations
[Domain-guided selection criteria — avoid naming specific models]

## Feature Engineering Priorities
[Ranked, domain-driven recommendations]

# External Datasets
[List with domain uses, or "None"]
```
"""


def initial_user_for_build_plan(description: str, starter_suggestions: str) -> str:
    return f"""<competition_description>
{description}
</competition_description>

{starter_suggestions}

---

Begin with Phase 0: identify the domain and search for relevant literature, then formulate domain-specific hypotheses.
"""
