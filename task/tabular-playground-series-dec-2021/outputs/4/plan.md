

### Technical Plan for Developer

#### **Step 1: Data Cleaning and Preprocessing**
**Evidence-Based Actions:**  
1. **Remove `Cover_Type` column from test data**  
   - *Why*: Test set should not contain target labels (per competition rules). EDA shows `test.csv` includes `Cover_Type`, which is invalid and risks accidental leakage if used during training.  
   - *Action*: Immediately drop the `Cover_Type` column from `test.csv` upon loading.  
   - *Risk*: If the competition's official test data does not have this column, this cleanup is critical to avoid future errors.  

2. **Drop Soil_Type7 and Soil_Type15**  
   - *Why*: EDA confirmed both columns contain only `0` values globally (zero variance). They provide no signal.  
   - *Action*: Remove these columns from both train and test datasets.  
   - *Risk*: None—these columns are useless and safely removable.  

3. **Clip Hillshade values to [0, 255]**  
   - *Why*: Hillshade features physically cannot exceed 0–255 (e.g., `Hillshade_9am` min=-4, max=297 in train). Synthetic data introduces invalid values.  
   - *Action*: Apply `np.clip(x, 0, 255)` to all hillshade columns (`Hillshade_9am`, `Hillshade_Noon`, `Hillshade_3pm`).  
   - *Risk*: Minimal—clipping preserves physical validity without distorting meaningful data.  

4. **Fix negative horizontal distances**  
   - *Why*: Horizontal distances to hydrology, roads, and fire points cannot be negative (physical distances). E.g., `Horizontal_Distance_To_Hydrology` min=-82 (train), `Horiz_Dist_To_Roadways` min=-287 (train).  
   - *Action*: Convert negative values to `0` for these columns (no physical meaning for negative distances).  
   - *Risk*: Assumes negative values are data errors (plausible for synthetic data). No meaningful negative distances exist in reality.  

5. **Normalize Aspect values**  
   - *Why*: Aspect is a circular variable (0–360°). EDA shows min=-33, max=405 in train (invalid values).  
   - *Action*: Apply `aspect = aspect % 360`. For negative values, use `(aspect % 360 + 360) % 360` to wrap into [0, 359] range. Then create circular encoding: `aspect_sin = sin(aspect)`, `aspect_cos = cos(aspect)`. Drop original `Aspect` column.  
   - *Risk*: Missing from EDA: Check if any `0–360` values are consistent. If synthetic data has isolated errors (e.g., single -33), wrapping is safe.  

6. **Handle rare `Cover_Type` classes in training data**  
   - *Why*: EDA reveals `Cover_Type` 5 has only 1 training example (0.00003%). `Cover_Type` 4 has 333 rows (0.01%), which is effectively negligible for meaningful learning.  
   - *Action*: Remove rows with `Cover_Type` 5 from train data. For `Cover_Type` 4, retain but apply class weighting during training to prioritize rare classes.  
   - *Risk*: Removing `Cover_Type` 5 reduces training samples but prevents model bias toward 1:3.6M ratio. If actual test data has rare classes, model may underperform on them (common for synthetic datasets).  

---

#### **Step 2: Investigate Data Structure Consistency**
**Evidence-Based Actions:**  
1. **Analyze Wilderness Area sum distributions**  
   - *Why*: EDA showed 366k train rows (10.2%) and 40k test rows (10%) have `Wilderness_Area` sums ≠ 1. This violates the assumption that Wilderness Areas are mutually exclusive (one-hot encoded).  
   - *Action*: Ask for breakdown of `Wilderness_Area` sums:  
     ```plaintext
     "Frequency table showing count of rows where Wilderness_Area1-4 sum equals 0, 1, 2, 3, or 4+ in train and test"
     ```  
   - *Hypothesis*: If sum=0 in many rows, fields are missing entirely; if sum>1, multiple areas are mixed (data error).  
   - *Next step*: Use sum distribution to decide whether to force mutually exclusive encoding (e.g., select highest-value column) or treat as independent binary features.  

2. **Check Soil_Type sum consistency**  
   - *Why*: Soil_Type columns should ideally be mutually exclusive (one-hot encoded), but EDA didn’t verify sum distribution.  
   - *Action*: Ask for `Soil_Type1–40` sum breakdown per row in train and test.  
   - *Hypothesis*: Extreme sum values (>1) imply synthetic data errors. If global sum >1 exceeds 10% of rows, soil columns should be treated as independent features (not one-hot).  

---

#### **Step 3: Feature Engineering for Physical Validity**
**Evidence-Based Actions:**  
1. **Create hydrology-related distance features**  
   - *Why*: Horizontal/vertical distances to hydrology are critical in forest cover contexts. EDA shows identical distributions in train/test (valid for synthesis), but negatives need fixing.  
   - *Action*: After cleaning negatives:  
     - Compute `Total_Distance_To_Hydrology = sqrt(Horizontal_Distance_To_Hydrology² + Vertical_Distance_To_Hydrology²)`  
     - Compute `To_Hydrology_Direction = atan(Vertical_Distance_To_Hydrology / Horizontal_Distance_To_Hydrology)` if horizontal distance > 0.  
   - *Risk*: Low. These are standard domain-engineered features for terrain data.  

2. **Aggregate hillshade statistics**  
   - *Why*: Hillshade features (9am, noon, 3pm) capture sun exposure over time. EDA shows clipping is needed to physical bounds.  
   - *Action*: After clipping:  
     - Create `Hillshade_Mean = average of H9am, HNoon, H3pm`  
     - Create `Hillshade_Range = max(Hillshades) - min(Hillshades)`  
   - *Risk*: None—these capture latent structure without overfitting.  

3. **Compute Wilderness/Area aggregations**  
   - *Why*: If Wilderness_Area sums are inconsistent, direct binary columns may mislead.  
   - *Action*: If sum analysis shows sum=0 for many rows, create a `Wilderness_Area_Count` feature. If sum>1 is frequent, keep individual columns as-is (no mutual exclusivity).  
   - *Next step*: Requires results from Wilderness Area sum distribution analysis.  

---

#### **Step 4: Validation Strategy**
**Evidence-Based Actions:**  
1. **Use stratified k-fold cross-validation (10–20 folds)**  
   - *Why*: Train data is highly imbalanced (Class 2 = 56.6%, others rare). Stratification ensures rare classes are represented in each fold.  
   - *Action*: Apply `StratifiedKFold` with `n_splits=20` and default `random_state=42`.  
   - *Risk*: High k-fold may increase training time but is justified for rare classes.  

2. **Validate train/test distribution shifts**  
   - *Why*: Synthetic data may have shifted distributions. EDA enabled test/train comparison for features (e.g., elevation min/max similar), but not dependent variables.  
   - *Action*: After cleaning, run EDA to compare key numeric feature distributions (e.g., Elevation, Hillshade) between train and test. Use Kolmogorov-Smirnov test for statistical differences.  
   - *Risk*: Safe—no leakage if applied post-training data cleaning.  

---

#### **Critical Unanswered Questions & Risks**
1. **Wilderness_Area and Soil_Type sum distributions**  
   - If >10% of rows have invalid sums, the model may struggle to learn relationships. If so, entreat the domain experts to clarify if mutual exclusivity is a hard constraint. If not, treat as independent features.  

2. **Test data leakage via synthetic CTGAN training**  
   - *Risk*: CTGAN might recreate real data patterns from the original Forest Cover Type dataset. If "train" is synthetic from the original, it may leak information about real forest types. However, the synthetic dataset is provided as base data, so this is acceptable.  

3. **Cover_Type 4 and 5 handling**  
   - If test set has realistic representation of these rare classes, model will underperform. If synthetic data doesn’t replicate real target distribution (e.g., test shows more Class 4 samples than train), focus on class-weighted training.  

**Final Note**: No external datasets are recommended for this competition—the synthetic data is self-contained, and original Forest Cover Type data is irrelevant (masked by CTGAN). Prioritize robust preprocessing and asymmetric class weighting.  

<details>
<summary>Tool Call Tracking Summary</summary>
- <code>ask_eda</code> for dataset shape/columns ✅  
- <code>ask_eda</code> for Cover_Type distribution ✅  
- <code>ask_eda</code> for test set target column presence ✅  
- <code>ask_eda</code> for test Cover_Type frequencies ✅  
- <code>ask_eda</code> for Wilderness_Area sum violations ✅  
- <code>ask_eda</code> for variable unique values (wilderness/soil types) ✅  
- <code>ask_eda</code> for numerical min/max ranges ✅  
- <code>ask_eda</code> for Wilderness_Area sum breakdown (needed for Step 2) ❌  
- <code>ask_eda</code> for Soil_Type sum breakdown (needed for Step 2) ❌  
</details>