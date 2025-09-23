# Technical Plan for Developer

## 📌 Step 1: Process Soil_Type Columns into Valid Categorical Feature
*Evidence*:  
- In train.csv, Soil_Type columns sum to: 0 (41.4%), 1 (39.05%), 2 (15.47%), and higher (sum up to 7)  
- In original `covtype.csv` dataset (real Forest Cover Type data), **every row has exactly one Soil_Type column = 1** (sum=1) for all 581,012 rows  
- **Critical discrepancy**: Synthetic data has 41.4% rows with Soil_Type sum=0 (invalid "no soil type" regions) and 19.55% with multiple soil types (structurally impossible for real data)

*Action*:  
- Create new feature `Soil_Type_Clean` by processing all 40 Soil_Type columns:  
  - For sum=0 → Assign as `0` (Unknown)  
  - For sum>1 → Take the first column with `1` (or highest column index)  
  - For sum=1 → Use the observed column index  
- Filter out invalid rows empirically (e.g., rows where `Soil_Type_Clean` is considered unreliable)  
- Validate consistency between train/test distributions for `Soil_Type_Clean` (e.g., via KS test)

*Rationale*:  
Synthetic GAN corrupted the one-hot encoding structure. Poorly encoded soil types will prevent models from learning meaningful patterns. The original data proves valid soil types must be mutually exclusive.

*Risks*:  
- Incorrect assignment (e.g., multiple 1s → first column) may lose signal; validate via feature importance  
- Unknown soil types (sum=0) might correlate with weak-to-predict cover types  

---

## 📌 Step 2: Address Severe Target Class Imbalance
*Evidence*:  
- Train target distribution:  
  - Class 2: **56.56%** (2,036,254 samples)  
  - Class 1: 36.69% (1,320,866)  
  - Class 3: 4.89% (176,184)  
  - **Class 4: 0.01% (333 samples)**  
  - **Class 5: 0.0% (1 sample)**  
  - Class 6: 0.28% (10,237)  
  - Class 7: 1.56% (56,125)  
- Original `covtype.csv` has more balanced rare classes (Class 4: 0.47%, Class 5: 1.63%) → Synthetic data fails to replicate rare classes accurately

*Action*:  
- Apply **class-weighted loss** with weights inversely proportional to class frequencies  
- Use **stratified K-fold validation** to ensure rare classes are represented in every fold  
- For Classes 4/5 (near-zero samples), consider:  
  - Abandoning class-specific predictions (since model cannot learn from 1 sample)  
  - Focusing on predicting top 5 classes and treating others as "miscellaneous"  
- *Avoid SMOTE*: Won't work for Class 5 (only 1 example)

*Rationale*:  
Without handling imbalance, models will bias toward Class 2 (achieves ~56% accuracy trivially), so we must explicitly prioritize minority classes via weighting. Exception: Class 5 is so scarce it cannot be reliably predicted.

*Risks*:  
- Models will still ignore rare classes unless weighted aggressively  
- Test data might not contain Class 4/5 (competition could intentionally drop these)  

---

## 📌 Step 3: Validate Train-Test Distribution Shift
*Evidence*:  
- **Elevation**: Train (mean=2980.15, std=289.04) vs Test (mean=2980.59, std=289.09) → No shift  
- **Wilderness_Area1**: Train=940,369 ones vs Test=104,403 ones (scaled similarly)  
- **Wilderness_Area3**: Train=2,352,891 vs Test=261,402 (massive gap but no shift in distribution)  
- Similar for other features (Aspect, Slope, etc.) preliminarily  
- *Missing insight:* No full feature-wise analysis done (e.g., KS tests for all 50+ features)

*Action*:  
- Run **KS test for all continuous features** (Elevation, Aspect, Slope, Distances, Hillshade)  
- Run **chi-squared tests for categorical features** (Wilderness_Area, Soil_Type_Clean)  
- Flag any feature with p-value < 0.01 for shift  
- For shifted features:  
  - Normalize using train distribution statistics  
  - Check if sampling method (e.g., stratification) can reconcile shifts  

*Rationale*:  
Synthetic data may have unintentional shifts despite similar means. Critical if Hillshade or distance features differ (e.g., test instances are uniformly higher elevation).

*Risks*:  
- Undetected shift in internal features (e.g., correlation between Soil_Type and Elevation) → model generalization failure  

---

## 📌 Step 4: Engineer Key Domain-Specific Features
*Evidence*:  
- Topographic features (Elevation, Slope, Distances) are known predictors of forest cover types  
- Original Forest Cover Type datasets show:  
  - Euclidean distance to hydrology is more meaningful than horizontal/vertical separately  
  - Hillshade patterns change hourly with sun angle (3 channels → likely 2D feature)  

*Action*:  
- **Distance Features**:  
  - Add `Hydrology_Dist = sqrt(Horizontal_Distance_To_Hydrology² + Vertical_Distance_To_Hydrology²)`  
  - Add `Fire_Road_Dist = Horizontal_Distance_To_Fire_Points + Horizontal_Distance_To_Roadways`  
- **Hillshade**:  
  - Apply PCA to reduce 3 hillshade columns to 1-2 components capturing sunrise/sunset patterns  
- **Interaction Features**:  
  - `Soil_Type_Clean * Wilderness_Area` (e.g., one-hot interaction)  
  - `Elevation_Band = 100 * floor(Elevation / 100)` (binning for model stability)  

*Rationale*:  
Domain knowledge confirms these transformations improve signal extraction. Synthetic data is based on real-world properties, so these features should generalize.

*Risks*:  
- Over-engineering noisy interactions (e.g., Soil_Type * Elevation) → verify via feature importance after training  

---

## 📌 Step 5: Final Model Strategy & Leakage Check
*Evidence*:  
- No missing values or obvious ID leakage (all IDs are sequential)  
- Wilderness_Area and Soil_Type features show expected patterns *after cleaning*  
- Target classes are not perfectly predictable by any single feature  

*Action*:  
- **Primary Model**: LightGBM/XGBoost with:  
  - `class_weight = 'balanced'` or custom inverse-frequency weights  
  - Early stopping for robustness  
  - Stratified sampling to preserve class ratios in validation  
- **Leakage checks**:  
  - Compute mutual information between all features and target  
  - Confirm no feature has >0.95 MI with target (would indicate unintended leakage)  
  - Check if train/test id sequences correlate with target (unlikely but trivial check)  
- **Validation**:  
  - Use **stratified 5-fold CV** (not holdout) to capture rare class behavior  
  - Report per-class accuracy (not just overall)  

*Rationale*:  
Tree-based models robustly handle feature scaling and interactions. Strict leakage checks prevent accidental use of training-time information.

*Risks*:  
- Synthetic data may contain hidden non-physical correlations → stress-test models on random perturbations  

---

## 🔍 Critical Missing Evidence to Collect
1. **Exact test set target distribution** (only possible via competition leaderboard)  
2. **Full feature-wise KS/chi-squared tests** for train/test shifts (not just Elevation)  
3. **Impact of Soil_Type_Clean cleaning** on model performance (test multiple cleaning strategies)  
4. **Rare class inclusion in test data** – check if Class 4/5 exist in public test set  

## ✅ Next Steps for Developer
1. Process Soil_Type and validate with cleanup  
2. Run full train-test distribution checks  
3. Initialize LightGBM with weighted classes and stratified CV  
4. Prioritize features with highest MI scores for signal extraction  

*Critical Reminder*: For Class 5 (only 1 train sample), do not waste effort trying to predict it – focus on Classes 1-4,6-7. Accuracy will have intrinsic limits near 60-70% due to imbalance.