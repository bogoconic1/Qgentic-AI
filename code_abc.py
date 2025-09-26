data_path = "task/statoil-iceberg-classifier-challenge"
# Load test.json, create is_na flag, and basic dataset checks

import json
import os
import pandas as pd
import numpy as np

# Read test.json
test_path = os.path.join(data_path, "test.json")
with open(test_path, "r") as f:
    test_data = json.load(f)

df_test = pd.DataFrame(test_data)

# Create is_na flag for inc_angle; treat 'na', None, empty, or non-convertible as NA
def is_non_numeric_angle(val):
    if val is None:
        return True
    try:
        # Some values might already be float; others strings
        _ = float(val)
        return False
    except Exception:
        return True

df_test["is_na"] = df_test["inc_angle"].apply(is_non_numeric_angle)

print(f"Loaded test.json with {len(df_test)} records.")
print("Counts by is_na (inc_angle non-numeric):")
print(df_test["is_na"].value_counts(dropna=False).to_string())
print("\nSample inc_angle values per group:")
print("is_na=True:", df_test.loc[df_test["is_na"]==True, "inc_angle"].head(5).to_list())
print("is_na=False:", df_test.loc[df_test["is_na"]==False, "inc_angle"].head(5).to_list())

data_path = "task/statoil-iceberg-classifier-challenge"
# Compute per-image stats for band_1 and band_2

# Ensure bands are lists of numbers; compute mean and std per row
def band_stats(band_list):
    arr = np.asarray(band_list, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))

# Compute stats and assign to new columns
means_b1, stds_b1, means_b2, stds_b2 = [], [], [], []
for b1, b2 in zip(df_test["band_1"], df_test["band_2"]):
    m1, s1 = band_stats(b1)
    m2, s2 = band_stats(b2)
    means_b1.append(m1); stds_b1.append(s1)
    means_b2.append(m2); stds_b2.append(s2)

df_test["mean_b1"] = means_b1
df_test["std_b1"]  = stds_b1
df_test["mean_b2"] = means_b2
df_test["std_b2"]  = stds_b2

print("Computed per-image statistics (mean/std) for both bands.")
print(df_test[["id", "is_na", "mean_b1", "std_b1", "mean_b2", "std_b2"]].head(3).to_string(index=False))

data_path = "task/statoil-iceberg-classifier-challenge"
# Group means for is_na=True vs False and differences

group_means = df_test.groupby("is_na")[["mean_b1", "std_b1", "mean_b2", "std_b2"]].mean()

# Prepare differences: True minus False
if True in group_means.index and False in group_means.index:
    diffs = group_means.loc[True] - group_means.loc[False]
else:
    diffs = pd.Series(dtype=float)

print("Group means by is_na:")
print(group_means.to_string())

print("\nDifferences (is_na=True minus is_na=False):")
print(diffs.to_string())

# Also print counts per group to contextualize
counts = df_test["is_na"].value_counts()
print("\nCounts per is_na group:")
print(counts.to_string())