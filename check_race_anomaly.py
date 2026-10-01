import pandas as pd
import os
import sys

base_dir = r"d:\AffectNet+\AffectNet+\human_annotated\train_set"
manifest_path = os.path.join(base_dir, "master_manifest.csv")
report_file = os.path.join(r"d:\AffectNet+\AffectNet+", "race_report.txt")

sys.stdout = open(report_file, 'w')

print("Loading manifest...")
df = pd.read_csv(manifest_path, low_memory=False)

race_cols = ['race_asian', 'race_indian', 'race_black', 'race_white', 'race_mideast', 'race_hispanic']

# We know from earlier that there are 91 rows where sum is < 0.98 or > 1.02
race_df = df.dropna(subset=race_cols)
sums = race_df[race_cols].sum(axis=1)
outliers = race_df[(sums < 0.98) | (sums > 1.02)]

print(f"\n--- 1. & 2. ACTUAL VALUES AND SUMS FOR {len(outliers)} ROWS ---")
outlier_sums = sums.loc[outliers.index]
print(f"Min sum: {outlier_sums.min():.4f}, Max sum: {outlier_sums.max():.4f}, Mean sum: {outlier_sums.mean():.4f}")

# Print them out
print(f"\n{'ID':<10} | {'Asian':<6} | {'Indian':<6} | {'Black':<6} | {'White':<6} | {'MidEast':<7} | {'Hisp':<6} | {'SUM'}")
print("-" * 70)
for idx, row in outliers.iterrows():
    row_sum = outlier_sums.loc[idx]
    print(f"{row['image_id']:<10} | {row['race_asian']:<6.4f} | {row['race_indian']:<6.4f} | {row['race_black']:<6.4f} | {row['race_white']:<6.4f} | {row['race_mideast']:<7.4f} | {row['race_hispanic']:<6.4f} | {row_sum:.4f}")

print("\n--- 3. CROSS-TAB AGAINST HUMAN_LABEL ---")
label_counts = outliers['human_label'].value_counts().sort_index()
for label, count in label_counts.items():
    print(f"  Label {label:2}: {count} rows")

print("\nDone.")
