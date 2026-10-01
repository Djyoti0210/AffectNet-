import pandas as pd
import numpy as np
import os
import sys

base_dir = r"d:\AffectNet+\AffectNet+\human_annotated\train_set"
input_path = os.path.join(base_dir, "tiered_manifest.csv")
output_path = os.path.join(base_dir, "tier_assigned_manifest.csv")
report_file = os.path.join(r"d:\AffectNet+\AffectNet+", "tier_assignment_report.txt")

sys.stdout = open(report_file, 'w')

print("1. Loading tiered manifest...")
df = pd.read_csv(input_path, low_memory=False)

# Define rules
conditions = [
    df['subset'] == "Easy",
    df['subset'] == "Challenging",
    (df['subset'] == "Difficult") & (df['confidence_gap'] >= -0.3),
    (df['subset'] == "Difficult") & (df['confidence_gap'] < -0.3)
]

choices = ["gold", "standard", "soft_weighted", "high_risk"]

df['training_tier'] = np.select(conditions, choices, default="unknown")

print("2. Saving to tier_assigned_manifest.csv...")
df.to_csv(output_path, index=False)

print("\n--- FINAL ROW COUNT PER TRAINING TIER ---")
tier_counts = df['training_tier'].value_counts()
for tier in choices:
    count = tier_counts.get(tier, 0)
    print(f"{tier:<15}: {count} rows")

if (df['training_tier'] == "unknown").sum() > 0:
    print(f"WARNING: Unknown tiers found: {(df['training_tier'] == 'unknown').sum()}")

print("\n--- PER-CLASS COUNTS WITHIN EACH TRAINING TIER ---")
class_names = {0: "Neutral", 1: "Happy", 2: "Sad", 3: "Surprise", 4: "Fear", 5: "Disgust", 6: "Anger", 7: "Contempt"}

# Create cross-tabulation
crosstab = pd.crosstab(
    index=df['human_label'],
    columns=df['training_tier'],
    margins=False
)

# Rename index to class names
crosstab.index = [class_names[idx] for idx in crosstab.index]

# Ensure all columns exist and order them
crosstab = crosstab.reindex(columns=choices, fill_value=0)

# Print a formatted table with percentages
print(f"{'Class':<10} | {'gold':<10} | {'standard':<10} | {'soft_weighted':<15} | {'high_risk':<10} | {'% high_risk':<12}")
print("-" * 80)

for cls_name in class_names.values():
    if cls_name not in crosstab.index:
        continue
    row = crosstab.loc[cls_name]
    total_class = row.sum()
    hr_count = row['high_risk']
    hr_pct = (hr_count / total_class) * 100 if total_class > 0 else 0
    
    print(f"{cls_name:<10} | {row['gold']:<10} | {row['standard']:<10} | {row['soft_weighted']:<15} | {hr_count:<10} | {hr_pct:>5.1f}%")

print("\nDone.")
