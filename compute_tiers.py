import pandas as pd
import numpy as np
import os
import sys

base_dir = r"d:\AffectNet+\AffectNet+\human_annotated\train_set"
filtered_path = os.path.join(base_dir, "filtered_manifest.csv")
tiered_path = os.path.join(base_dir, "tiered_manifest.csv")
report_file = os.path.join(r"d:\AffectNet+\AffectNet+", "tier_report.txt")

sys.stdout = open(report_file, 'w')

print("1. Loading filtered manifest...")
df = pd.read_csv(filtered_path, low_memory=False)

# 1. confidence_gap and top_predicted_class
soft_cols = [f"soft_label_{i}" for i in range(8)]
soft_values = df[soft_cols].values

max_soft_label = soft_values.max(axis=1)
top_predicted_class = soft_values.argmax(axis=1)

# Get own_label_score efficiently using numpy indexing
human_labels = df['human_label'].values
row_indices = np.arange(len(df))
own_label_score = soft_values[row_indices, human_labels]

df['confidence_gap'] = own_label_score - max_soft_label
df['top_predicted_class'] = top_predicted_class

# 2. quality_flag distribution and threshold
abs_yaw = df['yaw'].abs()
abs_pitch = df['pitch'].abs()

print("\n--- DISTRIBUTION OF ABS(YAW) AND ABS(PITCH) ---")
print("abs(yaw) percentiles:")
print(abs_yaw.describe(percentiles=[.5, .75, .9, .95, .99]))
print("\nabs(pitch) percentiles:")
print(abs_pitch.describe(percentiles=[.5, .75, .9, .95, .99]))

print("\nApplying threshold > 0.5 for pose_degraded...")
df['quality_flag'] = np.where((abs_yaw > 0.5) | (abs_pitch > 0.5), 'pose_degraded', 'clean')

# 3. Save to tiered_manifest.csv
print(f"\nSaving to {tiered_path}...")
df.to_csv(tiered_path, index=False)

# 4. Cross-tabulation table
print("\n--- CROSS-TABULATION ---")
bins = [-np.inf, -0.5, -0.3, -0.1, np.inf]
labels = ["below -0.5", "-0.3 to -0.5", "-0.1 to -0.3", "0 to -0.1"]
df['gap_bin'] = pd.cut(df['confidence_gap'], bins=bins, labels=labels, right=True)

cross_tab = pd.crosstab(
    index=df['subset'],
    columns=[df['quality_flag'], df['gap_bin']],
    margins=True
)
print(cross_tab)


# 5. Misclassification analysis
class_names = {0: "Neutral", 1: "Happy", 2: "Sad", 3: "Surprise", 4: "Fear", 5: "Disgust", 6: "Anger", 7: "Contempt"}

print("\n--- TOP PREDICTED CLASS CONFUSION (WHEN WRONG) ---")
wrong_df = df[df['human_label'] != df['top_predicted_class']]
for label in range(8):
    label_name = class_names[label]
    wrong_this_label = wrong_df[wrong_df['human_label'] == label]
    
    if len(wrong_this_label) == 0:
        print(f"Human: {label_name:<10} -> Perfect match, no mistakes.")
        continue
        
    top_mistakes = wrong_this_label['top_predicted_class'].value_counts()
    
    mistake_strs = []
    for pred_idx, count in top_mistakes.head(3).items():
        mistake_strs.append(f"{class_names[pred_idx]}: {count}")
    
    print(f"Human: {label_name:<10} -> Most confused with: {', '.join(mistake_strs)}")

print("\nDone.")
