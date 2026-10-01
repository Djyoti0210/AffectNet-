import pandas as pd
import numpy as np
import os
import sys

base_dir = r"d:\AffectNet+\AffectNet+\human_annotated\train_set"
input_path = os.path.join(base_dir, "tier_assigned_manifest.csv")
output_path = os.path.join(base_dir, "final_manifest.csv")
report_file = os.path.join(r"d:\AffectNet+\AffectNet+", "weight_report.txt")

sys.stdout = open(report_file, 'w')

print("1. Loading tier_assigned_manifest...")
df = pd.read_csv(input_path, low_memory=False)

# 1. class_sample_weight
beta = 0.999
class_counts = df['human_label'].value_counts().sort_index()

class_sample_weight_dict = {}
raw_weights = []
for c in range(8):
    count = class_counts.get(c, 0)
    # effective_num = 1 - beta^count
    effective_num = 1.0 - (beta ** count)
    raw_w = (1.0 - beta) / effective_num if effective_num > 0 else 0
    raw_weights.append(raw_w)

# Normalize raw_weight across all 8 classes so they sum to 8 (average = 1)
sum_raw = sum(raw_weights)
normalized_weights = [(w / sum_raw) * 8.0 for w in raw_weights]

class_names = {0: "Neutral", 1: "Happy", 2: "Sad", 3: "Surprise", 4: "Fear", 5: "Disgust", 6: "Anger", 7: "Contempt"}

print("\n--- 1. CLASS SAMPLE WEIGHTS ---")
print(f"{'Class':<10} | {'Count':<10} | {'Weight':<10}")
for c in range(8):
    class_sample_weight_dict[c] = normalized_weights[c]
    print(f"{class_names[c]:<10} | {class_counts.get(c,0):<10} | {normalized_weights[c]:.4f}")

# Map weights to df
df['class_sample_weight'] = df['human_label'].map(class_sample_weight_dict)

# 2. tier_loss_weight
tier_weights = {
    "gold": 1.0,
    "standard": 1.0,
    "soft_weighted": 0.5,
    "high_risk": 0.15,
    "unknown": 1.0
}
df['tier_loss_weight'] = df['training_tier'].map(tier_weights)

# Save to final_manifest.csv
print(f"\nSaving to {output_path}...")
df.to_csv(output_path, index=False)

# 5. Expected per-epoch sample counts
print("\n--- 5. EXPECTED PER-EPOCH SAMPLE COUNTS ---")
# Prob for picking class c = (count * weight) / total_weight
total_dataset_weight = sum([class_counts.get(c, 0) * normalized_weights[c] for c in range(8)])

print(f"{'Class':<10} | {'Raw Count':<10} | {'Exp. Count':<12} | {'Multiplier':<10}")
for c in range(8):
    count = class_counts.get(c, 0)
    w = normalized_weights[c]
    expected_count = (count * w / total_dataset_weight) * len(df)
    multiplier = expected_count / count if count > 0 else 0
    print(f"{class_names[c]:<10} | {count:<10} | {expected_count:<12.0f} | {multiplier:.2f}x")
    
print("\nDone.")
