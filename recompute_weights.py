import pandas as pd
import numpy as np
import os
import sys

base_dir = r"d:\AffectNet+\AffectNet+\human_annotated\train_set"
input_path = os.path.join(base_dir, "final_manifest.csv")
output_path = os.path.join(base_dir, "final_manifest.csv") # overwrite
report_file = os.path.join(r"d:\AffectNet+\AffectNet+", "reweight_report.txt")

sys.stdout = open(report_file, 'w')

print("Loading final_manifest...")
df = pd.read_csv(input_path, low_memory=False)

beta = 0.9999
class_counts = df['human_label'].value_counts().sort_index()

class_names = {0: "Neutral", 1: "Happy", 2: "Sad", 3: "Surprise", 4: "Fear", 5: "Disgust", 6: "Anger", 7: "Contempt"}

effective_nums = {}
raw_weights = []
for c in range(8):
    count = class_counts.get(c, 0)
    eff_num = (1.0 - (beta ** count)) / (1.0 - beta)
    effective_nums[c] = eff_num
    raw_w = 1.0 / eff_num
    raw_weights.append(raw_w)

sum_raw = sum(raw_weights)
normalized_weights = [(w / sum_raw) * 8.0 for w in raw_weights]

print("\n--- 1. EFFECTIVE NUM & RAW WEIGHTS (beta=0.9999) ---")
print(f"{'Class':<10} | {'Count':<10} | {'Effective Num':<15} | {'Normalized Wt':<15}")
class_sample_weight_dict = {}
for c in range(8):
    class_sample_weight_dict[c] = normalized_weights[c]
    print(f"{class_names[c]:<10} | {class_counts.get(c,0):<10} | {effective_nums[c]:<15.2f} | {normalized_weights[c]:.4f}")

df['class_sample_weight'] = df['human_label'].map(class_sample_weight_dict)

print(f"\nSaving updated final_manifest...")
df.to_csv(output_path, index=False)

print("\n--- 2. EXPECTED PER-EPOCH SAMPLE COUNTS ---")
total_dataset_weight = sum([class_counts.get(c, 0) * normalized_weights[c] for c in range(8)])

print(f"{'Class':<10} | {'Raw Count':<10} | {'Exp. Count':<12} | {'Multiplier':<10}")
expected_counts = {}
for c in range(8):
    count = class_counts.get(c, 0)
    w = normalized_weights[c]
    exp_c = (count * w / total_dataset_weight) * len(df)
    expected_counts[c] = exp_c
    multiplier = exp_c / count if count > 0 else 0
    print(f"{class_names[c]:<10} | {count:<10} | {exp_c:<12.0f} | {multiplier:.2f}x")

print("\n--- 3. CONFIRMATION OF RELATIVE OVERSAMPLING ---")
happy_mult = expected_counts[1] / class_counts[1]
disgust_mult = expected_counts[5] / class_counts[5]
contempt_mult = expected_counts[7] / class_counts[7]

disgust_rel = disgust_mult / happy_mult
contempt_rel = contempt_mult / happy_mult

print(f"Happy absolute multiplier   : {happy_mult:.2f}x")
print(f"Disgust absolute multiplier : {disgust_mult:.2f}x (Relative to Happy: {disgust_rel:.2f}x)")
print(f"Contempt absolute multiplier: {contempt_mult:.2f}x (Relative to Happy: {contempt_rel:.2f}x)")

if 2.5 <= disgust_rel <= 4.0 and 2.5 <= contempt_rel <= 4.0:
    print("PASS: Both relative multipliers are in the 2.5x-4x range.")
else:
    print("WARNING: Multipliers are outside the 2.5x-4x target range!")
    
print("\nDone.")
