import pandas as pd
import os
import sys

base_dir = r"d:\AffectNet+\AffectNet+\human_annotated\train_set"
manifest_path = os.path.join(base_dir, "master_manifest.csv")
report_file = os.path.join(r"d:\AffectNet+\AffectNet+", "va_report.txt")

sys.stdout = open(report_file, 'w')

print("Loading manifest...")
df = pd.read_csv(manifest_path, low_memory=False)

# Check -2.0 for valence
val_minus_2 = df[df['valence'] == -2.0]
val_count = len(val_minus_2)
print(f"Exact count of rows with valence == -2.0: {val_count}")

# Check -2.0 for arousal
aro_minus_2 = df[df['arousal'] == -2.0]
aro_count = len(aro_minus_2)
print(f"Exact count of rows with arousal == -2.0: {aro_count}")

print("\n--- Cross-tabulation against human_label ---")

print("\nValence == -2.0 by human_label:")
val_dist = val_minus_2['human_label'].value_counts().sort_index()
for label, count in val_dist.items():
    print(f"  Label {label:2}: {count:6} rows ({(count/val_count)*100:.2f}%)")

print("\nArousal == -2.0 by human_label:")
aro_dist = aro_minus_2['human_label'].value_counts().sort_index()
for label, count in aro_dist.items():
    print(f"  Label {label:2}: {count:6} rows ({(count/aro_count)*100:.2f}%)")

print("\nCombined (Valence or Arousal == -2.0) by human_label:")
combined_minus_2 = df[(df['valence'] == -2.0) | (df['arousal'] == -2.0)]
comb_count = len(combined_minus_2)
comb_dist = combined_minus_2['human_label'].value_counts().sort_index()
for label, count in comb_dist.items():
    print(f"  Label {label:2}: {count:6} rows ({(count/comb_count)*100:.2f}%)")

print("\nDone.")
