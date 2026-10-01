import pandas as pd
import os
import sys

base_dir = r"d:\AffectNet+\AffectNet+\human_annotated\train_set"
manifest_path = os.path.join(base_dir, "master_manifest.csv")
filtered_path = os.path.join(base_dir, "filtered_manifest.csv")
report_file = os.path.join(r"d:\AffectNet+\AffectNet+", "filter_report.txt")

sys.stdout = open(report_file, 'w')

print("1. Loading master manifest and filtering...")
df = pd.read_csv(manifest_path, low_memory=False)

filtered_df = df[df['human_label'].isin([0, 1, 2, 3, 4, 5, 6, 7])]

print("2. Saving to filtered_manifest.csv...")
filtered_df.to_csv(filtered_path, index=False)

print("\n3. ROW COUNT CONFIRMATION")
print(f"Resulting row count: {len(filtered_df)}")
if len(filtered_df) == 287651:
    print("PASS: Row count is exactly 287,651")
else:
    print(f"FAIL: Row count is {len(filtered_df)} instead of 287,651")

print("\n4. CLASS BREAKDOWN CONFIRMATION")
class_names = {0: "Neutral", 1: "Happy", 2: "Sad", 3: "Surprise", 4: "Fear", 5: "Disgust", 6: "Anger", 7: "Contempt"}
expected_counts = {0: 74874, 1: 134415, 2: 25459, 3: 14090, 4: 6378, 5: 3803, 6: 24882, 7: 3750}

actual_counts = filtered_df['human_label'].value_counts().to_dict()
match = True
for label, exp_cnt in expected_counts.items():
    act_cnt = actual_counts.get(label, 0)
    name = class_names[label]
    print(f"{name:<10}: Expected {exp_cnt}, Actual {act_cnt}")
    if exp_cnt != act_cnt:
        match = False

if match:
    print("PASS: All classes match perfectly.")
else:
    print("FAIL: Class mismatches found.")

print("\n5. NULL CHECKS (pitch, valence, arousal)")
null_pitch = filtered_df['pitch'].isnull().sum()
null_valence = filtered_df['valence'].isnull().sum()
null_arousal = filtered_df['arousal'].isnull().sum()
minus2_valence = (filtered_df['valence'] == -2.0).sum()
minus2_arousal = (filtered_df['arousal'] == -2.0).sum()

print(f"Null pitch count: {null_pitch}")
print(f"Null valence count: {null_valence}")
print(f"Null arousal count: {null_arousal}")
print(f"Valence == -2.0 count: {minus2_valence}")
print(f"Arousal == -2.0 count: {minus2_arousal}")

if null_pitch == 0 and null_valence == 0 and null_arousal == 0 and minus2_valence == 0 and minus2_arousal == 0:
    print("PASS: 0% nulls and 0 rows with -2.0 sentinel values.")
else:
    print("FAIL: Unwanted nulls or sentinels found.")

print("\n6. SUBSET POPULATION CONFIRMATION")
expected_subset = {"Easy": 194254, "Challenging": 57427, "Difficult": 35970}
subset_counts = filtered_df['subset'].value_counts().to_dict()
subset_nulls = filtered_df['subset'].isnull().sum()

subset_match = True
for sub, exp_cnt in expected_subset.items():
    act_cnt = subset_counts.get(sub, 0)
    print(f"Subset {sub:<11}: Expected {exp_cnt}, Actual {act_cnt}")
    if act_cnt != exp_cnt:
        subset_match = False
print(f"Subset nulls count: {subset_nulls}")

if subset_match and subset_nulls == 0:
    print("PASS: Subset populated perfectly for all 287,651 rows with zero nulls.")
else:
    print("FAIL: Subset mismatch or nulls present.")
    
print("\nDone.")
