import pandas as pd
import os
import glob
import json
import random
import sys
import numpy as np

base_dir = r"d:\AffectNet+\AffectNet+\human_annotated\train_set"
manifest_path = os.path.join(base_dir, "master_manifest.csv")
images_dir = os.path.join(base_dir, "images")
annotations_dir = os.path.join(base_dir, "annotations")
report_file = os.path.join(r"d:\AffectNet+\AffectNet+", "report.txt")

sys.stdout = open(report_file, 'w')

print("Loading manifest...")
df = pd.read_csv(manifest_path, low_memory=False)

print("\n--- 1. ROW COUNT ---")
actual_rows = len(df)
expected_rows = 414799
print(f"Expected rows: {expected_rows}")
print(f"Actual rows: {actual_rows}")
if actual_rows == expected_rows:
    print("Result: PASS")
    status_1 = "PASS"
else:
    print(f"Result: FAIL (Diff: {actual_rows - expected_rows})")
    status_1 = "FAIL"


print("\n--- 2. IMAGE-JSON PARITY ---")
img_files = glob.glob(os.path.join(images_dir, "*.*"))
num_img_files = len(img_files)
print(f"a. Images in folder: {num_img_files}")

# Check parity using sets for speed
csv_images_set = set([os.path.basename(str(p)) for p in df['image_path']])
img_files_set = set([os.path.basename(f) for f in img_files])
missing_files = csv_images_set - img_files_set

print(f"b. Missing files: {len(missing_files)}")
if len(missing_files) > 0:
    print(f"   Missing files sample: {list(missing_files)[:10]}")
    
dup_ids = df['image_id'].duplicated().sum()
dup_paths = df['image_path'].duplicated().sum()
print(f"c. Duplicate image_ids: {dup_ids}")
print(f"   Duplicate image_paths: {dup_paths}")

if num_img_files == 414799 and len(missing_files) == 0 and dup_ids == 0 and dup_paths == 0:
    print("Result: PASS")
    status_2 = "PASS"
else:
    print("Result: FAIL")
    status_2 = "FAIL"


print("\n--- 3. LABEL DISTRIBUTION CHECK ---")
expected_counts = {
    0: 75374, 1: 134915, 2: 25959, 3: 14590, 4: 6878, 5: 4303, 6: 25382, 7: 4250,
    8: 33588, 9: 12145, 10: 82915
}
actual_counts = df['human_label'].value_counts().to_dict()
print(f"{'Label':<10} | {'Expected':<10} | {'Actual':<10} | {'Diff %':<10}")
label_fail = False
for label, expected in expected_counts.items():
    actual = actual_counts.get(label, 0)
    diff = abs(actual - expected)
    diff_pct = (diff / expected) * 100 if expected > 0 else 0
    print(f"{label:<10} | {expected:<10} | {actual:<10} | {diff_pct:.2f}%")
    if diff_pct > 1.0:
        label_fail = True

status_3 = "FAIL" if label_fail else "PASS"
print(f"Result: {status_3}")


print("\n--- 4. SOFT-LABEL SANITY ---")
soft_cols = [f"soft_label_{i}" for i in range(8)]
real_expr_df = df[df['human_label'].isin(range(8))]
non_expr_df = df[df['human_label'].isin([8,9,10])]

missing_softs = real_expr_df[soft_cols].isnull().any(axis=1).sum()
print(f"a. Rows with 0-7 human_label but missing soft-labels: {missing_softs}")

sums = real_expr_df[soft_cols].sum(axis=1)
not_normalized = ((sums < 0.98) | (sums > 1.02)).sum()
print(f"b. Rows not summing to 1.0 (±0.02): {not_normalized}")
print(f"   Sum min: {sums.min():.4f}, max: {sums.max():.4f}, mean: {sums.mean():.4f}")

nulls_in_non_expr = non_expr_df[soft_cols].isnull().all(axis=1).sum()
zeroes_in_non_expr = (non_expr_df[soft_cols].sum(axis=1) == 0).sum()
print(f"c. Non-expression rows (8,9,10) total: {len(non_expr_df)}")
print(f"   Rows with entirely null soft_labels: {nulls_in_non_expr}")
print(f"   Rows where soft_labels sum to 0: {zeroes_in_non_expr}")
print(f"   Rows with populated soft_labels (sum>0): {len(non_expr_df) - nulls_in_non_expr - zeroes_in_non_expr}")

mins = real_expr_df[soft_cols].min().min()
maxs = real_expr_df[soft_cols].max().max()
means = real_expr_df[soft_cols].mean().mean()
print(f"d. Soft-label range (real expr): min={mins:.4f}, max={maxs:.4f}, mean={means:.4f}")

status_4 = "FAIL" if missing_softs > 0 or mins < 0 or maxs > 1 else "PASS"
print(f"Result: {status_4}")


print("\n--- 5. SUBSET DISTRIBUTION CHECK ---")
exp_subset = {"Easy": 194254, "Challenging": 57427, "Difficult": 35970}
subset_counts = real_expr_df['subset'].value_counts().to_dict()
print("Real expression (0-7) subset distribution:")
print(f"{'Subset':<15} | {'Expected':<10} | {'Actual':<10} | {'Diff':<10}")
subset_fail = False
for k, exp in exp_subset.items():
    act = subset_counts.get(k, 0)
    print(f"{k:<15} | {exp:<10} | {act:<10} | {act-exp:<10}")
    if abs(act-exp) > 0:
        subset_fail = True

none_str_count = (non_expr_df['subset'] == 'None').sum()
null_count = non_expr_df['subset'].isnull().sum()
print(f"Non-expression subset exactly 'None': {none_str_count}, actual null: {null_count}")

status_5 = "FAIL" if subset_fail else "PASS"
print(f"Result: {status_5}")


print("\n--- 6. METADATA COMPLETENESS ---")
meta_cols = ['age', 'valence', 'arousal', 'gender_male_prob', 'gender_female_prob',
             'race_asian', 'race_indian', 'race_black', 'race_white', 'race_mideast', 'race_hispanic',
             'yaw', 'pitch', 'roll']
meta_fail = False
for col in meta_cols:
    null_pct = df[col].isnull().sum() / len(df) * 100
    print(f"{col:<20}: {null_pct:.2f}% null")
    if null_pct > 1.0:
        meta_fail = True
status_6 = "FAIL" if meta_fail else "PASS"
print(f"Result: {status_6}")


print("\n--- 7. RANGE SANITY CHECKS ---")
range_fail = False
val_min, val_max = df['valence'].min(), df['valence'].max()
aro_min, aro_max = df['arousal'].min(), df['arousal'].max()
print(f"Valence range: [{val_min:.4f}, {val_max:.4f}]")
if val_min < -1 or val_max > 1: range_fail = True
print(f"Arousal range: [{aro_min:.4f}, {aro_max:.4f}]")
if aro_min < -1 or aro_max > 1: range_fail = True

y_min, y_max = df['yaw'].min(), df['yaw'].max()
p_min, p_max = df['pitch'].min(), df['pitch'].max()
r_min, r_max = df['roll'].min(), df['roll'].max()
print(f"Yaw range:   [{y_min:.4f}, {y_max:.4f}]")
print(f"Pitch range: [{p_min:.4f}, {p_max:.4f}]")
print(f"Roll range:  [{r_min:.4f}, {r_max:.4f}]")
if any(v < -2 or v > 2 for v in [y_min, y_max, p_min, p_max, r_min, r_max]): 
    # Usually yaw/pitch/roll shouldn't exceed pi/2, but checking bounds around 1-2
    pass

age_min, age_max = df['age'].min(), df['age'].max()
print(f"Age range: [{age_min:.2f}, {age_max:.2f}]")
if age_min < 0 or age_max > 120: range_fail = True

# Dropna for gen sums
valid_gen = df[['gender_male_prob', 'gender_female_prob']].dropna()
gen_sums = valid_gen['gender_male_prob'] + valid_gen['gender_female_prob']
gen_sum_outliers = ((gen_sums < 0.98) | (gen_sums > 1.02)).sum()
print(f"Rows where gender prob sum != 1.0: {gen_sum_outliers}")

valid_race = df[['race_asian', 'race_indian', 'race_black', 'race_white', 'race_mideast', 'race_hispanic']].dropna()
race_sums = valid_race.sum(axis=1)
race_sum_outliers = ((race_sums < 0.98) | (race_sums > 1.02)).sum()
print(f"Rows where race prob sum != 1.0: {race_sum_outliers}")

if gen_sum_outliers > 0 or race_sum_outliers > 0: range_fail = True
status_7 = "FAIL" if range_fail else "PASS"
print(f"Result: {status_7}")


print("\n--- 8. LANDMARKS CHECK ---")
empty_lms = (df['landmarks_68'].isnull() | (df['landmarks_68'] == '[]')).sum()
print(f"b. Rows with null/empty landmarks: {empty_lms}")

lm_fail = False
sample_lms = df[df['landmarks_68'].notnull() & (df['landmarks_68'] != '[]')].sample(min(100, len(df)))['landmarks_68']
valid_len_count = 0
for lm_str in sample_lms:
    try:
        arr = json.loads(lm_str)
        if len(arr) == 136:
            valid_len_count += 1
    except:
        pass
print(f"a. Valid length (136) in 100 sample: {valid_len_count}")
if valid_len_count < len(sample_lms):
    lm_fail = True
status_8 = "FAIL" if lm_fail else "PASS"
print(f"Result: {status_8}")


print("\n--- 9. SPOT-CHECK AGAINST RAW JSON ---")
spot_check_fail = False
sample_ids = df['image_id'].sample(20).tolist()
for sid in sample_ids:
    row = df[df['image_id'] == sid].iloc[0]
    with open(os.path.join(annotations_dir, f"{sid}.json"), 'r') as f:
        data = json.load(f)
    print(f"\nImage ID: {sid}")
    print(f"JSON label: {data.get('human-label')} | CSV label: {row['human_label']}")
    if data.get('human-label') != row['human_label']:
        spot_check_fail = True

status_9 = "FAIL" if spot_check_fail else "PASS"
print(f"Result: {status_9}")

print("\n--- 10. FINAL REPORT ---")
print(f"{'Check Name':<30} | {'Pass/Fail':<10} | {'Notes'}")
print("-" * 60)
print(f"{'1. Row Count':<30} | {status_1:<10} |")
print(f"{'2. Image-JSON Parity':<30} | {status_2:<10} |")
print(f"{'3. Label Distribution':<30} | {status_3:<10} |")
print(f"{'4. Soft-Label Sanity':<30} | {status_4:<10} |")
print(f"{'5. Subset Distribution':<30} | {status_5:<10} |")
print(f"{'6. Metadata Completeness':<30} | {status_6:<10} |")
print(f"{'7. Range Sanity Checks':<30} | {status_7:<10} |")
print(f"{'8. Landmarks Check':<30} | {status_8:<10} |")
print(f"{'9. Spot-Check':<30} | {status_9:<10} |")
