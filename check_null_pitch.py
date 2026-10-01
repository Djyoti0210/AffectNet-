import pandas as pd
import os
import json
import sys

base_dir = r"d:\AffectNet+\AffectNet+\human_annotated\train_set"
manifest_path = os.path.join(base_dir, "master_manifest.csv")
annotations_dir = os.path.join(base_dir, "annotations")
report_file = os.path.join(r"d:\AffectNet+\AffectNet+", "pitch_report.txt")

sys.stdout = open(report_file, 'w')

print("Loading manifest...")
df = pd.read_csv(manifest_path, low_memory=False)

null_pitch_df = df[df['pitch'].isnull()]
print(f"Total rows with null pitch: {len(null_pitch_df)} out of {len(df)}")

print("\n--- 1. RAW JSON PARITY FOR NULL PITCH ---")
sample_df = null_pitch_df.sample(min(200, len(null_pitch_df)), random_state=42)
absent_count = 0
explicit_null_count = 0
parser_bug_count = 0

for _, row in sample_df.iterrows():
    sid = row['image_id']
    with open(os.path.join(annotations_dir, f"{sid}.json"), 'r') as f:
        data = json.load(f)
        
    meta = data.get("meta-data", {})
    pose = meta.get("pose", {})
    
    if "pitch" not in pose:
        absent_count += 1
    elif pose["pitch"] is None:
        explicit_null_count += 1
    else:
        # Wait, if pitch is present and not null, then it's a parser bug
        parser_bug_count += 1

print(f"Out of {len(sample_df)} sampled JSONs:")
print(f"  'pitch' key absent: {absent_count}")
print(f"  'pitch' key explicitly null: {explicit_null_count}")
print(f"  'pitch' present with valid value (Parser Bug): {parser_bug_count}")


print("\n--- 2. CORRELATIONS OVER ENTIRE DATASET ---")

print("\na. Correlation with Label (Human Label Distribution):")
print("Label | Null-Pitch Count | % of Null-Pitch | % of Total dataset")
total_null = len(null_pitch_df)
null_label_dist = null_pitch_df['human_label'].value_counts()
overall_label_dist = df['human_label'].value_counts()
for label, count in null_label_dist.items():
    pct_of_null = (count / total_null) * 100
    pct_of_total = (overall_label_dist[label] / len(df)) * 100
    print(f"{label:5} | {count:16} | {pct_of_null:14.2f}% | {pct_of_total:17.2f}%")

print("\nb. Correlation with Yaw values:")
yaw_null = null_pitch_df['yaw'].dropna()
yaw_overall = df['yaw'].dropna()
print(f"Null-Pitch rows Yaw -> min: {yaw_null.min():.4f}, max: {yaw_null.max():.4f}, mean: {yaw_null.mean():.4f}, std: {yaw_null.std():.4f}")
print(f"Overall dataset Yaw -> min: {yaw_overall.min():.4f}, max: {yaw_overall.max():.4f}, mean: {yaw_overall.mean():.4f}, std: {yaw_overall.std():.4f}")

# Check extreme yaw counts
extreme_yaw_null = len(null_pitch_df[null_pitch_df['yaw'].abs() > 0.5])
extreme_yaw_total = len(df[df['yaw'].abs() > 0.5])
print(f"Rows with |yaw| > 0.5 in Null-Pitch: {extreme_yaw_null} ({(extreme_yaw_null/total_null)*100:.2f}%)")
print(f"Rows with |yaw| > 0.5 in Overall: {extreme_yaw_total} ({(extreme_yaw_total/len(df))*100:.2f}%)")

print("\nc. Image ID Pattern (Batch analysis):")
# We'll see if the image_ids for null-pitch are clustered.
null_ids_sorted = null_pitch_df['image_id'].astype(int).sort_values().values
print(f"Min image ID: {null_ids_sorted.min()}, Max image ID: {null_ids_sorted.max()}")
# Are they clustered in a specific range?
q25 = pd.Series(null_ids_sorted).quantile(0.25)
q75 = pd.Series(null_ids_sorted).quantile(0.75)
print(f"25th percentile ID: {q25:.0f}")
print(f"75th percentile ID: {q75:.0f}")
overall_q25 = df['image_id'].astype(int).quantile(0.25)
overall_q75 = df['image_id'].astype(int).quantile(0.75)
print(f"Overall 25th percentile ID: {overall_q25:.0f}")
print(f"Overall 75th percentile ID: {overall_q75:.0f}")

print("\nDone.")
