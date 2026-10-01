import pandas as pd
import os
import hashlib
import time

train_dir = r"d:\AffectNet+\AffectNet+\human_annotated\train_set\images"
val_dir = r"d:\AffectNet+\AffectNet+\human_annotated\validation_set\images"

train_df = pd.read_csv(r"d:\AffectNet+\AffectNet+\human_annotated\train_set\final_manifest_v1_FROZEN.csv", low_memory=False)
val_df = pd.read_csv(r"d:\AffectNet+\AffectNet+\human_annotated\validation_set\val_manifest.csv", low_memory=False)

train_df['image_id_str'] = train_df['image_id'].astype(str)
val_df['image_id_str'] = val_df['image_id'].astype(str)

train_ids = set(train_df['image_id_str'])
val_ids = set(val_df['image_id_str'])
overlap = list(train_ids.intersection(val_ids))

print(f"Total overlapping IDs checked: {len(overlap)}")

# Index dataframes for O(1) lookup
train_df_indexed = train_df.set_index('image_id_str')
val_df_indexed = val_df.set_index('image_id_str')
soft_cols = [f"soft_label_{i}" for i in range(8)]

identical_hashes = []
suspicious_metadata = []

start_time = time.time()

for img_id in overlap:
    t_path = os.path.join(train_dir, f"{img_id}.jpg")
    v_path = os.path.join(val_dir, f"{img_id}.jpg")
    
    t_hash = "not_checked"
    v_hash = "not_checked"
    
    # Only hash if files exist and are the same size (saves massive amounts of time)
    if os.path.exists(t_path) and os.path.exists(v_path):
        if os.path.getsize(t_path) == os.path.getsize(v_path):
            with open(t_path, 'rb') as f:
                t_hash = hashlib.md5(f.read()).hexdigest()
            with open(v_path, 'rb') as f:
                v_hash = hashlib.md5(f.read()).hexdigest()
            if t_hash == v_hash:
                identical_hashes.append(img_id)
        else:
            t_hash = "size_A"
            v_hash = "size_B"

    # Metadata Sanity Layer
    t_row = train_df_indexed.loc[img_id]
    v_row = val_df_indexed.loc[img_id]
    
    # Handle edge case where ID might somehow be duplicated in manifest
    if isinstance(t_row, pd.DataFrame): t_row = t_row.iloc[0]
    if isinstance(v_row, pd.DataFrame): v_row = v_row.iloc[0]
    
    t_label = t_row['human_label']
    v_label = v_row['human_label']
    
    # fillna(-1) ensures NaN == NaN logic holds for lists
    t_soft = t_row[soft_cols].fillna(-1).values.tolist()
    v_soft = v_row[soft_cols].fillna(-1).values.tolist()
    
    # If the images are DIFFERENT (hashes don't match) but the metadata is mathematically IDENTICAL
    if (t_hash != v_hash) and (t_label == v_label) and (t_soft == v_soft):
        suspicious_metadata.append(img_id)

print(f"\nTime taken: {time.time() - start_time:.2f} seconds")
print(f"Exact count of byte-for-byte identical hash matches: {len(identical_hashes)}")
if identical_hashes:
    print(f"-> IDENTICAL MATCHES: {identical_hashes}")

print(f"\nSuspicious Metadata Count (Different image, identically matched hard/soft labels): {len(suspicious_metadata)}")
if suspicious_metadata:
    print(f"-> SUSPICIOUS IDs: {suspicious_metadata[:25]}")
    if len(suspicious_metadata) > 25:
        print(f"   ... and {len(suspicious_metadata)-25} more.")
