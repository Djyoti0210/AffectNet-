import pandas as pd
import os
import hashlib

train_dir = r"d:\AffectNet+\AffectNet+\human_annotated\train_set\images"
val_dir = r"d:\AffectNet+\AffectNet+\human_annotated\validation_set\images"

train_df = pd.read_csv(r"d:\AffectNet+\AffectNet+\human_annotated\train_set\final_manifest_v1_FROZEN.csv", low_memory=False)
val_df = pd.read_csv(r"d:\AffectNet+\AffectNet+\human_annotated\validation_set\val_manifest.csv", low_memory=False)

train_ids = set(train_df['image_id'].astype(str))
val_ids = set(val_df['image_id'].astype(str))
overlap = list(train_ids.intersection(val_ids))

print(f"Overlapping IDs found: {len(overlap)}")

identical_count = 0
for img_id in overlap[:100]: # check first 100 to prove content parity
    t_path = os.path.join(train_dir, f"{img_id}.jpg")
    v_path = os.path.join(val_dir, f"{img_id}.jpg")
    
    if os.path.getsize(t_path) == os.path.getsize(v_path):
        with open(t_path, 'rb') as f:
            t_hash = hashlib.md5(f.read()).hexdigest()
        with open(v_path, 'rb') as f:
            v_hash = hashlib.md5(f.read()).hexdigest()
        
        if t_hash == v_hash:
            identical_count += 1

print(f"Sampled 100 overlapping IDs: {identical_count} were byte-for-byte identical images.")
