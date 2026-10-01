import cv2
import pandas as pd
import numpy as np
import os
import hashlib
import concurrent.futures

train_df = pd.read_csv(r"d:\AffectNet+\AffectNet+\human_annotated\train_set\final_manifest_v1_FROZEN.csv")
val_df = pd.read_csv(r"d:\AffectNet+\AffectNet+\human_annotated\validation_set\val_manifest.csv")

train_dir = r"d:\AffectNet+\AffectNet+\human_annotated\train_set"
val_dir = r"d:\AffectNet+\AffectNet+\human_annotated\validation_set"

def compute_dhash(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: return 0
    resized = cv2.resize(img, (9, 8))
    diff = resized[:, 1:] > resized[:, :-1]
    return sum([int(v) << i for i, v in enumerate(diff.flatten())])

def get_md5(p):
    with open(p, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

print("Loading hashes...")
# Load near dup validation ids
with open("near_dup_ids.txt", "r") as f:
    near_val_ids = [int(x) for x in f.read().splitlines() if x.strip()]

# We need the dhashes of these near val ids and their matched train images.
# Actually we can just hash all near val ids, and then find their matches in train.
val_hashes = []
for idx in near_val_ids:
    val_hashes.append(compute_dhash(os.path.join(val_dir, val_df.iloc[idx]['image_path'])))

val_hashes = np.array(val_hashes, dtype=np.uint64)

train_paths = [os.path.join(train_dir, p) for p in train_df['image_path']]
print("Computing train dhashes...")
with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
    train_hashes = list(executor.map(compute_dhash, train_paths))
train_hashes = np.array(train_hashes, dtype=np.uint64)

pairs = []
print("Finding pairs...")
for i, vh in enumerate(val_hashes):
    if len(pairs) >= 8:
        break
    xor = np.bitwise_xor(train_hashes, vh)
    xor_bytes = xor.view(np.uint8)
    bits = np.unpackbits(xor_bytes.reshape(-1, 8), axis=1)
    distances = bits.sum(axis=1)
    
    matches = np.where((distances >= 1) & (distances <= 4))[0]
    for m in matches:
        val_idx = near_val_ids[i]
        train_idx = m
        
        # Check MD5
        v_md5 = get_md5(os.path.join(val_dir, val_df.iloc[val_idx]['image_path']))
        t_md5 = get_md5(os.path.join(train_dir, train_df.iloc[train_idx]['image_path']))
        
        if v_md5 != t_md5:
            pairs.append((val_idx, train_idx, distances[m]))
            break # one per val image

print(f"Found {len(pairs)} pairs!")

if not os.path.exists("temp_aug"):
    os.makedirs("temp_aug")

for idx, (v_idx, t_idx, dist) in enumerate(pairs):
    v_path = os.path.join(val_dir, val_df.iloc[v_idx]['image_path'])
    t_path = os.path.join(train_dir, train_df.iloc[t_idx]['image_path'])
    
    img_v = cv2.imread(v_path)
    img_t = cv2.imread(t_path)
    
    if img_v is not None and img_t is not None:
        img_v = cv2.resize(img_v, (224, 224))
        img_t = cv2.resize(img_t, (224, 224))
        combo = np.concatenate((img_v, img_t), axis=1)
        out_path = f"temp_aug/pair_{idx}_dist_{dist}.jpg"
        cv2.imwrite(out_path, combo)
        print(f"Saved {out_path}")
