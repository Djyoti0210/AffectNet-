import cv2
import pandas as pd
import numpy as np
import os
import concurrent.futures

def compute_dhash(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: return 0
    resized = cv2.resize(img, (9, 8))
    diff = resized[:, 1:] > resized[:, :-1]
    # Return as 64-bit integer
    return sum([int(v) << i for i, v in enumerate(diff.flatten())])

def hamming_distance(h1, h2):
    return bin(h1 ^ h2).count('1')

if __name__ == '__main__':
    train_df = pd.read_csv(r"d:\AffectNet+\AffectNet+\human_annotated\train_set\final_manifest_v1_FROZEN.csv")
    val_df = pd.read_csv(r"d:\AffectNet+\AffectNet+\human_annotated\validation_set\val_manifest.csv")

    train_dir = r"d:\AffectNet+\AffectNet+\human_annotated\train_set"
    val_dir = r"d:\AffectNet+\AffectNet+\human_annotated\validation_set"

    print("Computing dHash for validation set...", flush=True)
    val_hashes = []
    for i, row in val_df.iterrows():
        val_hashes.append(compute_dhash(os.path.join(val_dir, row['image_path'])))
    
    val_hashes = np.array(val_hashes, dtype=np.uint64)
    
    print("Computing dHash for training set in parallel...", flush=True)
    train_paths = [os.path.join(train_dir, p) for p in train_df['image_path']]
    
    train_hashes = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        train_hashes = list(executor.map(compute_dhash, train_paths))
        
    train_hashes = np.array(train_hashes, dtype=np.uint64)
    
    print("Finding near-duplicates (Hamming distance <= 4)...", flush=True)
    # Threshold of 4 bits out of 64 allows for minor compression artifacts or tiny crops
    near_dup_val_indices = set()
    exact_dup_val_indices = set()
    
    for i, vh in enumerate(val_hashes):
        # bitwise XOR and popcount
        # numpy doesn't have a fast popcount for uint64 out of the box, but we can do it:
        xor = np.bitwise_xor(train_hashes, vh)
        # Brian Kernighan's or just string format loop is slow for 287k * 4000.
        # But wait, 287k array XOR takes 1ms. We do this 4000 times -> 4 seconds.
        # To count bits, we can view as uint8 and sum.
        xor_bytes = xor.view(np.uint8)
        # Using np.unpackbits
        bits = np.unpackbits(xor_bytes.reshape(-1, 8), axis=1)
        distances = bits.sum(axis=1)
        
        min_dist = distances.min()
        if min_dist == 0:
            exact_dup_val_indices.add(i)
        elif min_dist <= 4:
            near_dup_val_indices.add(i)
            
    print(f"Exact dHash duplicates (dist=0): {len(exact_dup_val_indices)}")
    print(f"Near dHash duplicates (0 < dist <= 4): {len(near_dup_val_indices)}")
    
    all_dups = sorted(list(exact_dup_val_indices.union(near_dup_val_indices)))
    print(f"Total images to remove from validation: {len(all_dups)} out of {len(val_df)}")
    
    with open("exact_dup_ids.txt", "w") as f:
        f.write("\n".join(map(str, sorted(list(exact_dup_val_indices)))))
    with open("near_dup_ids.txt", "w") as f:
        f.write("\n".join(map(str, sorted(list(near_dup_val_indices)))))
        
    print("Saved ID lists to exact_dup_ids.txt and near_dup_ids.txt")
