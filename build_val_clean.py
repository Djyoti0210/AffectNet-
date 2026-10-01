import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

# Read IDs
with open("exact_dup_ids.txt", "r") as f:
    lines = f.read().splitlines()
    exact_ids = set([int(x) for x in lines if x.strip()])
    
with open("near_dup_ids.txt", "r") as f:
    lines = f.read().splitlines()
    near_ids = set([int(x) for x in lines if x.strip()])
    
all_dups = exact_ids.union(near_ids)

print(f"Loaded {len(exact_ids)} exact duplicates and {len(near_ids)} near duplicates.")
print(f"Total unique IDs to remove: {len(all_dups)}")

val_df = pd.read_csv(r"d:\AffectNet+\AffectNet+\human_annotated\validation_set\val_manifest.csv")
print(f"Original val set size: {len(val_df)}")

# Filter out duplicates
val_clean = val_df[~val_df.index.isin(all_dups)].copy()
val_clean = val_clean.reset_index(drop=True)
print(f"Clean val set size: {len(val_clean)}")

# Print per-class counts
print("\nPer-class counts in val_clean:")
counts = val_clean['human_label'].value_counts().sort_index()
for label, count in counts.items():
    print(f"Label {label}: {count}")

# Make stratified 2-fold split
skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
val_clean['fold'] = 0
for fold_idx, (train_idx, test_idx) in enumerate(skf.split(val_clean, val_clean['human_label'])):
    val_clean.loc[test_idx, 'fold'] = fold_idx

val_clean.to_csv("val_clean_folds.csv", index=False)
print("\nSaved val_clean_folds.csv")
