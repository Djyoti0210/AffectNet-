import os
import pandas as pd
import numpy as np

base_dir = r"d:\AffectNet+\AffectNet+\human_annotated"
val_manifest = os.path.join(base_dir, "validation_set", "val_manifest.csv")

df = pd.read_csv(val_manifest)

folds = np.zeros(len(df), dtype=int)
for cls in range(8):
    cls_idx = df.index[df['human_label'] == cls].tolist()
    np.random.seed(42 + cls)
    np.random.shuffle(cls_idx)
    
    # 500 images per class in validation set. Split into two halves (250 each).
    mid = len(cls_idx) // 2
    fold_0_idx = cls_idx[:mid]
    fold_1_idx = cls_idx[mid:]
    
    folds[fold_0_idx] = 0
    folds[fold_1_idx] = 1

df['fold'] = folds
df.to_csv("val_folds.csv", index=False)
print("Fold assignments saved to val_folds.csv")
for f in [0, 1]:
    sub = df[df['fold'] == f]
    print(f"Fold {f}:")
    print(sub['human_label'].value_counts().sort_index())
