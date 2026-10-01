import torch
import torch.nn.functional as F
import pandas as pd
import hashlib
import os
import cv2
import numpy as np
import concurrent.futures

def get_md5(file_path):
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

train_df = pd.read_csv(r"d:\AffectNet+\AffectNet+\human_annotated\train_set\final_manifest_v1_FROZEN.csv")
val_df = pd.read_csv(r"d:\AffectNet+\AffectNet+\human_annotated\validation_set\val_manifest.csv")

train_dir = r"d:\AffectNet+\AffectNet+\human_annotated\train_set"
val_dir = r"d:\AffectNet+\AffectNet+\human_annotated\validation_set"

print("Hashing validation images...", flush=True)
val_hashes = {}
for i, row in val_df.iterrows():
    h = get_md5(os.path.join(val_dir, row['image_path']))
    val_hashes[h] = i

print("Hashing training images in parallel...", flush=True)
exact_duplicates = 0

def check_file(rel_path):
    h = get_md5(os.path.join(train_dir, rel_path))
    return h in val_hashes

with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
    results = executor.map(check_file, train_df['image_path'].tolist())
    exact_duplicates = sum(results)

print(f"Exact MD5 duplicates: {exact_duplicates} out of {len(val_df)}", flush=True)

print("Loading cached train features...", flush=True)
cache = torch.load("train_features_cache.pt", weights_only=False)
train_features = cache['features']
train_labels = cache['labels']

print("Loading validation features (we need to extract them)...", flush=True)
from torchvision.models import efficientnet_b3
import torch.nn as nn
from test_dataloader import AffectNetDataset
from torch.utils.data import DataLoader

model = efficientnet_b3()
model.classifier = nn.Identity()
ckpt = torch.load(r"d:\AffectNet+\AffectNet+\checkpoints\avg_26_30.pt", map_location='cpu', weights_only=False)
state_dict = {k: v for k, v in ckpt['model_state'].items() if not k.startswith('classifier')}
model.load_state_dict(state_dict)
model.eval()

dataset = AffectNetDataset(r"d:\AffectNet+\AffectNet+\human_annotated\validation_set\val_manifest.csv", val_dir, mock_images=False, is_validation=True)
dataset.heavy_transform = dataset.std_transform
dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

val_features = []
with torch.no_grad():
    for imgs, _, _, _ in dataloader:
        feat = model(imgs)
        val_features.append(feat)
val_features = torch.cat(val_features, dim=0)

# Normalize both for cosine similarity
train_features_norm = F.normalize(train_features.float().cuda(), p=2, dim=1)
val_features_norm = F.normalize(val_features.float().cuda(), p=2, dim=1)

print("Computing cosine similarities on GPU...", flush=True)
near_duplicates = []
for i in range(len(val_features_norm)):
    sims = torch.matmul(train_features_norm, val_features_norm[i])
    max_sim = sims.max().item()
    if max_sim > 0.98:
        near_duplicates.append(i)

print(f"Validation images with a train-set nearest neighbor > 0.98 similarity: {len(near_duplicates)} out of {len(val_features)}", flush=True)

near_dup_set = set(near_duplicates)
is_near_dup = np.array([i in near_dup_set for i in range(len(val_features))])

print("Loading retrained head to compute accuracies...", flush=True)
head = nn.Linear(1536, 8)
head.load_state_dict(torch.load("retrained_head.pt", weights_only=False))
head.eval()

with torch.no_grad():
    val_logits = head(val_features)
    val_preds = val_logits.argmax(dim=1).numpy()
    val_labels = val_df['human_label'].values
    
    if len(near_duplicates) > 0:
        dup_acc = (val_preds[is_near_dup] == val_labels[is_near_dup]).mean()
        print(f"Accuracy on near-duplicates (>0.98): {dup_acc*100:.2f}%", flush=True)
    else:
        print("Accuracy on near-duplicates (>0.98): N/A (none found)", flush=True)
        
    non_dup_acc = (val_preds[~is_near_dup] == val_labels[~is_near_dup]).mean()
    print(f"Accuracy on non-duplicates (<=0.98): {non_dup_acc*100:.2f}%", flush=True)

    # Compute Train Set Accuracy
    print("Computing Train Set Accuracy...", flush=True)
    train_preds = []
    for i in range(0, len(train_features), 4096):
        batch = train_features[i:i+4096]
        train_preds.append(head(batch).argmax(dim=1))
    train_preds = torch.cat(train_preds).numpy()
    
    from sklearn.metrics import accuracy_score, f1_score
    train_labels_np = train_labels.numpy()
    train_acc = accuracy_score(train_labels_np, train_preds)
    train_f1 = f1_score(train_labels_np, train_preds, average='macro')
    print(f"Train Set Accuracy: {train_acc*100:.2f}%", flush=True)
    print(f"Train Set Macro F1: {train_f1:.4f}", flush=True)
