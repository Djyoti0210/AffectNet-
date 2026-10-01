import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import time

val_df = pd.read_csv(r"d:\AffectNet+\AffectNet+\human_annotated\validation_set\val_manifest.csv")

print("Loading cached train features...")
cache = torch.load("train_features_cache.pt", weights_only=False)
train_features = cache['features']
train_labels = cache['labels']

print("Loading validation features...")
from torchvision.models import efficientnet_b3
from test_dataloader import AffectNetDataset
from torch.utils.data import DataLoader

model = efficientnet_b3()
model.classifier = nn.Identity()
ckpt = torch.load(r"d:\AffectNet+\AffectNet+\checkpoints\avg_26_30.pt", map_location='cpu', weights_only=False)
state_dict = {k: v for k, v in ckpt['model_state'].items() if not k.startswith('classifier')}
model.load_state_dict(state_dict)
model.cuda()
model.eval()

dataset = AffectNetDataset(r"d:\AffectNet+\AffectNet+\human_annotated\validation_set\val_manifest.csv", r"d:\AffectNet+\AffectNet+\human_annotated\validation_set", mock_images=False, is_validation=True)
dataset.heavy_transform = dataset.std_transform
dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)

val_features = []
with torch.no_grad():
    for imgs, _, _, _ in dataloader:
        feat = model(imgs.cuda())
        val_features.append(feat.cpu())
val_features = torch.cat(val_features, dim=0)

train_features_norm = F.normalize(train_features.float(), p=2, dim=1)
val_features_norm = F.normalize(val_features.float(), p=2, dim=1)

print("Computing cosine similarities in small chunks on CPU...")
near_duplicates = []
t0 = time.time()
# Chunk over validation (4000)
for i in range(len(val_features_norm)):
    # 1 x 1536  dot  575302 x 1536
    sims = (train_features_norm * val_features_norm[i]).sum(dim=1)
    max_sim = sims.max().item()
    if max_sim > 0.98:
        near_duplicates.append(i)
    if i % 1000 == 0:
        print(f"Done {i}/4000...", flush=True)

print(f"Validation images with > 0.98 similarity: {len(near_duplicates)} out of {len(val_features)}")

near_dup_set = set(near_duplicates)
is_near_dup = np.array([i in near_dup_set for i in range(len(val_features))])

print("Computing Train Set Accuracy...")
head = nn.Linear(1536, 8)
head.load_state_dict(torch.load("retrained_head.pt", weights_only=False))
head.cuda()
head.eval()

with torch.no_grad():
    val_logits = head(val_features.cuda())
    val_preds = val_logits.argmax(dim=1).cpu().numpy()
    val_labels = val_df['human_label'].values
    
    if len(near_duplicates) > 0:
        dup_acc = (val_preds[is_near_dup] == val_labels[is_near_dup]).mean()
        print(f"Accuracy on near-duplicates (>0.98): {dup_acc*100:.2f}%")
    else:
        print("Accuracy on near-duplicates (>0.98): N/A (none found)")
        
    non_dup_acc = (val_preds[~is_near_dup] == val_labels[~is_near_dup]).mean()
    print(f"Accuracy on non-duplicates (<=0.98): {non_dup_acc*100:.2f}%")

    train_preds = []
    for i in range(0, len(train_features), 4096):
        batch = train_features[i:i+4096].cuda()
        train_preds.append(head(batch).argmax(dim=1).cpu())
    train_preds = torch.cat(train_preds).numpy()
    
    from sklearn.metrics import accuracy_score, f1_score
    train_labels_np = train_labels.numpy()
    train_acc = accuracy_score(train_labels_np, train_preds)
    train_f1 = f1_score(train_labels_np, train_preds, average='macro')
    print(f"Train Set Accuracy: {train_acc*100:.2f}%")
    print(f"Train Set Macro F1: {train_f1:.4f}")
