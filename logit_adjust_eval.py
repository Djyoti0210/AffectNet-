import os
import sys
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from collections import Counter

base_dir = r"d:\AffectNet+\AffectNet+\human_annotated\train_set"
val_dir = r"d:\AffectNet+\AffectNet+\human_annotated\validation_set"
sys.path.append(r"d:\AffectNet+\AffectNet+")
from test_dataloader import AffectNetDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Computing Dataset Priors...")
train_manifest = os.path.join(base_dir, "final_manifest_v1_FROZEN.csv")
df_train = pd.read_csv(train_manifest, usecols=['human_label'], low_memory=False)
counts = df_train['human_label'].value_counts().sort_index().values
total = counts.sum()
freqs = counts / total
freqs_tensor = torch.tensor(freqs, dtype=torch.float32).to(device)
class_names = {0:"Neutral", 1:"Happy", 2:"Sad", 3:"Surprise", 4:"Fear", 5:"Disgust", 6:"Anger", 7:"Contempt"}

for k in range(8):
    print(f"Prior P({class_names[k]}) = {freqs[k]:.4f}")

model = efficientnet_b3()
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 8)
ckpt = torch.load(r"d:\AffectNet+\AffectNet+\checkpoints\best_model.pt", weights_only=False)
model.load_state_dict(ckpt['model_state'])
model = model.to(device)
model.eval()

val_manifest = os.path.join(val_dir, "val_manifest.csv")
dataset = AffectNetDataset(val_manifest, val_dir, mock_images=False)
dataset.heavy_transform = dataset.std_transform

contempt_indices = dataset.df[dataset.df['human_label'] == 7].index.tolist()
contempt_dataset = torch.utils.data.Subset(dataset, contempt_indices)
dataloader = DataLoader(contempt_dataset, batch_size=64, shuffle=False)

preds_list_raw = []
preds_list_adj = []

tau = 1.0

print("\nRunning Inference...")
with torch.no_grad():
    for imgs, labels, _, _ in dataloader:
        imgs = imgs.to(device)
        with torch.amp.autocast('cuda'):
            logits = model(imgs)
            adjusted_logits = logits - tau * torch.log(freqs_tensor)
            
        preds_raw = torch.argmax(logits, dim=1).cpu().numpy()
        preds_adj = torch.argmax(adjusted_logits, dim=1).cpu().numpy()
        
        preds_list_raw.extend(preds_raw)
        preds_list_adj.extend(preds_adj)

print("\n--- RAW CONFUSION MATRIX ROW (NO ADJUSTMENT) ---")
counts_raw = Counter(preds_list_raw)
for k in range(8):
    print(f"Predicted as {class_names[k]:<10}: {counts_raw.get(k, 0)}")

print(f"\n--- LOGIT ADJUSTED (Tau = {tau}) CONFUSION MATRIX ROW ---")
counts_adj = Counter(preds_list_adj)
for k in range(8):
    print(f"Predicted as {class_names[k]:<10}: {counts_adj.get(k, 0)}")
