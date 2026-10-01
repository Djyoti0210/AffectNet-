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

model = efficientnet_b3()
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 8)
ckpt = torch.load(r"d:\AffectNet+\AffectNet+\checkpoints\best_model.pt", weights_only=False)
model.load_state_dict(ckpt['model_state'])
model = model.to(device)
model.eval()

val_manifest = os.path.join(val_dir, "val_manifest.csv")
dataset = AffectNetDataset(val_manifest, val_dir, mock_images=False)
dataset.heavy_transform = dataset.std_transform

# Filter to ONLY Contempt (7)
contempt_indices = dataset.df[dataset.df['human_label'] == 7].index.tolist()
contempt_dataset = torch.utils.data.Subset(dataset, contempt_indices)
dataloader = DataLoader(contempt_dataset, batch_size=64, shuffle=False)

preds_list = []
logits_samples = []

class_names = {0:"Neutral", 1:"Happy", 2:"Sad", 3:"Surprise", 4:"Fear", 5:"Disgust", 6:"Anger", 7:"Contempt"}

with torch.no_grad():
    for step, (imgs, labels, _, _) in enumerate(dataloader):
        imgs = imgs.to(device)
        with torch.amp.autocast('cuda'):
            logits = model(imgs)
            probs = torch.nn.functional.softmax(logits, dim=1)
            
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        preds_list.extend(preds)
        
        if step == 0:
            for i in range(5):
                logits_samples.append((logits[i].cpu().numpy(), probs[i].cpu().numpy()))

print("--- CONTEMPT CONFUSION MATRIX ROW ---")
counts = Counter(preds_list)
for k in range(8):
    print(f"Predicted as {class_names[k]:<10}: {counts.get(k, 0)} / {len(contempt_dataset)}")

print("\n--- RAW LOGITS & PROBABILITIES FOR 5 CONTEMPT SAMPLES ---")
for idx, (l, p) in enumerate(logits_samples):
    print(f"\nSample {idx+1}:")
    top3 = np.argsort(p)[::-1][:3]
    for c in top3:
        print(f"  {class_names[c]:<10} -> Prob: {p[c]*100:.2f}% (Logit: {l[c]:.2f})")
    print(f"  Contempt   -> Prob: {p[7]*100:.2f}% (Logit: {l[7]:.2f})")
