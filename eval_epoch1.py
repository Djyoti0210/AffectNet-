import os
import sys
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import pandas as pd

base_dir = r"d:\AffectNet+\AffectNet+\human_annotated\train_set"
val_dir = r"d:\AffectNet+\AffectNet+\human_annotated\validation_set"
sys.path.append(r"d:\AffectNet+\AffectNet+")
from test_dataloader import AffectNetDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading architecture...")
model = efficientnet_b3()
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 8)

ckpt_path = r"d:\AffectNet+\AffectNet+\checkpoints\latest.pt"
print(f"Loading weights from {ckpt_path}...")
ckpt = torch.load(ckpt_path, weights_only=False)
model.load_state_dict(ckpt['model_state'])
model = model.to(device)
model.eval()

print("Loading Validation Dataset...")
val_manifest = os.path.join(val_dir, "val_manifest.csv")
dataset = AffectNetDataset(val_manifest, val_dir, mock_images=False)
dataset.heavy_transform = dataset.std_transform

dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

all_preds = []
all_labels = []

print("Running Inference over 4,000 perfectly balanced validation images...")
with torch.no_grad():
    for imgs, labels, _, _ in dataloader:
        imgs = imgs.to(device)
        with torch.amp.autocast('cuda'):
            logits = model(imgs)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

class_names = ["0: Neutral", "1: Happy", "2: Sad", "3: Surprise", "4: Fear", "5: Disgust", "6: Anger", "7: Contempt"]
report = classification_report(all_labels, all_preds, target_names=class_names, digits=4, zero_division=0)

out_file = r"d:\AffectNet+\AffectNet+\epoch1_detailed_analysis.txt"
with open(out_file, 'w') as f:
    f.write("=== AFFECTNET EPOCH 1 DETAILED VALIDATION REPORT ===\n")
    f.write(f"Global Epoch: {ckpt['global_epoch']}\n\n")
    f.write(report)
    
print(f"\n--- SUCCESS ---\nReport saved to: {out_file}")
