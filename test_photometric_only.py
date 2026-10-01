import os
import sys
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd
from sklearn.metrics import classification_report

base_dir = r"d:\AffectNet+\AffectNet+\human_annotated\train_set"
val_dir = r"d:\AffectNet+\AffectNet+\human_annotated\validation_set"
sys.path.append(r"d:\AffectNet+\AffectNet+")
from test_dataloader import AffectNetDataset
from train import combined_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler = torch.amp.GradScaler('cuda')

print("Initializing 1-Epoch Photometric-Only Experiment...")
model = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 8)
model = model.to(device)

manifest_path = os.path.join(base_dir, "final_manifest_v1_FROZEN.csv")
dataset = AffectNetDataset(manifest_path, base_dir, mock_images=False, allowed_tiers=['gold'])

# MONKEY PATCH: Photometric ONLY pipeline for minority classes. (No Rotate, Shear, Translate)
dataset.heavy_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
    transforms.RandomPosterize(bits=4, p=0.5),
    transforms.RandomAutocontrast(p=0.5),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

weights = torch.tensor(dataset.df['class_sample_weight'].values, dtype=torch.double)
sampler = WeightedRandomSampler(weights=weights, num_samples=len(dataset), replacement=True)
dataloader = DataLoader(dataset, batch_size=32, sampler=sampler, num_workers=4)

optimizer = torch.optim.AdamW([
    {'params': model.features.parameters(), 'lr': 3e-5},
    {'params': model.classifier.parameters(), 'lr': 3e-4}
], weight_decay=1e-4)

model.train()
print(f"Starting 1-Epoch Training Loop ({len(dataloader)} steps)...")
for step, (imgs, labels, softs, batch_weights) in enumerate(dataloader):
    imgs, labels = imgs.to(device), labels.to(device)
    softs, batch_weights = softs.to(device), batch_weights.to(device)
    
    with torch.amp.autocast('cuda'):
        logits = model(imgs)
        tot_loss, ce, kl = combined_loss(logits, labels, softs, batch_weights, 1.0)
        loss = tot_loss / 2
        
    scaler.scale(loss).backward()
    if (step + 1) % 2 == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    if step % 500 == 0:
        print(f"Step {step}/{len(dataloader)}")
        
print("Experiment training complete. Running validation...")
model.eval()

val_manifest = os.path.join(val_dir, "val_manifest.csv")
val_dataset = AffectNetDataset(val_manifest, val_dir, mock_images=False)
val_dataset.heavy_transform = val_dataset.std_transform

val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels, _, _ in val_dataloader:
        imgs = imgs.to(device)
        with torch.amp.autocast('cuda'):
            logits = model(imgs)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

class_names = ["0: Neutral", "1: Happy", "2: Sad", "3: Surprise", "4: Fear", "5: Disgust", "6: Anger", "7: Contempt"]
report = classification_report(all_labels, all_preds, target_names=class_names, digits=4, zero_division=0)

out_file = r"d:\AffectNet+\AffectNet+\photometric_experiment.txt"
with open(out_file, 'w') as f:
    f.write(report)
print("Experiment complete!")
