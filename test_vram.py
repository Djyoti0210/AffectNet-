import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

base_dir = r"d:\AffectNet+\AffectNet+\human_annotated\train_set"
sys.path.append(r"d:\AffectNet+\AffectNet+")
from test_dataloader import AffectNetDataset
from train import combined_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

manifest_path = os.path.join(base_dir, "final_manifest_v1_FROZEN.csv")
dataset = AffectNetDataset(manifest_path, base_dir, mock_images=False)
# Minimal workers for fast startup test
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

model = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 8)
model = model.to(device)
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
scaler = torch.cuda.amp.GradScaler()

accumulation_steps = 2
optimizer.zero_grad()

print("--- RUNNING VRAM STRESS TEST ---")
torch.cuda.reset_peak_memory_stats()
start_time = time.time()

for step, (imgs, labels, softs, weights) in enumerate(dataloader):
    if step >= 10:
        break
        
    imgs = imgs.to(device)
    labels = labels.to(device)
    softs = softs.to(device)
    weights = weights.to(device)
    
    with torch.cuda.amp.autocast():
        logits = model(imgs)
        tot_loss, ce, kl = combined_loss(logits, labels, softs, weights)
        # Normalize loss for accumulation to match mathematically batch=64
        loss = tot_loss / accumulation_steps
        
    scaler.scale(loss).backward()
    
    if (step + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
    print(f"Step {step+1}/10 completed. Local Loss: {tot_loss.item():.4f}")

peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
print(f"\n--- TEST COMPLETE ({time.time()-start_time:.1f} sec) ---")
print(f"Peak VRAM Allocated: {peak_vram:.2f} GB")
