import pandas as pd
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import sys

base_dir = r"d:\AffectNet+\AffectNet+\human_annotated\train_set"
sys.path.append(r"d:\AffectNet+\AffectNet+")
from test_dataloader import AffectNetDataset

manifest_path = os.path.join(base_dir, "final_manifest_v1_FROZEN.csv")
dataset = AffectNetDataset(manifest_path, base_dir, mock_images=False)

# Find one real Contempt image
df = dataset.df
contempt_rows = df[df['human_label'] == 7]
if len(contempt_rows) == 0:
    print("No contempt rows found!")
    sys.exit(1)

sample_row = contempt_rows.iloc[0]
img_path = os.path.join(base_dir, sample_row['image_path'])
print(f"Testing visual augmentation on: {img_path}")

img = Image.open(img_path).convert('RGB')
out_dir = r"d:\AffectNet+\AffectNet+\temp_aug"
os.makedirs(out_dir, exist_ok=True)
img.save(os.path.join(out_dir, "0_original.png"))

# The heavy_transform normalizes, so we must un-normalize for visualization
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

for i in range(1, 5):
    torch.manual_seed(i * 100) 
    out_tensor = dataset.heavy_transform(img)
    
    # Un-normalize
    unnorm_tensor = out_tensor * std + mean
    # Clamp to strictly [0, 1]
    unnorm_tensor = torch.clamp(unnorm_tensor, 0, 1)
    
    # Convert to PIL and save
    out_pil = transforms.ToPILImage()(unnorm_tensor)
    out_name = os.path.join(out_dir, f"{i}_augmented.png")
    out_pil.save(out_name)
    print(f"Saved: {out_name}")

print("Visual augmentation generation complete.")
