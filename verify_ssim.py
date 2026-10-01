import os
import sys
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torchvision.transforms.autoaugment as autoaugment
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim

base_dir = r"d:\AffectNet+\AffectNet+\human_annotated\train_set"
sys.path.append(r"d:\AffectNet+\AffectNet+")
from test_dataloader import AffectNetDataset

manifest_path = os.path.join(base_dir, "final_manifest_v1_FROZEN.csv")
dataset = AffectNetDataset(manifest_path, base_dir, mock_images=False)

df = dataset.df
contempt_rows = df[df['human_label'] == 7]
img_path = os.path.join(base_dir, contempt_rows.iloc[0]['image_path'])
img = Image.open(img_path).convert('RGB')

base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
orig_tensor = base_transform(img)

# Center vs Border masks
center_mask = torch.zeros((224, 224), dtype=torch.bool)
# Center 100x100 means from 62 to 162
center_mask[62:162, 62:162] = True
border_mask = ~center_mask

# Un-normalize to [0,1]
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
orig_unnorm = torch.clamp(orig_tensor * std + mean, 0, 1)

print("--- SSIM AND MAE ANALYSIS ---")
fill_used = []

original_apply_op = autoaugment._apply_op
def patched_apply_op(img, op_name, magnitude, interpolation, fill):
    fill_used.append(fill)
    return original_apply_op(img, op_name, magnitude, interpolation, fill)
autoaugment._apply_op = patched_apply_op

for i in range(1, 5):
    torch.manual_seed(i * 100)
    fill_used.clear()
    
    heavy = dataset.heavy_transform
    heavy.transforms[1].p = 0.0 # remove horizontal flip
    out_tensor = heavy(img)
    out_unnorm = torch.clamp(out_tensor * std + mean, 0, 1)
    
    orig_np = orig_unnorm.permute(1, 2, 0).numpy()
    out_np = out_unnorm.permute(1, 2, 0).numpy()
    
    # SSIM using exact [0, 1] data range
    val_ssim = ssim(orig_np, out_np, data_range=1.0, channel_axis=-1)
    
    print(f"\nRun {i} | Seed: {i*100}")
    print(f"  SSIM (Unnormalized): {val_ssim:.4f}")
    if fill_used:
        print(f"  Fill param passed to RandAugment: {fill_used[0]}")
    
    if i == 2:
        # Diff averaged over color channels
        diff = torch.abs(orig_unnorm - out_unnorm).mean(dim=0)
        center_mae = diff[center_mask].mean().item()
        border_mae = diff[border_mask].mean().item()
        overall_mae = diff.mean().item()
        
        print(f"  -> Run 2 Overall MAE (Unnormalized): {overall_mae:.4f}")
        print(f"  -> Run 2 Center 100x100 MAE        : {center_mae:.4f}")
        print(f"  -> Run 2 Border MAE                : {border_mae:.4f}")
        
        # Black fill checking
        black_pixels = (out_unnorm == 0.0).all(dim=0).sum().item()
        pct_black = (black_pixels / (224*224)) * 100
        print(f"  -> Run 2 Pure Black (Fill) Area    : {pct_black:.2f}% ({black_pixels} pixels)")

print("\nDone.")
