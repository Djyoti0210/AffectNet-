import torch
import torchvision.transforms.autoaugment as autoaugment
from PIL import Image
import os
import sys

# 1. Monkey patch _apply_op to intercept which operations RandAugment actually picks
try:
    original_apply_op = autoaugment._apply_op
except AttributeError:
    # If using older torchvision, the function might be named differently or class method
    original_apply_op = None

sampled_ops = []

def patched_apply_op(img, op_name, magnitude, interpolation, fill):
    sampled_ops.append(op_name)
    return original_apply_op(img, op_name, magnitude, interpolation, fill)

if original_apply_op:
    autoaugment._apply_op = patched_apply_op

base_dir = r"d:\AffectNet+\AffectNet+\human_annotated\train_set"
sys.path.append(r"d:\AffectNet+\AffectNet+")
from test_dataloader import AffectNetDataset
manifest_path = os.path.join(base_dir, "final_manifest_v1_FROZEN.csv")
dataset = AffectNetDataset(manifest_path, base_dir, mock_images=False)

df = dataset.df
contempt_rows = df[df['human_label'] == 7]
sample_row = contempt_rows.iloc[0]
img_path = os.path.join(base_dir, sample_row['image_path'])

img = Image.open(img_path).convert('RGB')

# 2. Same un-normalization tensors used in save_augmented_faces.py
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

print("--- RE-RUNNING AUGMENTATIONS TO CAPTURE OPS AND MIN/MAX ---")
for i in range(1, 5):
    seed = i * 100
    torch.manual_seed(seed)
    sampled_ops.clear()
    
    # Run pipeline
    out_tensor = dataset.heavy_transform(img)
    
    # Calculate pre-clamp un-normalization math
    unnorm_tensor = (out_tensor * std) + mean
    
    min_val = unnorm_tensor.min().item()
    max_val = unnorm_tensor.max().item()
    
    ops_str = ", ".join(sampled_ops) if sampled_ops else "Could not intercept (likely older torchvision version)"
    
    print(f"Run {i} | Seed: {seed}")
    print(f"  Ops chosen     : [{ops_str}]")
    print(f"  Pre-clamp Min  : {min_val:.4f}")
    print(f"  Pre-clamp Max  : {max_val:.4f}")

print("\nDone.")
