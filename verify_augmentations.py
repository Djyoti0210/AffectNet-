import os
import sys
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torchvision.transforms.autoaugment as autoaugment
from PIL import Image
import numpy as np

try:
    from skimage.metrics import structural_similarity as ssim
    has_skimage = True
except ImportError:
    has_skimage = False

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

# Produce original tensor matching exactly the base pipeline
base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # Skipping RandomHorizontalFlip to establish a static baseline
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
orig_tensor = base_transform(img)

# Monkey patch F.affine to catch affine magnitude applied (Shear, Rotate, Translate)
affine_params = []
original_affine = F.affine

def patched_affine(img, angle, translate, scale, shear, interpolation=F.InterpolationMode.NEAREST, fill=None, center=None):
    affine_params.append({'angle': angle, 'shear': shear, 'translate': translate})
    return original_affine(img, angle, translate, scale, shear, interpolation=interpolation, fill=fill, center=center)

F.affine = patched_affine

# Monkey patch _apply_op to catch operation names
sampled_ops = []
try:
    original_apply_op = autoaugment._apply_op
except AttributeError:
    original_apply_op = None

def patched_apply_op(img, op_name, magnitude, interpolation, fill):
    sampled_ops.append(op_name)
    return original_apply_op(img, op_name, magnitude, interpolation, fill)

if original_apply_op:
    autoaugment._apply_op = patched_apply_op

print("--- ANALYZING AUGMENTED TENSORS ---")
for i in range(1, 5):
    torch.manual_seed(i * 100)
    sampled_ops.clear()
    affine_params.clear()
    
    # We will temporarily mock RandomHorizontalFlip to probability 0 inside the pipeline
    # to purely isolate the RandAugment changes from a random flip.
    heavy = dataset.heavy_transform
    heavy.transforms[1].p = 0.0
    
    out_tensor = heavy(img)
    
    # 1. Mean Absolute Pixel Difference
    mae = torch.abs(orig_tensor - out_tensor).mean().item()
    
    # 3. Structural Similarity (SSIM)
    if has_skimage:
        orig_np = orig_tensor.permute(1, 2, 0).numpy()
        out_np = out_tensor.permute(1, 2, 0).numpy()
        
        # data_range = Max - Min pixel intensity
        data_range = out_np.max() - out_np.min()
        if data_range == 0: data_range = 1.0
        
        val_ssim = ssim(orig_np, out_np, data_range=float(data_range), channel_axis=-1)
    else:
        val_ssim = "skimage not installed"
        
    print(f"\nRun {i} | Seed: {i*100}")
    print(f"  Ops chosen    : {sampled_ops}")
    print(f"  Mean Abs Diff : {mae:.4f}")
    print(f"  SSIM          : {val_ssim if isinstance(val_ssim, str) else f'{val_ssim:.4f}'}")
    
    if affine_params:
        print(f"  Affine Params applied: {affine_params}")

print("\nDone.")
