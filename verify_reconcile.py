import os
import sys
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim

base_dir = r"d:\AffectNet+\AffectNet+\human_annotated\train_set"
sys.path.append(r"d:\AffectNet+\AffectNet+")
from test_dataloader import AffectNetDataset

manifest_path = os.path.join(base_dir, "final_manifest_v1_FROZEN.csv")
dataset = AffectNetDataset(manifest_path, base_dir, mock_images=False)

df = dataset.df
img_path = os.path.join(base_dir, df[df['human_label'] == 7].iloc[0]['image_path'])
img = Image.open(img_path).convert('RGB')

base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
orig_tensor = base_transform(img)

# Unnormalize
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
orig_unnorm = torch.clamp(orig_tensor * std + mean, 0.0, 1.0)

torch.manual_seed(200) # Run 2 (ShearX, ShearY)
heavy = dataset.heavy_transform
heavy.transforms[1].p = 0.0 # Suppress H-Flip to isolate shears
out_tensor = heavy(img)
out_unnorm = torch.clamp(out_tensor * std + mean, 0.0, 1.0)

print("\n--- 1. MAE CALCULATIONS SIDE BY SIDE ---")
# First method (produced 0.5261)
mae_normalized = torch.abs(orig_tensor - out_tensor).mean().item()
print("Code 1 (First attempt):")
print("mae = torch.abs(orig_tensor - out_tensor).mean().item()")
print(f"Value Range : PyTorch Normalized (approx -2.11 to +2.64, mean=0, std=1)")
print(f"Result      : {mae_normalized:.4f}")

# Second method (produced 0.1190)
mae_unnormalized = torch.abs(orig_unnorm - out_unnorm).mean().item()
print("\nCode 2 (Second attempt):")
print("mae = torch.abs(orig_unnorm - out_unnorm).mean().item()")
print(f"Value Range : Strictly [0.0, 1.0] bounded image tensors")
print(f"Result      : {mae_unnormalized:.4f}")
print("\nVerdict: Code 2 is the correct standard for computing visual MAE. Normalization stretches the mathematical distance aggressively.")

print("\n--- 2. SSIM CALL SIGNATURE ---")
orig_np = orig_unnorm.permute(1, 2, 0).numpy()
out_np = out_unnorm.permute(1, 2, 0).numpy()
print("Call: val_ssim = ssim(orig_np, out_np, data_range=1.0, channel_axis=-1)")
print(f"orig_np -> shape: {orig_np.shape}, min: {orig_np.min():.4f}, max: {orig_np.max():.4f}")
print(f"out_np  -> shape: {out_np.shape}, min: {out_np.min():.4f}, max: {out_np.max():.4f}")
val_ssim = ssim(orig_np, out_np, data_range=1.0, channel_axis=-1)
print(f"Re-computed SSIM: {val_ssim:.4f}")

print("\n--- 3. BORDER PIXEL SAMPLE (RGB 0-255 scale) ---")
# Scale back to 0-255 uint8 for human-readable RGB
out_255 = (out_np * 255.0).round().astype(np.uint8)

corners = {
    "Top-Left (0,0)": out_255[0, 0],
    "Top-Right (0,223)": out_255[0, 223],
    "Bottom-Left (223,0)": out_255[223, 0],
    "Bottom-Right (223,223)": out_255[223, 223]
}
midpoints = {
    "Top-Mid (0,112)": out_255[0, 112],
    "Bottom-Mid (223,112)": out_255[223, 112],
    "Left-Mid (112,0)": out_255[112, 0],
    "Right-Mid (112,223)": out_255[112, 223]
}

print("Corners:")
for name, val in corners.items():
    print(f"  {name:<25}: {val.tolist()}")
print("\nEdge Midpoints:")
for name, val in midpoints.items():
    print(f"  {name:<25}: {val.tolist()}")
