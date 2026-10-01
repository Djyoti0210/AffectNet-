import os
import sys
import torch
import torchvision.transforms as transforms
from PIL import Image

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
out_dir = r"d:\AffectNet+\AffectNet+\temp_aug"

base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

orig_tensor = base_transform(img)

print("Original tensor slice [0, 100:105, 100:105]:")
print(orig_tensor[0, 100:105, 100:105])

for i in range(1, 5):
    torch.manual_seed(i * 100)
    out_tensor = dataset.heavy_transform(img)
    print(f"\nAugmented {i} tensor slice [0, 100:105, 100:105]:")
    print(out_tensor[0, 100:105, 100:105])
    
print("\nVerifying lines in save_augmented_faces.py...")
with open(r"d:\AffectNet+\AffectNet+\save_augmented_faces.py", "r") as f:
    lines = f.readlines()
    
print("Code calling transform:")
for i, line in enumerate(lines):
    if "out_tensor = dataset.heavy_transform(img)" in line:
        print(f"Line {i+1}: {line.strip()}")
        
print("Code saving image:")
for i, line in enumerate(lines):
    if "out_pil.save" in line:
        print(f"Line {i+1}: {line.strip()}")
