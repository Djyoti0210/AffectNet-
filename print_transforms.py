import sys
import os

base_dir = r"d:\AffectNet+\AffectNet+\human_annotated\train_set"
sys.path.append(r"d:\AffectNet+\AffectNet+")
from test_dataloader import AffectNetDataset

manifest_path = os.path.join(base_dir, "final_manifest_v1_FROZEN.csv")

# Create dataset (mock_images=True to avoid loading the whole df just to get the transforms)
dataset = AffectNetDataset(manifest_path, base_dir, mock_images=True)

print("--- (a) Pipeline for a gold-tier Happy image (std_transform) ---")
print(dataset.std_transform)

print("\n--- (b) Pipeline for a high-risk Contempt image (heavy_transform) ---")
print(dataset.heavy_transform)
