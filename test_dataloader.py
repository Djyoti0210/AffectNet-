import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import os
import sys
import time

class AffectNetDataset(Dataset):
    def __init__(self, manifest_path, root_dir, mock_images=False, allowed_tiers=None, is_validation=False):
        self.df = pd.read_csv(manifest_path, low_memory=False)
        if allowed_tiers is not None:
            self.df = self.df[self.df['training_tier'].isin(allowed_tiers)].copy()
            self.df.reset_index(drop=True, inplace=True)
            
        self.root_dir = root_dir
        self.mock_images = mock_images
        self.is_validation = is_validation
        
        # Standard augmentations for Gold/Standard tiers
        self.std_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Heavier augmentations for Minority Classes & Soft-Weighted/High-Risk tiers
        try:
            from torchvision.transforms import RandAugment, InterpolationMode
            self.heavy_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                RandAugment(num_ops=2, magnitude=9, interpolation=InterpolationMode.BILINEAR, fill=[124, 116, 104]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        except ImportError:
            self.heavy_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.RandomRotation(20),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        # Define minority classes logically (those with class_sample_weight significantly > 1.0)
        # Base on our prior report: Surprise, Fear, Disgust, Contempt
        self.minority_classes = {3, 4, 5, 7}
        self.soft_cols = [f"soft_label_{i}" for i in range(8)]
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        human_label = int(row['human_label'])
        tier = row.get('training_tier', 'standard')
        
        if self.mock_images:
            # Bypass slow disk IO for sampling logic validation
            image_tensor = torch.zeros((3, 224, 224), dtype=torch.float32)
        else:
            img_path = os.path.join(self.root_dir, row['image_path'])
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception:
                img = Image.new('RGB', (224, 224)) # Failsafe
                
            # Apply augmentation conditionally
            if getattr(self, 'is_validation', False):
                image_tensor = self.std_transform(img)
            elif (human_label in self.minority_classes) or (tier in ['soft_weighted', 'high_risk']):
                image_tensor = self.heavy_transform(img)
            else:
                image_tensor = self.std_transform(img)
                
        # Normalize soft labels (divide by sum so they yield a valid PDF summing to 1.0)
        soft_labels = row[self.soft_cols].values.astype('float32')
        soft_sum = soft_labels.sum()
        if soft_sum > 0:
            soft_labels = soft_labels / soft_sum
        else:
            soft_labels = np.ones(8, dtype='float32') / 8.0
            
        soft_label_vector = torch.tensor(soft_labels, dtype=torch.float32)
        tier_loss_weight = torch.tensor(row.get('tier_loss_weight', 1.0), dtype=torch.float32)
        
        return image_tensor, human_label, soft_label_vector, tier_loss_weight

def test_sampler():
    base_dir = r"d:\AffectNet+\AffectNet+\human_annotated\train_set"
    manifest_path = os.path.join(base_dir, "final_manifest_v1_FROZEN.csv")
    report_file = os.path.join(r"d:\AffectNet+\AffectNet+", "dataloader_report.txt")
    
    sys.stdout = open(report_file, 'w')
    
    print("Loading Dataset (using mock_images=True for instant sampler validation)...")
    dataset = AffectNetDataset(manifest_path, base_dir, mock_images=True)
    
    # Extract weights for the WeightedRandomSampler
    weights = dataset.df['class_sample_weight'].values
    sample_weights = torch.tensor(weights, dtype=torch.double)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=2048, # Max out for fast dummy iteration
        sampler=sampler,
        num_workers=0 
    )
    
    print("Iterating over one full epoch (287,651 samples)...")
    class_counts = {i: 0 for i in range(8)}
    
    start_time = time.time()
    for batch_idx, (img, label, soft, t_weight) in enumerate(dataloader):
        labels_arr = label.numpy()
        for i in range(8):
            class_counts[i] += (labels_arr == i).sum()
            
    end_time = time.time()
    print(f"\nEpoch completed in {end_time - start_time:.2f} seconds.")
    
    class_names = {0: "Neutral", 1: "Happy", 2: "Sad", 3: "Surprise", 4: "Fear", 5: "Disgust", 6: "Anger", 7: "Contempt"}
    print("\n--- ACTUAL OBSERVED CLASS DISTRIBUTION OVER 1 EPOCH ---")
    print(f"{'Class':<10} | {'Raw Count':<10} | {'Observed Yield':<15}")
    print("-" * 45)
    
    raw_counts = dataset.df['human_label'].value_counts().to_dict()
    for c in range(8):
        print(f"{class_names[c]:<10} | {raw_counts.get(c,0):<10} | {class_counts[c]:<15}")
        
    print("\nDone.")

if __name__ == '__main__':
    test_sampler()
