import os
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3
from test_dataloader import AffectNetDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms.functional as TF

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading avg_26_30.pt...")
    model = efficientnet_b3()
    # We want features, so we can just use model.features and model.avgpool
    # But let's just replace the classifier with an Identity layer
    model.classifier = nn.Identity()
    
    # We must load the state dict carefully because the saved model has a classifier
    ckpt = torch.load(r"d:\AffectNet+\AffectNet+\checkpoints\avg_26_30.pt", weights_only=False, map_location='cpu')
    state_dict = ckpt['model_state']
    # Remove classifier weights
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('classifier')}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    base_dir = r"d:\AffectNet+\AffectNet+\human_annotated\train_set"
    train_manifest = os.path.join(base_dir, "final_manifest_v1_FROZEN.csv")
    dataset = AffectNetDataset(train_manifest, base_dir, mock_images=False, allowed_tiers=['gold', 'standard', 'soft_weighted', 'high_risk'])
    
    # We don't want the dataset's heavy transform to mess up our specific extraction.
    # We want exactly: standard transform + standard transform on HFlip.
    # The dataloader gives us standard transform if we set heavy_transform = std_transform.
    dataset.heavy_transform = dataset.std_transform
    
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

    all_features = []
    all_labels = []
    all_soft_labels = []
    all_tiers = []

    print("Extracting features (clean and hflip)...")
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for imgs, labels, soft_labels, tiers in tqdm(dataloader, total=len(dataloader)):
            imgs = imgs.to(device)
            imgs_flipped = TF.hflip(imgs)
            
            feat_clean = model(imgs)
            feat_flip = model(imgs_flipped)
            
            # EfficientNet outputs (B, 1536) if classifier is Identity
            all_features.append(feat_clean.cpu())
            all_features.append(feat_flip.cpu())
            
            all_labels.append(labels)
            all_labels.append(labels)
            
            all_soft_labels.append(soft_labels)
            all_soft_labels.append(soft_labels)
            
            all_tiers.append(tiers.cpu())
            all_tiers.append(tiers.cpu())
            
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_soft_labels = torch.cat(all_soft_labels, dim=0)
    all_tiers = torch.cat(all_tiers, dim=0)

    print(f"Features shape: {all_features.shape}")
    
    cache = {
        'features': all_features,
        'labels': all_labels,
        'soft_labels': all_soft_labels,
        'tiers': all_tiers
    }
    torch.save(cache, "train_features_cache.pt")
    print("Saved to train_features_cache.pt")

if __name__ == '__main__':
    main()
