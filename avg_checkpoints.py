import os
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3
from test_dataloader import AffectNetDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_dir = r"d:\AffectNet+\AffectNet+\human_annotated\train_set"
    train_manifest = os.path.join(base_dir, "final_manifest_v1_FROZEN.csv")

    epochs_to_avg = [26, 27, 28, 29, 30]
    checkpoints_dir = r"d:\AffectNet+\AffectNet+\checkpoints"

    print(f"Loading and averaging checkpoints: {epochs_to_avg}")
    state_dicts = []
    for ep in epochs_to_avg:
        ckpt_path = os.path.join(checkpoints_dir, f"epoch_{ep}.pt")
        ckpt = torch.load(ckpt_path, weights_only=False, map_location='cpu')
        state_dicts.append(ckpt['model_state'])

    avg_state_dict = {}
    for key in state_dicts[0].keys():
        avg_state_dict[key] = sum(sd[key] for sd in state_dicts) / len(state_dicts)

    model = efficientnet_b3()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 8)
    model.load_state_dict(avg_state_dict)
    model = model.to(device)

    print("Resetting BatchNorm statistics...")
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.reset_running_stats()
            module.momentum = None # Use simple average

    print("Setting up training dataloader for one pass...")
    dataset = AffectNetDataset(train_manifest, base_dir, mock_images=False, allowed_tiers=['gold', 'standard', 'soft_weighted', 'high_risk'])
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, prefetch_factor=4)

    model.train() # MUST be in train mode for BN to update

    print("Recalibrating BatchNorm (1 epoch forward pass)...")
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for imgs, _, _, _ in tqdm(dataloader, total=len(dataloader)):
            imgs = imgs.to(device)
            _ = model(imgs)

    save_path = os.path.join(checkpoints_dir, "avg_26_30.pt")
    torch.save({'model_state': model.state_dict()}, save_path)
    print(f"Averaged model with recalibrated BN saved to {save_path}")

if __name__ == '__main__':
    main()
