import os
import sys
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3
from test_dataloader import AffectNetDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms.functional as TF

def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate_model.py <checkpoint_path>")
        sys.exit(1)

    ckpt_path = sys.argv[1]
    model_name = os.path.basename(ckpt_path).split('.')[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Evaluating {model_name}...")
    model = efficientnet_b3()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 8)
    ckpt = torch.load(ckpt_path, weights_only=False, map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    model = model.to(device)
    model.eval()

    val_dir = r"d:\AffectNet+\AffectNet+\human_annotated\validation_set"
    val_manifest = os.path.join(val_dir, "val_manifest.csv")
    dataset = AffectNetDataset(val_manifest, val_dir, mock_images=False)
    dataset.heavy_transform = dataset.std_transform
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    logits_no_tta = []
    logits_tta = []

    with torch.no_grad(), torch.amp.autocast('cuda'):
        for imgs, _, _, _ in tqdm(dataloader, total=len(dataloader)):
            imgs = imgs.to(device)
            imgs_flipped = TF.hflip(imgs)
            
            logits_orig = model(imgs)
            logits_flip = model(imgs_flipped)
            
            logits_avg = (logits_orig + logits_flip) / 2.0
            
            logits_no_tta.append(logits_orig.cpu())
            logits_tta.append(logits_avg.cpu())

    logits_no_tta = torch.cat(logits_no_tta, dim=0)
    logits_tta = torch.cat(logits_tta, dim=0)

    torch.save(logits_no_tta, f"logits_{model_name}_no_tta.pt")
    torch.save(logits_tta, f"logits_{model_name}_tta.pt")
    print(f"Logits saved for {model_name}.")

if __name__ == '__main__':
    main()
