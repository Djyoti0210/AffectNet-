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

    print(f"Evaluating retrained head...")
    model = efficientnet_b3()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 8)
    
    ckpt = torch.load(r"d:\AffectNet+\AffectNet+\checkpoints\avg_26_30.pt", weights_only=False, map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    
    head_state = torch.load("retrained_head.pt")
    model.classifier[1].load_state_dict(head_state)
    
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

    torch.save(logits_no_tta, f"logits_retrained_no_tta.pt")
    torch.save(logits_tta, f"logits_retrained_tta.pt")
    print(f"Logits saved for retrained model.")

if __name__ == '__main__':
    main()
