import os
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

base_dir = r"d:\AffectNet+\AffectNet+\human_annotated"
val_dir = os.path.join(base_dir, "validation_set")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading epoch_28.pt...")
model = efficientnet_b3()
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 8)
ckpt = torch.load(r"d:\AffectNet+\AffectNet+\checkpoints\epoch_28.pt", weights_only=False)
model.load_state_dict(ckpt['model_state'])
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_manifest = os.path.join(val_dir, "val_manifest.csv")
df = pd.read_csv(val_manifest)

# Get 50 Contempt (7) and 50 Happy (0) images
contempt_df = df[df['human_label'] == 7].head(50)
happy_df = df[df['human_label'] == 0].head(50)

def get_mean_abs_logits(sub_df):
    logits_list = []
    with torch.no_grad():
        for _, row in sub_df.iterrows():
            img_path = os.path.join(val_dir, row['image_path'])
            img = Image.open(img_path).convert('RGB')
            img_t = transform(img).unsqueeze(0).to(device)
            logits = model(img_t)
            logits_list.append(logits.cpu().numpy())
    
    logits_array = np.concatenate(logits_list, axis=0) # Shape: (50, 8)
    mean_abs = np.mean(np.abs(logits_array))
    return mean_abs

contempt_mag = get_mean_abs_logits(contempt_df)
happy_mag = get_mean_abs_logits(happy_df)

print(f"Mean Absolute Logit Magnitude for Contempt Images: {contempt_mag:.4f}")
print(f"Mean Absolute Logit Magnitude for Happy Images: {happy_mag:.4f}")
