import os
import sys
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from collections import Counter

# Load manifest
val_dir = r"d:\AffectNet+\AffectNet+\human_annotated\validation_set"
manifest_path = os.path.join(val_dir, "val_manifest.csv")
df = pd.read_csv(manifest_path)

# Filter contempt
contempt_df = df[df['human_label'] == 7].head(50) # Pick first 50

# Transform
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(ckpt_path):
    model = efficientnet_b3()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 8)
    state = torch.load(ckpt_path, weights_only=False)
    if 'model_state' in state:
        model.load_state_dict(state['model_state'])
    elif 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

print("Loading Epoch 23 Model...")
model_ep12 = load_model(r"d:\AffectNet+\AffectNet+\checkpoints\epoch_23.pt")
print("Loading Epoch 25 Model...")
model_ep22 = load_model(r"d:\AffectNet+\AffectNet+\checkpoints\epoch_25.pt")

ep12_probs = []
ep22_probs = []
ep22_preds = []

print("Running 50 Contempt images...")
with torch.no_grad():
    for idx, row in contempt_df.iterrows():
        img_path = os.path.join(val_dir, row['image_path'])
        img = Image.open(img_path).convert('RGB')
        img_t = transform(img).unsqueeze(0).to(device)
        
        logits12 = model_ep12(img_t)
        probs12 = torch.softmax(logits12, dim=1)[0]
        ep12_probs.append(probs12[7].item())
        
        logits22 = model_ep22(img_t)
        probs22 = torch.softmax(logits22, dim=1)[0]
        ep22_probs.append(probs22[7].item())
        ep22_preds.append(torch.argmax(logits22, dim=1).item())

print(f"Epoch 23 Average Contempt Confidence: {np.mean(ep12_probs)*100:.2f}%")
print(f"Epoch 25 Average Contempt Confidence: {np.mean(ep22_probs)*100:.2f}%")

class_names = {0: "Happy", 1: "Neutral", 2: "Sad", 3: "Surprise", 4: "Fear", 5: "Disgust", 6: "Anger", 7: "Contempt"}
c = Counter(ep22_preds)
print("\nEpoch 25 Predictions for these 50 Contempt images:")
for k, v in c.items():
    print(f"  {class_names[k]}: {v} times")
