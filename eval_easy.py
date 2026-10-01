import os
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

freq_dict = {
    0: 134415,
    1: 74874,
    2: 25459,
    3: 14090,
    4: 6378,
    5: 3803,
    6: 24882,
    7: 3752
}
total = sum(freq_dict.values())
freqs = {k: v/total for k, v in freq_dict.items()}
log_freqs = torch.tensor([np.log(freqs[i]) for i in range(8)], dtype=torch.float32).to(device)

base_dir = r"d:\AffectNet+\AffectNet+\human_annotated"
val_dir = os.path.join(base_dir, "validation_set")

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

easy_df = df[df['subset'] == 'Easy']
print(f"Total Easy images: {len(easy_df)}")

all_preds = []
all_labels = []

tau = 0.0

print("Running inference on Easy subset...")
with torch.no_grad():
    for _, row in easy_df.iterrows():
        img_path = os.path.join(val_dir, row['image_path'])
        img = Image.open(img_path).convert('RGB')
        img_t = transform(img).unsqueeze(0).to(device)
        logits = model(img_t)
        
        adj_logits = logits + tau * log_freqs
        pred = torch.argmax(adj_logits, dim=1).item()
        
        all_preds.append(pred)
        all_labels.append(row['human_label'])

acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='macro')

print(f"--- EASY SUBSET RESULTS (tau={tau}) ---")
print(f"Overall Accuracy: {acc*100:.2f}%")
print(f"Macro F1: {f1:.4f}")
