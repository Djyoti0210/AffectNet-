import os
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm

base_dir = r"d:\AffectNet+\AffectNet+\human_annotated\train_set"
val_dir = r"d:\AffectNet+\AffectNet+\human_annotated\validation_set"

# 1. Compute training frequencies (for Stage 3)
train_manifest = os.path.join(base_dir, "final_manifest_v1_FROZEN.csv")
train_df = pd.read_csv(train_manifest)
allowed_tiers = ["gold", "standard", "soft_weighted", "high_risk"]
train_df = train_df[train_df['training_tier'].isin(allowed_tiers)]
freqs = train_df['human_label'].value_counts(normalize=True).sort_index()
freqs_tensor = torch.tensor(freqs.values, dtype=torch.float32)
log_freqs = torch.log(freqs_tensor)

# 2. Load validation data
val_manifest = os.path.join(val_dir, "val_manifest.csv")
val_df = pd.read_csv(val_manifest)

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 3. Load model
print("Loading model...")
model = efficientnet_b3()
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 8)
ckpt = torch.load(r"d:\AffectNet+\AffectNet+\checkpoints\epoch_18.pt", weights_only=False)
model.load_state_dict(ckpt['model_state'])
model.to(device)
model.eval()

# 4. Extract logits for the whole validation set
print("Extracting raw logits from validation set...")
all_logits = []
all_labels = []

batch_size = 64
images = []
labels = []
with torch.no_grad():
    for idx, row in tqdm(val_df.iterrows(), total=len(val_df)):
        img_path = os.path.join(val_dir, row['image_path'])
        img = Image.open(img_path).convert('RGB')
        images.append(transform(img))
        labels.append(row['human_label'])
        
        if len(images) == batch_size or idx == len(val_df) - 1:
            batch_t = torch.stack(images).to(device)
            logits = model(batch_t)
            all_logits.append(logits.cpu())
            all_labels.extend(labels)
            images = []
            labels = []

all_logits = torch.cat(all_logits, dim=0)
all_labels = np.array(all_labels)

# 5. Fine Sweep
print("\n--- FINE SWEEP (tau: -0.3 to 0.3) ---")
taus = np.arange(-0.3, 0.35, 0.05)

for tau in taus:
    # adjusted = logit + tau * log(p)
    adjusted_logits = all_logits + tau * log_freqs
    preds = torch.argmax(adjusted_logits, dim=1).numpy()
    
    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds, average='macro')
    cm = confusion_matrix(all_labels, preds, labels=list(range(8)))
    per_class = cm.diagonal() / cm.sum(axis=1)
    
    print(f"tau={tau:.1f} | Acc: {acc*100:.2f}% | F1: {f1:.4f} | C0: {per_class[0]*100:.1f} | C1: {per_class[1]*100:.1f} | C2: {per_class[2]*100:.1f} | C3: {per_class[3]*100:.1f} | C4: {per_class[4]*100:.1f} | C5: {per_class[5]*100:.1f} | C6: {per_class[6]*100:.1f} | C7: {per_class[7]*100:.1f}")
