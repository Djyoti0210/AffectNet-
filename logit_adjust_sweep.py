import os
import sys
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

base_dir = r"d:\AffectNet+\AffectNet+\human_annotated\train_set"
val_dir = r"d:\AffectNet+\AffectNet+\human_annotated\validation_set"
sys.path.append(r"d:\AffectNet+\AffectNet+")
from test_dataloader import AffectNetDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_manifest = os.path.join(base_dir, "final_manifest_v1_FROZEN.csv")
df_train = pd.read_csv(train_manifest, usecols=['human_label'], low_memory=False)
counts = df_train['human_label'].value_counts().sort_index().values
freqs = counts / counts.sum()
freqs_tensor = torch.tensor(freqs, dtype=torch.float32).to(device)

model = efficientnet_b3()
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 8)
ckpt = torch.load(r"d:\AffectNet+\AffectNet+\checkpoints\best_model.pt", weights_only=False)
model.load_state_dict(ckpt['model_state'])
model = model.to(device)
model.eval()

val_manifest = os.path.join(val_dir, "val_manifest.csv")
dataset = AffectNetDataset(val_manifest, val_dir, mock_images=False)
dataset.heavy_transform = dataset.std_transform

dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)

all_logits = []
all_labels = []

print("Running single-pass inference over entire validation set...")
with torch.no_grad():
    for imgs, labels, _, _ in dataloader:
        imgs = imgs.to(device)
        with torch.amp.autocast('cuda'):
            logits = model(imgs)
        all_logits.append(logits.cpu())
        all_labels.extend(labels.numpy())

all_logits = torch.cat(all_logits, dim=0)
all_labels = np.array(all_labels)

taus = [0.4, 0.45, 0.5, 0.55, 0.6]
results = []
cm_tau_5 = None

print("Computing fine sweeps...")
for tau in taus:
    adj_logits = all_logits - tau * torch.log(freqs_tensor.cpu())
    preds = torch.argmax(adj_logits, dim=1).numpy()
    
    acc = accuracy_score(all_labels, preds)
    macro_f1 = f1_score(all_labels, preds, average='macro')
    
    cm = confusion_matrix(all_labels, preds, labels=list(range(8)))
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    if tau == 0.5: cm_tau_5 = per_class_acc
        
    results.append({
        'tau': tau,
        'acc': acc,
        'macro_f1': macro_f1
    })

print("\n--- FINE TAU SWEEP RESULTS ---")
print(f"{'Tau':<5} | {'Acc':<7} | {'MacroF1':<7}")
print("-" * 25)
for r in results:
    print(f"{r['tau']:<5.2f} | {r['acc']:.4f}  | {r['macro_f1']:.4f}")

class_names = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]
print("\n--- FULL 8-CLASS ACCURACY FOR TAU=0.5 ---")
for i in range(8):
    print(f"{class_names[i]:<10}: {cm_tau_5[i]*100:.2f}%")
