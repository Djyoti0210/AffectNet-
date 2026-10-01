import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

df = pd.read_csv("val_folds.csv")
labels = df['human_label'].values

freq_dict = {0: 74874, 1: 134415, 2: 25459, 3: 14090, 4: 6378, 5: 3803, 6: 24882, 7: 3752}
total = sum(freq_dict.values())
freqs = {k: v/total for k, v in freq_dict.items()}
log_freqs = torch.tensor([np.log(freqs[i]) for i in range(8)], dtype=torch.float32)

logits_no_tta = torch.load("logits_avg_26_30_no_tta.pt")
logits_tta = torch.load("logits_avg_26_30_tta.pt")

taus = [-0.30, -0.35, -0.40, -0.45, -0.50, -0.60]

print("=== avg_26_30.pt (No TTA) ===")
for t in taus:
    adj = logits_no_tta + t * log_freqs
    p = torch.argmax(adj, dim=1).numpy()
    acc = accuracy_score(labels, p)
    f1 = f1_score(labels, p, average='macro')
    print(f"Tau={t:>5.2f} | Acc={acc*100:>5.2f}% | F1={f1:>6.4f}")

print("\n=== avg_26_30.pt (With TTA) ===")
for t in taus:
    adj = logits_tta + t * log_freqs
    p = torch.argmax(adj, dim=1).numpy()
    acc = accuracy_score(labels, p)
    f1 = f1_score(labels, p, average='macro')
    print(f"Tau={t:>5.2f} | Acc={acc*100:>5.2f}% | F1={f1:>6.4f}")

print("\n=== Per-Class Accuracy at Tau=-0.40 (With TTA) ===")
adj = logits_tta - 0.40 * log_freqs
p = torch.argmax(adj, dim=1).numpy()
cm = confusion_matrix(labels, p)
acc_per_class = cm.diagonal() / cm.sum(axis=1)
classes = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']
for i, c in enumerate(classes):
    print(f"{c:<10}: {acc_per_class[i]*100:.2f}%")
