import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv("val_folds.csv")
labels = df['human_label'].values
folds = df['fold'].values

freq_dict = {0: 74874, 1: 134415, 2: 25459, 3: 14090, 4: 6378, 5: 3803, 6: 24882, 7: 3752}
total = sum(freq_dict.values())
freqs = {k: v/total for k, v in freq_dict.items()}
log_freqs = torch.tensor([np.log(freqs[i]) for i in range(8)], dtype=torch.float32)

def evaluate_tau(logits, labels_subset, tau):
    adj = logits + tau * log_freqs
    preds = torch.argmax(adj, dim=1).numpy()
    acc = accuracy_score(labels_subset, preds)
    f1 = f1_score(labels_subset, preds, average='macro')
    return acc, f1

def find_best_tau(logits, labels_subset):
    best_tau = 0.0
    best_f1 = 0.0
    for tau in np.arange(-0.60, 0.20, 0.01):
        _, f1 = evaluate_tau(logits, labels_subset, tau)
        if f1 > best_f1:
            best_f1 = f1
            best_tau = tau
    return best_tau

conditions = [
    ("Epoch 28 (Single)", "logits_epoch_28_no_tta.pt"),
    ("Epoch 28 (Single) + TTA", "logits_epoch_28_tta.pt"),
    ("Epoch 26-30 (Avg)", "logits_avg_26_30_no_tta.pt"),
    ("Epoch 26-30 (Avg) + TTA", "logits_avg_26_30_tta.pt"),
    ("Avg + Retrained Head", "logits_retrained_no_tta.pt"),
    ("Avg + Retrained Head + TTA", "logits_retrained_tta.pt")
]

print(f"{'Condition':<25} | {'Raw Acc':<8} | {'Raw F1':<8} | {'CF Acc':<8} | {'CF F1':<8}")
print("-" * 65)

for name, logit_file in conditions:
    if not os.path.exists(logit_file):
        print(f"{name:<25} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8}")
        continue
        
    logits = torch.load(logit_file)
    
    # Raw
    raw_acc, raw_f1 = evaluate_tau(logits, labels, 0.0)
    
    # Cross-Fitted
    mask_0 = (folds == 0)
    mask_1 = (folds == 1)
    
    tau_0 = find_best_tau(logits[mask_0], labels[mask_0])
    tau_1 = find_best_tau(logits[mask_1], labels[mask_1])
    
    acc_on_1, f1_on_1 = evaluate_tau(logits[mask_1], labels[mask_1], tau_0)
    acc_on_0, f1_on_0 = evaluate_tau(logits[mask_0], labels[mask_0], tau_1)
    
    cf_acc = (acc_on_1 * sum(mask_1) + acc_on_0 * sum(mask_0)) / len(labels)
    cf_f1 = (f1_on_1 + f1_on_0) / 2.0
    
    print(f"{name:<25} | {raw_acc*100:>5.2f}% | {raw_f1:>6.4f} | {cf_acc*100:>5.2f}% | {cf_f1:>6.4f}")
    if name == "Epoch 26-30 (Avg) + TTA":
        print(f"  --> Fold A: Tuned tau={tau_0:.2f}, eval on Fold B (Acc: {acc_on_1*100:.2f}%, F1: {f1_on_1:.4f})")
        print(f"  --> Fold B: Tuned tau={tau_1:.2f}, eval on Fold A (Acc: {acc_on_0*100:.2f}%, F1: {f1_on_0:.4f})")
