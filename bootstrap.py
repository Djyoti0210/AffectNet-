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

def get_cf_preds(logits):
    mask_0 = (folds == 0)
    mask_1 = (folds == 1)
    
    # tune on 0, predict 1
    best_tau_0 = 0.0
    best_f1 = 0.0
    for t in np.arange(-0.60, 0.20, 0.01):
        adj = logits[mask_0] + t * log_freqs
        p = torch.argmax(adj, dim=1).numpy()
        f1 = f1_score(labels[mask_0], p, average='macro')
        if f1 > best_f1:
            best_f1 = f1
            best_tau_0 = t
            
    # tune on 1, predict 0
    best_tau_1 = 0.0
    best_f1 = 0.0
    for t in np.arange(-0.60, 0.20, 0.01):
        adj = logits[mask_1] + t * log_freqs
        p = torch.argmax(adj, dim=1).numpy()
        f1 = f1_score(labels[mask_1], p, average='macro')
        if f1 > best_f1:
            best_f1 = f1
            best_tau_1 = t

    preds = np.zeros_like(labels)
    preds[mask_1] = torch.argmax(logits[mask_1] + best_tau_0 * log_freqs, dim=1).numpy()
    preds[mask_0] = torch.argmax(logits[mask_0] + best_tau_1 * log_freqs, dim=1).numpy()
    
    return preds

print("Generating cross-fitted predictions...")
logits_base = torch.load("logits_epoch_28_no_tta.pt")
preds_base = get_cf_preds(logits_base)

logits_avg = torch.load("logits_avg_26_30_no_tta.pt")
preds_avg = get_cf_preds(logits_avg)

logits_avg_tta = torch.load("logits_avg_26_30_tta.pt")
preds_avg_tta = get_cf_preds(logits_avg_tta)

logits_retrained = torch.load("logits_retrained_no_tta.pt")
preds_retrained = get_cf_preds(logits_retrained)

logits_retrained_tta = torch.load("logits_retrained_tta.pt")
preds_retrained_tta = get_cf_preds(logits_retrained_tta)

n_bootstraps = 1000
np.random.seed(42)

diffs_acc_avg = []
diffs_f1_avg = []
diffs_acc_tta = []
diffs_f1_tta = []

diffs_acc_ret = []
diffs_f1_ret = []
diffs_acc_ret_tta = []
diffs_f1_ret_tta = []

print("Running bootstrap...")
for _ in range(n_bootstraps):
    idx = np.random.choice(len(labels), len(labels), replace=True)
    y_true = labels[idx]
    
    pb = preds_base[idx]
    pa = preds_avg[idx]
    pat = preds_avg_tta[idx]
    
    acc_base = accuracy_score(y_true, pb)
    f1_base = f1_score(y_true, pb, average='macro')
    
    acc_avg = accuracy_score(y_true, pa)
    f1_avg = f1_score(y_true, pa, average='macro')
    
    acc_tta = accuracy_score(y_true, pat)
    f1_tta = f1_score(y_true, pat, average='macro')
    
    pr = preds_retrained[idx]
    prt = preds_retrained_tta[idx]
    
    acc_ret = accuracy_score(y_true, pr)
    f1_ret = f1_score(y_true, pr, average='macro')
    acc_ret_tta = accuracy_score(y_true, prt)
    f1_ret_tta = f1_score(y_true, prt, average='macro')
    
    diffs_acc_avg.append(acc_avg - acc_base)
    diffs_f1_avg.append(f1_avg - f1_base)
    diffs_acc_tta.append(acc_tta - acc_base)
    diffs_f1_tta.append(f1_tta - f1_base)
    
    diffs_acc_ret.append(acc_ret - acc_avg)
    diffs_f1_ret.append(f1_ret - f1_avg)
    diffs_acc_ret_tta.append(acc_ret_tta - acc_tta)
    diffs_f1_ret_tta.append(f1_ret_tta - f1_tta)

def get_ci(diffs):
    diffs = np.array(diffs)
    return np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)

ci_acc_avg = get_ci(diffs_acc_avg)
ci_f1_avg = get_ci(diffs_f1_avg)
ci_acc_tta = get_ci(diffs_acc_tta)
ci_f1_tta = get_ci(diffs_f1_tta)

ci_acc_ret = get_ci(diffs_acc_ret)
ci_f1_ret = get_ci(diffs_f1_ret)
ci_acc_ret_tta = get_ci(diffs_acc_ret_tta)
ci_f1_ret_tta = get_ci(diffs_f1_ret_tta)

print("\n--- BOOTSTRAP 95% CIs (Diff vs Single Epoch 28) ---")
print(f"Avg vs Base - Acc: [{ci_acc_avg[0]*100:.2f}%, {ci_acc_avg[1]*100:.2f}%]")
print(f"Avg vs Base - F1 : [{ci_f1_avg[0]:.4f}, {ci_f1_avg[1]:.4f}]")
print(f"Avg+TTA vs Base - Acc: [{ci_acc_tta[0]*100:.2f}%, {ci_acc_tta[1]*100:.2f}%]")
print(f"Avg+TTA vs Base - F1 : [{ci_f1_tta[0]:.4f}, {ci_f1_tta[1]:.4f}]")

print("\n--- BOOTSTRAP 95% CIs (Diff: Retrained vs Avg) ---")
print(f"Retrained vs Avg - Acc: [{ci_acc_ret[0]*100:.2f}%, {ci_acc_ret[1]*100:.2f}%]")
print(f"Retrained vs Avg - F1 : [{ci_f1_ret[0]:.4f}, {ci_f1_ret[1]:.4f}]")
print(f"Retrained+TTA vs Avg+TTA - Acc: [{ci_acc_ret_tta[0]*100:.2f}%, {ci_acc_ret_tta[1]*100:.2f}%]")
print(f"Retrained+TTA vs Avg+TTA - F1 : [{ci_f1_ret_tta[0]:.4f}, {ci_f1_ret_tta[1]:.4f}]")
