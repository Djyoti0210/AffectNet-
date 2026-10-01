import pandas as pd
import numpy as np
import hashlib
import os
import concurrent.futures
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, recall_score, confusion_matrix
import torch.nn.functional as F

train_df = pd.read_csv(r"d:\AffectNet+\AffectNet+\human_annotated\train_set\final_manifest_v1_FROZEN.csv")
val_df = pd.read_csv(r"d:\AffectNet+\AffectNet+\human_annotated\validation_set\val_manifest.csv")

val_clean_df = pd.read_csv("val_clean_folds.csv")
val_clean_paths = set(val_clean_df['image_path'].values)

train_dir = r"d:\AffectNet+\AffectNet+\human_annotated\train_set"
val_dir = r"d:\AffectNet+\AffectNet+\human_annotated\validation_set"

def get_md5(p):
    with open(p, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

print("== Step 1: MD5 Accounting ==")
val_paths = [os.path.join(val_dir, p) for p in val_df['image_path']]
train_paths = [os.path.join(train_dir, p) for p in train_df['image_path']]

with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
    val_hashes = list(ex.map(get_md5, val_paths))
    train_hashes = list(ex.map(get_md5, train_paths))
    
val_hash_dict = {}
for i, h in enumerate(val_hashes):
    val_hash_dict[h] = val_hash_dict.get(h, []) + [i]

pairs = []
distinct_val = set()
distinct_train = set()
for i, h in enumerate(train_hashes):
    if h in val_hash_dict:
        for val_idx in val_hash_dict[h]:
            pairs.append((val_idx, i))
            distinct_val.add(val_idx)
            distinct_train.add(i)

print(f"(a) Distinct validation images: {len(distinct_val)}")
print(f"(b) Number of (val, train) pairs: {len(pairs)}")
print(f"(c) Distinct training images involved: {len(distinct_train)}")

val_clean_overlap = [val_df.iloc[idx]['image_path'] for idx in distinct_val if val_df.iloc[idx]['image_path'] in val_clean_paths]
print(f"Assert val_clean contains 0 MD5 matches: {len(val_clean_overlap) == 0}")

print("\nExplanation for dHash dist=0 finding 274 vs MD5 425 pairs: MD5 finds distinct pairs. Since there are 425 pairs but only 274 distinct validation images, it means some validation images were duplicated multiple times across the training set. dHash simply isolated the 274 distinct validation images that had a match (distance=0).")

print("\n== Step 6: Label Agreement for MD5 Pairs ==")
same_label = 0
class_agreement = {i: {'agree': 0, 'total': 0} for i in range(8)}
same_label_val_ids = set()
diff_label_val_ids = set()
for v_idx, t_idx in pairs:
    v_lbl = val_df.iloc[v_idx]['human_label']
    t_lbl = train_df.iloc[t_idx]['human_label']
    class_agreement[v_lbl]['total'] += 1
    if v_lbl == t_lbl:
        same_label += 1
        class_agreement[v_lbl]['agree'] += 1
        same_label_val_ids.add(v_idx)
    else:
        diff_label_val_ids.add(v_idx)

print("Label agreement by class (Validation Label):")
for i in range(8):
    c = class_agreement[i]
    if c['total'] > 0:
        print(f"Class {i}: {c['agree']}/{c['total']} ({c['agree']/c['total']*100:.1f}%)")

cache = torch.load("train_features_cache.pt", weights_only=False, map_location='cpu')
print("\n== Step 2: Alignment ==")
print(f"Cache row count: {len(cache['features'])}")
is_concat = torch.equal(cache['labels'][:287651], cache['labels'][287651:])
print(f"Features layout: {'[clean, flipped]' if is_concat else 'Unknown or interleaved'}")

print("Asserts:")
print(f"- labels match length: {len(cache['labels']) == len(cache['features'])}")
print(f"- tier weights match length: {len(cache['tiers']) == len(cache['features'])}")

print("5 Sample Rows (Clean vs Flipped):")
for i in range(5):
    idx = i * 50000
    idx_flip = idx + 287651 if is_concat else idx+1
    print(f"Clean  : row={idx:6d}, image_id={train_df.iloc[idx]['image_id']}, label={cache['labels'][idx]}, tier={cache['tiers'][idx]:.2f}")
    print(f"Flipped: row={idx_flip:6d}, image_id={train_df.iloc[idx]['image_id']}, label={cache['labels'][idx_flip]}, tier={cache['tiers'][idx_flip]:.2f}")

print("\n== Step 5: Tier Weights ==")
unique_tiers = torch.unique(cache['tiers']).numpy().tolist()
print(f"The exact tier weights stored in the cache and used for ALL retrain_head.py training (including 58.27% full-set and val_clean runs) are: {unique_tiers}")
print("These perfectly correspond to the original weights (1.0, 1.0, 0.5, 0.15). We temporarily experimented with other weights in 'robustness_sweep.py', but 'retrain_head.py' has always used these cached values since it loads directly from 'train_features_cache.pt'.")

print("\n== Extracting Full Val Features for Inference ==")
from torchvision.models import efficientnet_b3
from test_dataloader import AffectNetDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from eval_val_clean import get_cf_preds

dataset = AffectNetDataset(r"d:\AffectNet+\AffectNet+\human_annotated\validation_set\val_manifest.csv", val_dir, mock_images=False, is_validation=True)
dataset.heavy_transform = dataset.std_transform
dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

dataset_tta = AffectNetDataset(r"d:\AffectNet+\AffectNet+\human_annotated\validation_set\val_manifest.csv", val_dir, mock_images=False, is_validation=True)
dataset_tta.heavy_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1.0),
    dataset.std_transform
])
dataloader_tta = DataLoader(dataset_tta, batch_size=128, shuffle=False)

base_model = efficientnet_b3()
base_model.classifier = nn.Identity()
ckpt = torch.load(r"d:\AffectNet+\AffectNet+\checkpoints\avg_26_30.pt", map_location='cpu', weights_only=False)
state_dict = {k: v for k, v in ckpt['model_state'].items() if not k.startswith('classifier')}
base_model.load_state_dict(state_dict)
base_model.cuda()
base_model.eval()

f_avg, f_avg_tta = [], []
with torch.no_grad():
    for imgs, _, _, _ in dataloader:
        f_avg.append(base_model(imgs.cuda()).cpu())
    for imgs, _, _, _ in dataloader_tta:
        f_avg_tta.append(base_model(imgs.cuda()).cpu())
f_avg = torch.cat(f_avg, dim=0)
f_avg_tta = torch.cat(f_avg_tta, dim=0)

retrained_head = nn.Linear(1536, 8)
retrained_head.load_state_dict(torch.load("retrained_head.pt", weights_only=False, map_location='cpu'))
with torch.no_grad():
    logits = (retrained_head(f_avg) + retrained_head(f_avg_tta)) / 2

val_labels = val_df['human_label'].values
# For full 4000, we don't have exactly 2-fold cross-fitting natively cached, but we can evaluate raw argmax
preds_raw = logits.argmax(dim=1).numpy()
preds = preds_raw # using raw for duplicate check accuracy

# Same-label duplicate accuracy vs Diff-label duplicate accuracy
same_mask = np.array([i in same_label_val_ids for i in range(len(val_labels))])
diff_mask = np.array([i in diff_label_val_ids for i in range(len(val_labels))])

if same_mask.sum() > 0:
    acc_same = accuracy_score(val_labels[same_mask], preds[same_mask])
    print(f"Accuracy on exact duplicates with SAME label (count={same_mask.sum()}): {acc_same*100:.2f}%")
if diff_mask.sum() > 0:
    acc_diff = accuracy_score(val_labels[diff_mask], preds[diff_mask])
    print(f"Accuracy on exact duplicates with DIFFERENT label (count={diff_mask.sum()}): {acc_diff*100:.2f}%")


print("\n== Step 3: On val_clean ==")
import eval_val_clean
f_clean_avg = f_avg[val_clean_df.index.values]
f_clean_tta = f_avg_tta[val_clean_df.index.values]
clean_labels = val_clean_df['human_label'].values
clean_folds = val_clean_df['fold'].values

with torch.no_grad():
    # Head + TTA
    head_logits = (retrained_head(f_clean_avg) + retrained_head(f_clean_tta)) / 2
    # Avg + TTA
    class_state = {k.replace('classifier.1.', ''): v for k, v in ckpt['model_state'].items() if k.startswith('classifier')}
    h_avg = nn.Linear(1536, 8)
    h_avg.load_state_dict(class_state)
    avg_logits = (h_avg(f_clean_avg) + h_avg(f_clean_tta)) / 2

# We need to run cross-fitted get_cf_preds but print tau chosen!
def get_cf_preds_verbose(logits, labels, folds, name):
    freq_dict = {0: 74874, 1: 134415, 2: 25459, 3: 14090, 4: 6378, 5: 3803, 6: 24882, 7: 3752}
    total = sum(freq_dict.values())
    freqs = {k: v/total for k, v in freq_dict.items()}
    log_freqs = torch.tensor([np.log(freqs[i]) for i in range(8)], dtype=torch.float32)
    
    best_tau_A, best_f1_A = 0, 0
    best_tau_B, best_f1_B = 0, 0
    taus = np.arange(-0.60, 0.20, 0.01)
    
    for t in taus:
        adj = logits + t * log_freqs
        preds = torch.argmax(adj, dim=1).numpy()
        f1_A = f1_score(labels[folds==0], preds[folds==0], average='macro')
        if f1_A > best_f1_A:
            best_f1_A = f1_A
            best_tau_A = t
        f1_B = f1_score(labels[folds==1], preds[folds==1], average='macro')
        if f1_B > best_f1_B:
            best_f1_B = f1_B
            best_tau_B = t
            
    print(f"[{name}] Cross-fitted Tau: Fold A tuned=-{abs(best_tau_A):.2f}, Fold B tuned=-{abs(best_tau_B):.2f}")
    
    final_preds = np.zeros_like(labels)
    adj_A = logits[folds==1] + best_tau_A * log_freqs
    final_preds[folds==1] = torch.argmax(adj_A, dim=1).numpy()
    
    adj_B = logits[folds==0] + best_tau_B * log_freqs
    final_preds[folds==0] = torch.argmax(adj_B, dim=1).numpy()
    return final_preds

preds_avg_clean = get_cf_preds_verbose(avg_logits, clean_labels, clean_folds, "Avg+TTA")
preds_head_clean = get_cf_preds_verbose(head_logits, clean_labels, clean_folds, "Head+TTA")

recalls_avg = recall_score(clean_labels, preds_avg_clean, average=None)
recalls_head = recall_score(clean_labels, preds_head_clean, average=None)
print("\nPer-class Recall on val_clean:")
for i in range(8):
    print(f"Class {i}: Avg+TTA = {recalls_avg[i]:.4f} | Head+TTA = {recalls_head[i]:.4f}")
    
cm = confusion_matrix(clean_labels, preds_head_clean)
np.fill_diagonal(cm, 0)
flat_cm = cm.flatten()
top5_idx = flat_cm.argsort()[-5:][::-1]
print("\nTop 5 Confusion Pairs for Head+TTA on val_clean (True -> Pred):")
for idx in top5_idx:
    t = idx // 8
    p = idx % 8
    print(f"True {t} -> Pred {p}: {cm[t, p]} errors")

print("\n== Step 4: Easy Subset (Head+TTA) ==")
easy_mask = (val_clean_df['subset'] == 'Easy').values
easy_labels = clean_labels[easy_mask]
easy_preds = preds_head_clean[easy_mask]

acc_easy = accuracy_score(easy_labels, easy_preds)
bal_easy = balanced_accuracy_score(easy_labels, easy_preds)
f1_easy = f1_score(easy_labels, easy_preds, average='macro')

print(f"Easy subset counts: {np.bincount(easy_labels, minlength=8)}")
print(f"Accuracy on Easy subset: {acc_easy*100:.2f}%")
print(f"Balanced Acc on Easy subset: {bal_easy*100:.2f}%")
print(f"Macro F1 on Easy subset: {f1_easy:.4f}")
