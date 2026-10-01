import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, recall_score
from torchvision.models import efficientnet_b3
from test_dataloader import AffectNetDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Define frequencies (Corrected)
freq_dict = {0: 74874, 1: 134415, 2: 25459, 3: 14090, 4: 6378, 5: 3803, 6: 24882, 7: 3752}
total_freq = sum(freq_dict.values())
freqs = {k: v/total_freq for k, v in freq_dict.items()}
log_freqs = torch.tensor([np.log(freqs[i]) for i in range(8)], dtype=torch.float32)

class_names = {0:'Neutral', 1:'Happy', 2:'Sad', 3:'Surprise', 4:'Fear', 5:'Disgust', 6:'Anger', 7:'Contempt'}

def get_cf_preds(logits, labels, folds):
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
            
    final_preds = np.zeros_like(labels)
    adj_A = logits[folds==1] + best_tau_A * log_freqs
    final_preds[folds==1] = torch.argmax(adj_A, dim=1).numpy()
    
    adj_B = logits[folds==0] + best_tau_B * log_freqs
    final_preds[folds==0] = torch.argmax(adj_B, dim=1).numpy()
    
    return final_preds, best_tau_A, best_tau_B

def print_metrics(preds, labels, name, extra=""):
    acc = accuracy_score(labels, preds)
    bal_acc = balanced_accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    recalls = recall_score(labels, preds, average=None)
    
    print(f"\n--- {name} {extra} ---")
    print(f"Overall Acc: {acc*100:.2f}% | BalAcc: {bal_acc*100:.2f}% | Macro F1: {f1:.4f}")
    for i in range(8):
        print(f"  Class {class_names[i]}: Recall = {recalls[i]*100:.2f}%")
        
def main():
    val_df = pd.read_csv(r"d:\AffectNet+\AffectNet+\human_annotated\validation_set\val_manifest.csv")
    val_clean_df = pd.read_csv("val_clean_folds.csv")
    
    # 1. Feature Extraction
    print("Extracting full val features...")
    val_dir = r"d:\AffectNet+\AffectNet+\human_annotated\validation_set"
    dataset = AffectNetDataset(r"d:\AffectNet+\AffectNet+\human_annotated\validation_set\val_manifest.csv", val_dir, mock_images=False, is_validation=True)
    dataset.heavy_transform = dataset.std_transform
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    
    dataset_tta = AffectNetDataset(r"d:\AffectNet+\AffectNet+\human_annotated\validation_set\val_manifest.csv", val_dir, mock_images=False, is_validation=True)
    dataset_tta.heavy_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
        dataset.std_transform
    ])
    dataloader_tta = DataLoader(dataset_tta, batch_size=128, shuffle=False, num_workers=0)
    
    base_model = efficientnet_b3()
    base_model.classifier = nn.Identity()
    base_model.cuda()
    base_model.eval()

    def get_feats_and_head(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if 'model_state' in ckpt:
            state = {k: v for k, v in ckpt['model_state'].items() if not k.startswith('classifier')}
            class_state = {k.replace('classifier.1.', ''): v for k, v in ckpt['model_state'].items() if k.startswith('classifier')}
        else:
            state = {k: v for k, v in ckpt.items() if not k.startswith('classifier')}
            class_state = {k.replace('classifier.1.', ''): v for k, v in ckpt.items() if k.startswith('classifier')}
        base_model.load_state_dict(state)
        head = nn.Linear(1536, 8)
        head.load_state_dict(class_state)
        head.cuda()
        head.eval()
        
        f, f_tta = [], []
        with torch.no_grad():
            for imgs, _, _, _ in dataloader:
                f.append(base_model(imgs.cuda()).cpu())
            for imgs, _, _, _ in dataloader_tta:
                f_tta.append(base_model(imgs.cuda()).cpu())
        return torch.cat(f, dim=0), torch.cat(f_tta, dim=0), head

    print("Processing epoch 28...")
    f_28, _, h_28 = get_feats_and_head(r"d:\AffectNet+\AffectNet+\checkpoints\epoch_28.pt")
    
    print("Processing avg_26_30...")
    f_avg, f_avg_tta, h_avg = get_feats_and_head(r"d:\AffectNet+\AffectNet+\checkpoints\avg_26_30.pt")
    
    retrained_head = nn.Linear(1536, 8)
    retrained_head.load_state_dict(torch.load("retrained_head.pt", weights_only=False, map_location='cpu'))
    
    # Generate full logits
    with torch.no_grad():
        h_28.cpu()
        h_avg.cpu()
        retrained_head.cpu()
        
        logits_ep28 = h_28(f_28)
        logits_avg = h_avg(f_avg)
        logits_avg_tta = h_avg(f_avg_tta)
        logits_avg_comb = (logits_avg + logits_avg_tta) / 2
        
        logits_head = retrained_head(f_avg)
        logits_head_tta = retrained_head(f_avg_tta)
        logits_head_comb = (logits_head + logits_head_tta) / 2

    # Map to val_clean
    id_to_idx = {row['image_id']: i for i, row in val_df.iterrows()}
    clean_indices = [id_to_idx[img_id] for img_id in val_clean_df['image_id']]
    
    clean_labels = val_clean_df['human_label'].values
    clean_folds = val_clean_df['fold'].values
    
    # Sliced logits
    ep28_clean = logits_ep28[clean_indices]
    avg_comb_clean = logits_avg_comb[clean_indices]
    head_comb_clean = logits_head_comb[clean_indices]

    # Part 1: val_clean evaluations
    print("\n\n====== CANONICAL VAL_CLEAN EVALUATION ======")
    preds, tA, tB = get_cf_preds(ep28_clean, clean_labels, clean_folds)
    print_metrics(preds, clean_labels, "Single ep28 (Cross-Fitted)", f"Taus: A={tA:.2f}, B={tB:.2f}")

    preds, tA, tB = get_cf_preds(avg_comb_clean, clean_labels, clean_folds)
    print_metrics(preds, clean_labels, "Avg+TTA (Cross-Fitted)", f"Taus: A={tA:.2f}, B={tB:.2f}")

    preds, tA, tB = get_cf_preds(head_comb_clean, clean_labels, clean_folds)
    print_metrics(preds, clean_labels, "Head+TTA (Cross-Fitted)", f"Taus: A={tA:.2f}, B={tB:.2f}")

    # Part 2: Easy subset (val_clean and full 4000)
    print("\n\n====== EASY SUBSET EVALUATIONS ======")
    # Full labels and subset flags
    full_labels = val_df['human_label'].values
    full_easy_mask = (val_df['subset'] == 'Easy').values
    clean_easy_mask = (val_clean_df['subset'] == 'Easy').values
    
    def apply_tau(logits, tau):
        return torch.argmax(logits + tau * log_freqs, dim=1).numpy()
        
    def eval_subset(preds, labels, mask, name):
        p = preds[mask]
        l = labels[mask]
        acc = accuracy_score(l, p)
        bal = balanced_accuracy_score(l, p)
        f1 = f1_score(l, p, average='macro')
        print(f"{name}: Acc {acc*100:.2f}%, BalAcc {bal*100:.2f}%, MacroF1 {f1:.4f}")

    models = [
        ("Single ep28", logits_ep28, ep28_clean),
        ("Avg+TTA", logits_avg_comb, avg_comb_clean),
        ("Head+TTA", logits_head_comb, head_comb_clean)
    ]
    
    for name, full_logits, clean_logits in models:
        print(f"\n-- {name} --")
        # (a) tau = 0
        p_full_t0 = apply_tau(full_logits, 0.0)
        p_clean_t0 = apply_tau(clean_logits, 0.0)
        eval_subset(p_full_t0, full_labels, full_easy_mask, f"Full 4000 (tau=0)")
        eval_subset(p_clean_t0, clean_labels, clean_easy_mask, f"val_clean (tau=0)")
        
        # (b) fixed tau = -0.20
        p_full_t20 = apply_tau(full_logits, -0.20)
        p_clean_t20 = apply_tau(clean_logits, -0.20)
        eval_subset(p_full_t20, full_labels, full_easy_mask, f"Full 4000 (tau=-0.20)")
        eval_subset(p_clean_t20, clean_labels, clean_easy_mask, f"val_clean (tau=-0.20)")
        
        # (c) cross-fitted tau
        # Since full 4000 doesn't have native folds, we can just print for val_clean
        p_clean_cf, _, _ = get_cf_preds(clean_logits, clean_labels, clean_folds)
        eval_subset(p_clean_cf, clean_labels, clean_easy_mask, f"val_clean (cross-fitted)")

if __name__ == '__main__':
    main()
