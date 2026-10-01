import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import sys

def get_cf_preds(logits):
    df = pd.read_csv("val_folds.csv")
    labels = df['human_label'].values
    folds = df['fold'].values
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
            
    final_preds = np.zeros_like(labels)
    adj_A = logits[folds==1] + best_tau_A * log_freqs
    final_preds[folds==1] = torch.argmax(adj_A, dim=1).numpy()
    
    adj_B = logits[folds==0] + best_tau_B * log_freqs
    final_preds[folds==0] = torch.argmax(adj_B, dim=1).numpy()
    
    return final_preds


def train_head(seed, custom_tier_weights_map=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cache = torch.load("train_features_cache.pt", weights_only=False)
    features = cache['features']
    labels = cache['labels']
    soft_labels = cache['soft_labels']
    tiers = cache['tiers']
    
    if custom_tier_weights_map:
        new_tiers = tiers.clone()
        for old_w, new_w in custom_tier_weights_map.items():
            new_tiers[tiers == old_w] = new_w
        tiers = new_tiers

    labels_np = labels.numpy()
    c_indices = [np.where(labels_np == i)[0] for i in range(8)]
    target_samples = len(labels) // 8

    model = nn.Linear(1536, 8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def combined_loss(logits, targets, soft_targets, tier_weights):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        ce_loss = (ce_loss * tier_weights).mean()
        log_probs = F.log_softmax(logits, dim=1)
        kl_loss = F.kl_div(log_probs, soft_targets, reduction='batchmean')
        return ce_loss + kl_loss

    for epoch in range(10):
        model.train()
        indices = np.concatenate([np.random.choice(c_idx, size=target_samples, replace=True) for c_idx in c_indices])
        np.random.shuffle(indices)
        
        for i in range(0, len(indices), 4096):
            batch_idx = indices[i:i+4096]
            feat = features[batch_idx].to(device)
            tgt = labels[batch_idx].to(device)
            soft = soft_labels[batch_idx].to(device)
            tier = tiers[batch_idx].to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits = model(feat)
                loss = combined_loss(logits, tgt, soft, tier)
            loss.backward()
            optimizer.step()
            
    model.eval()
    return model

from torchvision.models import efficientnet_b3
from test_dataloader import AffectNetDataset

def main():
    device = torch.device('cuda')
    
    print("Extracting Val Features...", flush=True)
    val_model = efficientnet_b3()
    val_model.classifier = nn.Identity()
    ckpt = torch.load(r"d:\AffectNet+\AffectNet+\checkpoints\avg_26_30.pt", map_location='cpu', weights_only=False)
    state_dict = {k: v for k, v in ckpt['model_state'].items() if not k.startswith('classifier')}
    val_model.load_state_dict(state_dict)
    val_model.to(device)
    val_model.eval()
    
    dataset = AffectNetDataset(r"d:\AffectNet+\AffectNet+\human_annotated\validation_set\val_manifest.csv", r"d:\AffectNet+\AffectNet+\human_annotated\validation_set", mock_images=False, is_validation=True)
    dataset.heavy_transform = dataset.std_transform
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    
    val_features = []
    df = pd.read_csv("val_folds.csv")
    labels_val = df['human_label'].values
    
    with torch.no_grad():
        for imgs, lbls, _, _ in dataloader:
            imgs = imgs.to(device)
            val_features.append(val_model(imgs).cpu())
            
    val_features = torch.cat(val_features)
    print(f"Val features done: {val_features.shape}", flush=True)
    
    results = []
    print("Running seeds with original weights (1.0, 1.0, 0.5, 0.15)", flush=True)
    for seed in [42, 100, 2026]:
        head = train_head(seed, None)
        head.to('cpu')
        with torch.no_grad():
            logits = head(val_features)
            preds = get_cf_preds(logits)
            acc = accuracy_score(labels_val, preds)
            f1 = f1_score(labels_val, preds, average='macro')
            results.append({'seed': seed, 'weights': 'original', 'acc': acc, 'f1': f1})
            print(f"Seed {seed}: Acc={acc*100:.2f}%, F1={f1:.4f}", flush=True)
            
    print("Running seeds with alternative weights (1.0, 0.5, 0.2, 0.05)", flush=True)
    alt_map = {0.5: 0.2, 0.15: 0.05}
    for seed in [42, 100, 2026]:
        head = train_head(seed, alt_map)
        head.to('cpu')
        with torch.no_grad():
            logits = head(val_features)
            preds = get_cf_preds(logits)
            acc = accuracy_score(labels_val, preds)
            f1 = f1_score(labels_val, preds, average='macro')
            results.append({'seed': seed, 'weights': 'alt', 'acc': acc, 'f1': f1})
            print(f"Seed {seed} (Alt Weights): Acc={acc*100:.2f}%, F1={f1:.4f}", flush=True)

if __name__ == '__main__':
    main()
