import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, recall_score
from torchvision.models import efficientnet_b3
from test_dataloader import AffectNetDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

def get_cf_preds(logits, labels, folds):
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

def eval_model(model_name, features, labels, folds, head_model=None):
    if head_model is None:
        logits = features
    else:
        with torch.no_grad():
            logits = head_model(features)
            
    preds = get_cf_preds(logits, labels, folds)
    acc = accuracy_score(labels, preds)
    bal_acc = balanced_accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    recalls = recall_score(labels, preds, average=None)
    return acc, bal_acc, f1, recalls

if __name__ == '__main__':
    df = pd.read_csv("val_clean_folds.csv")
    labels = df['human_label'].values
    folds = df['fold'].values
    
    print("Loading datasets...")
    dataset = AffectNetDataset(r"d:\AffectNet+\AffectNet+\human_annotated\validation_set\val_manifest.csv", r"d:\AffectNet+\AffectNet+\human_annotated\validation_set", mock_images=False, is_validation=True)
    dataset.df = df
    dataset.heavy_transform = dataset.std_transform
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    
    dataset_tta = AffectNetDataset(r"d:\AffectNet+\AffectNet+\human_annotated\validation_set\val_manifest.csv", r"d:\AffectNet+\AffectNet+\human_annotated\validation_set", mock_images=False, is_validation=True)
    dataset_tta.df = df
    dataset_tta.heavy_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
        dataset.std_transform
    ])
    dataloader_tta = DataLoader(dataset_tta, batch_size=128, shuffle=False, num_workers=0)

    device = torch.device('cuda')
    base_model = efficientnet_b3()
    base_model.classifier = nn.Identity()
    base_model.cuda()
    base_model.eval()
    
    def extract_feats(ckpt_path):
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
        
        feats = []
        feats_tta = []
        with torch.no_grad():
            for imgs, _, _, _ in dataloader:
                feats.append(base_model(imgs.cuda()).cpu())
            for imgs, _, _, _ in dataloader_tta:
                feats_tta.append(base_model(imgs.cuda()).cpu())
        return torch.cat(feats, dim=0), torch.cat(feats_tta, dim=0), head

    print("Extracting features for epoch 28...")
    f_28, _, h_28 = extract_feats(r"d:\AffectNet+\AffectNet+\checkpoints\epoch_28.pt")
    h_28.cpu()
    
    print("Extracting features for avg_26_30...")
    f_avg, f_avg_tta, h_avg = extract_feats(r"d:\AffectNet+\AffectNet+\checkpoints\avg_26_30.pt")
    h_avg.cpu()
    
    retrained_head = nn.Linear(1536, 8)
    retrained_head.load_state_dict(torch.load("retrained_head.pt", weights_only=False, map_location='cpu'))
    
    print("Running evaluations...")
    # Single Epoch 28
    logits_28 = h_28(f_28)
    acc1, bal1, f11, rec1 = eval_model("Single ep28", logits_28, labels, folds)
    
    # Avg
    logits_avg = h_avg(f_avg)
    acc2, bal2, f12, rec2 = eval_model("Avg", logits_avg, labels, folds)
    
    # Avg + TTA
    logits_avg_tta = h_avg(f_avg_tta)
    logits_avg_combined = (logits_avg + logits_avg_tta) / 2
    acc3, bal3, f13, rec3 = eval_model("Avg + TTA", logits_avg_combined, labels, folds)
    
    # Avg + Retrained Head
    logits_retrain = retrained_head(f_avg)
    acc4, bal4, f14, rec4 = eval_model("Retrained Head", logits_retrain, labels, folds)
    
    # Avg + Retrained Head + TTA
    logits_retrain_tta = retrained_head(f_avg_tta)
    logits_retrain_combined = (logits_retrain + logits_retrain_tta) / 2
    acc5, bal5, f15, rec5 = eval_model("Retrained Head + TTA", logits_retrain_combined, labels, folds)
    
    # Bootstrap for Head vs Avg on val_clean
    n_bootstraps = 1000
    np.random.seed(42)
    boot_diffs = []
    
    preds_avg = get_cf_preds(logits_avg_combined, labels, folds)
    preds_head = get_cf_preds(logits_retrain_combined, labels, folds)
    
    for i in range(n_bootstraps):
        idx = np.random.choice(len(labels), len(labels), replace=True)
        acc_a = accuracy_score(labels[idx], preds_avg[idx])
        acc_h = accuracy_score(labels[idx], preds_head[idx])
        boot_diffs.append(acc_h - acc_a)
        
    ci_lower = np.percentile(boot_diffs, 2.5) * 100
    ci_upper = np.percentile(boot_diffs, 97.5) * 100
    
    print("\nRESULTS on val_clean (3466 images)")
    print(f"{'Model':<25} | {'Acc':<7} | {'BalAcc':<7} | {'MacroF1':<7}")
    print("-" * 55)
    print(f"{'Single ep28':<25} | {acc1*100:.2f}% | {bal1*100:.2f}% | {f11:.4f}")
    print(f"{'Avg (26-30)':<25} | {acc2*100:.2f}% | {bal2*100:.2f}% | {f12:.4f}")
    print(f"{'Avg + TTA':<25} | {acc3*100:.2f}% | {bal3*100:.2f}% | {f13:.4f}")
    print(f"{'Retrained Head':<25} | {acc4*100:.2f}% | {bal4*100:.2f}% | {f14:.4f}")
    print(f"{'Retrained Head + TTA':<25} | {acc5*100:.2f}% | {bal5*100:.2f}% | {f15:.4f}")
    
    print(f"\nBootstrap CI (Head+TTA vs Avg+TTA): [{ci_lower:.2f}%, {ci_upper:.2f}%]")
