import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, recall_score
from torchvision.models import efficientnet_b3
from test_dataloader import AffectNetDataset
from torch.utils.data import DataLoader

def main():
    print("Extracting ep28 features...")
    val_dir = r"d:\AffectNet+\AffectNet+\human_annotated\validation_set"
    val_df = pd.read_csv(r"d:\AffectNet+\AffectNet+\human_annotated\validation_set\val_manifest.csv")
    val_clean_df = pd.read_csv("val_clean_folds.csv")
    
    dataset = AffectNetDataset(r"d:\AffectNet+\AffectNet+\human_annotated\validation_set\val_manifest.csv", val_dir, mock_images=False, is_validation=True)
    dataset.heavy_transform = dataset.std_transform
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    
    base_model = efficientnet_b3()
    base_model.classifier = nn.Identity()
    base_model.cuda()
    base_model.eval()
    
    ckpt = torch.load(r"d:\AffectNet+\AffectNet+\checkpoints\epoch_28.pt", map_location='cpu', weights_only=False)
    state = {k: v for k, v in ckpt['model_state'].items() if not k.startswith('classifier')}
    class_state = {k.replace('classifier.1.', ''): v for k, v in ckpt['model_state'].items() if k.startswith('classifier')}
    base_model.load_state_dict(state)
    
    head = nn.Linear(1536, 8)
    head.load_state_dict(class_state)
    head.cuda()
    head.eval()
    
    f_28 = []
    with torch.no_grad():
        for imgs, _, _, _ in dataloader:
            f_28.append(base_model(imgs.cuda()).cpu())
    f_28 = torch.cat(f_28, dim=0)
    
    with torch.no_grad():
        head.cpu()
        logits_ep28 = head(f_28)
        
    id_to_idx = {row['image_id']: i for i, row in val_df.iterrows()}
    clean_indices = [id_to_idx[img_id] for img_id in val_clean_df['image_id']]
    
    ep28_clean_logits = logits_ep28[clean_indices]
    clean_labels = val_clean_df['human_label'].values
    clean_easy_mask = (val_clean_df['subset'] == 'Easy').values
    
    # Slice to Easy subset
    easy_logits = ep28_clean_logits[clean_easy_mask]
    easy_labels = clean_labels[clean_easy_mask]
    
    # Setup Frequencies
    freq_dict_corrected = {0: 74874, 1: 134415, 2: 25459, 3: 14090, 4: 6378, 5: 3803, 6: 24882, 7: 3752}
    freq_dict_swapped = {0: 134415, 1: 74874, 2: 25459, 3: 14090, 4: 6378, 5: 3803, 6: 24882, 7: 3752}
    
    def get_log_freqs(fd):
        t = sum(fd.values())
        return torch.tensor([np.log(v/t) for k, v in fd.items()], dtype=torch.float32)
        
    lf_corr = get_log_freqs(freq_dict_corrected)
    lf_swap = get_log_freqs(freq_dict_swapped)
    
    tau = -0.20
    preds_corr = torch.argmax(easy_logits + tau * lf_corr, dim=1).numpy()
    preds_swap = torch.argmax(easy_logits + tau * lf_swap, dim=1).numpy()
    
    acc_corr = accuracy_score(easy_labels, preds_corr)
    acc_swap = accuracy_score(easy_labels, preds_swap)
    
    rec_corr = recall_score(easy_labels, preds_corr, average=None)
    rec_swap = recall_score(easy_labels, preds_swap, average=None)
    
    class_names = {0:'Neutral', 1:'Happy', 2:'Sad', 3:'Surprise', 4:'Fear', 5:'Disgust', 6:'Anger', 7:'Contempt'}
    
    print("\n== val_clean Easy Subset (Single ep28, tau=-0.20) ==")
    print(f"Overall Accuracy (Swapped): {acc_swap*100:.2f}%")
    print(f"Overall Accuracy (Corrected): {acc_corr*100:.2f}%")
    print(f"Gain: {(acc_corr - acc_swap)*100:.2f}%\n")
    
    print("Per-Class Accuracy (Recall) on Easy Subset:")
    print(f"{'Class':<10} | {'Swapped':<10} | {'Corrected':<10} | {'Delta':<10}")
    print("-" * 45)
    for i in range(8):
        print(f"{class_names[i]:<10} | {rec_swap[i]*100:>8.2f}% | {rec_corr[i]*100:>8.2f}% | {(rec_corr[i] - rec_swap[i])*100:>8.2f}%")

if __name__ == '__main__':
    main()
