import torch
import torch.nn.functional as F
import pandas as pd
import hashlib
import os

def get_md5(file_path):
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

train_df = pd.read_csv(r"d:\AffectNet+\AffectNet+\human_annotated\train_set\final_manifest_v1_FROZEN.csv")
val_df = pd.read_csv(r"d:\AffectNet+\AffectNet+\human_annotated\validation_set\val_manifest.csv")

train_dir = r"d:\AffectNet+\AffectNet+\human_annotated\train_set"
val_dir = r"d:\AffectNet+\AffectNet+\human_annotated\validation_set"

import concurrent.futures

val_hashes = {}
for i, row in val_df.iterrows():
    h = get_md5(os.path.join(val_dir, row['image_path']))
    val_hashes[h] = (i, row['human_label'])

train_to_val_map = [] # (train_idx, val_idx, train_lbl, val_lbl)

def check_train_file(i):
    row = train_df.iloc[i]
    h = get_md5(os.path.join(train_dir, row['image_path']))
    if h in val_hashes:
        val_idx, val_lbl = val_hashes[h]
        return (i, val_idx, row['human_label'], val_lbl)
    return None

with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
    results = list(executor.map(check_train_file, range(len(train_df))))
    
for res in results:
    if res is not None:
        train_to_val_map.append(res)

print(f"Found {len(train_to_val_map)} exact MD5 matches.")

if len(train_to_val_map) > 0:
    cache = torch.load("train_features_cache.pt", weights_only=False)
    train_features = cache['features']
    
    # We also need validation features. We can extract them exactly as we did before.
    from torchvision.models import efficientnet_b3
    import torch.nn as nn
    from test_dataloader import AffectNetDataset
    from torch.utils.data import DataLoader

    model = efficientnet_b3()
    model.classifier = nn.Identity()
    ckpt = torch.load(r"d:\AffectNet+\AffectNet+\checkpoints\avg_26_30.pt", map_location='cpu', weights_only=False)
    state_dict = {k: v for k, v in ckpt['model_state'].items() if not k.startswith('classifier')}
    model.load_state_dict(state_dict)
    model.cuda()
    model.eval()

    dataset = AffectNetDataset(r"d:\AffectNet+\AffectNet+\human_annotated\validation_set\val_manifest.csv", val_dir, mock_images=False, is_validation=True)
    dataset.heavy_transform = dataset.std_transform
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

    val_features = []
    with torch.no_grad():
        for imgs, _, _, _ in dataloader:
            feat = model(imgs.cuda())
            val_features.append(feat.cpu())
    val_features = torch.cat(val_features, dim=0)

    train_features_norm = F.normalize(train_features.float(), p=2, dim=1)
    val_features_norm = F.normalize(val_features.float(), p=2, dim=1)
    
    cos_sims = []
    same_label = 0
    
    for train_idx, val_idx, train_lbl, val_lbl in train_to_val_map:
        sim = (train_features_norm[train_idx] * val_features_norm[val_idx]).sum().item()
        cos_sims.append(sim)
        if train_lbl == val_lbl:
            same_label += 1
            
    print(f"Cosine similarities: min={min(cos_sims):.4f}, mean={sum(cos_sims)/len(cos_sims):.4f}, max={max(cos_sims):.4f}")
    
    below_threshold = [s for s in cos_sims if s < 0.98]
    print(f"Pairs with < 0.98 similarity: {len(below_threshold)}")
    if len(below_threshold) > 0:
        print(f"Sample low similarities: {below_threshold[:5]}")
        
    print(f"Share of pairs with identical labels: {same_label}/{len(train_to_val_map)} ({same_label/len(train_to_val_map)*100:.2f}%)")
