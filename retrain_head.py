import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import numpy as np
from tqdm import tqdm

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading cached features...")
    cache = torch.load("train_features_cache.pt")
    features = cache['features']
    labels = cache['labels']
    soft_labels = cache['soft_labels']
    tiers = cache['tiers']

    print(f"Features shape: {features.shape}")

    freq_dict = {0: 74874, 1: 134415, 2: 25459, 3: 14090, 4: 6378, 5: 3803, 6: 24882, 7: 3752}
    total = sum(freq_dict.values())
    class_weights = {k: total / v for k, v in freq_dict.items()}

    print("Setting up WeightedRandomSampler...")
    sample_weights = np.zeros(len(labels), dtype=np.float32)
    labels_np = labels.numpy()
    for i in range(8):
        sample_weights[labels_np == i] = class_weights[i]

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(labels), replacement=True)

    dataset = TensorDataset(features, labels, soft_labels, tiers)
    dataloader = DataLoader(dataset, batch_size=4096, sampler=sampler, num_workers=4)

    model = nn.Linear(1536, 8).to(device)
    epochs = 10
    lr = 1e-3
    weight_decay = 1e-4
    print(f"Optimizer Config: epochs={epochs}, lr={lr}, weight_decay={weight_decay}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def combined_loss(logits, targets, soft_targets, tier_weights):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        ce_loss = (ce_loss * tier_weights).mean()
        
        log_probs = F.log_softmax(logits, dim=1)
        kl_loss = F.kl_div(log_probs, soft_targets, reduction='batchmean')
        
        return ce_loss + kl_loss

    print(f"Retraining head for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for feat, tgt, soft, tier in tqdm(dataloader):
            feat = feat.to(device)
            tgt = tgt.to(device)
            soft = soft.to(device)
            tier = tier.to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits = model(feat)
                loss = combined_loss(logits, tgt, soft, tier)
                
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/10, Loss: {total_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), "retrained_head.pt")
    print("Saved retrained head to retrained_head.pt")

if __name__ == '__main__':
    main()
