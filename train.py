import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np

base_dir = r"d:\AffectNet+\AffectNet+\human_annotated\train_set"
val_dir = r"d:\AffectNet+\AffectNet+\human_annotated\validation_set"
sys.path.append(r"d:\AffectNet+\AffectNet+")
from test_dataloader import AffectNetDataset

# Configuration
BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 2
LAMBDA_KL = 1.0
BASE_LR_HEAD = 3e-4
BASE_LR_BACKBONE = 3e-5
WEIGHT_DECAY = 1e-4

curriculum_stages = [
    {"name": "Stage 1", "tiers": ["gold"], "epochs": 6, "lr_mult": 1.0},
    {"name": "Stage 2", "tiers": ["gold", "standard"], "epochs": 10, "lr_mult": 0.5},
    {"name": "Stage 3", "tiers": ["gold", "standard", "soft_weighted", "high_risk"], "epochs": 14, "lr_mult": 0.5}
]

def combined_loss(logits, human_label, soft_label_normalized, tier_loss_weight, lambda_kl=1.0, is_stage3=False, freqs_tensor=None):
    if is_stage3 and freqs_tensor is not None:
        adjusted_logits = logits + 0.5 * torch.log(freqs_tensor)
        ce_per_sample = F.cross_entropy(adjusted_logits, human_label, reduction='none')
    else:
        ce_per_sample = F.cross_entropy(logits, human_label, reduction='none')
    # Contempt (Class 7) targeted boost
    contempt_mask = (human_label == 7).float()
    contempt_boost = 1.0 + (contempt_mask * 3.0) # 4.0 multiplier for Contempt, 1.0 for everything else
    
    weighted_ce = (ce_per_sample * tier_loss_weight * contempt_boost).mean()
    log_probs = F.log_softmax(logits, dim=1)
    kl_per_sample = F.kl_div(log_probs, soft_label_normalized, reduction='none').sum(dim=1)
    kl_loss = kl_per_sample.mean()
    return weighted_ce + lambda_kl * kl_loss, weighted_ce.item(), kl_loss.item()

def validate(model, device, epoch, stage_name):
    print("\n--- Running Validation ---", flush=True)
    val_manifest = os.path.join(val_dir, "val_manifest.csv")
    
    dataset = AffectNetDataset(val_manifest, val_dir, mock_images=False)
    # Monkey-patch: force validation loader to bypass RandAugment completely
    dataset.heavy_transform = dataset.std_transform
    
    # 2x batch size for inference speed since we don't hold backward graphs in memory
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=4, pin_memory=True)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels, _, _ in dataloader:
            imgs = imgs.to(device)
            with torch.amp.autocast('cuda'):
                logits = model(imgs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(8)))
    # Add small epsilon to avoid divide-by-zero if somehow a class is entirely missing in a broken dataset
    per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-8)
    
    print(f"Validation Epoch {epoch} | Overall Acc: {acc:.4f} | Macro F1: {f1:.4f}", flush=True)
    
    log_row = {"epoch": epoch, "stage": stage_name, "acc": acc, "macro_f1": f1}
    for c in range(8):
        log_row[f"class_{c}_acc"] = per_class_acc[c]
        
    df_log = pd.DataFrame([log_row])
    log_file = r"d:\AffectNet+\AffectNet+\validation_logs.csv"
    if not os.path.exists(log_file):
        df_log.to_csv(log_file, index=False)
    else:
        df_log.to_csv(log_file, mode='a', header=False, index=False)
        
    model.train()
    return acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--stop_after_epoch", type=int, default=None, help="Kill training after X overall epochs for testing")
    parser.add_argument("--debug_fast_epoch", action="store_true", help="Only run 5 steps per epoch for fast testing")
    parser.add_argument("--disable_sampler", action="store_true", help="Disable WeightedRandomSampler")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = torch.amp.GradScaler('cuda')
    
    print("Initializing architecture...")
    model = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 8)
    model = model.to(device)
    
    checkpoint_path = r"d:\AffectNet+\AffectNet+\checkpoints\latest.pt"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    start_stage_idx = 0
    start_epoch_in_stage = 0
    global_epoch = 0
    best_acc = 0.0
    
    if args.resume and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        start_stage_idx = ckpt['stage_idx']
        start_epoch_in_stage = ckpt['epoch_in_stage']
        global_epoch = ckpt['global_epoch']
        best_acc = ckpt.get('best_acc', 0.0)
        scaler.load_state_dict(ckpt['scaler_state'])
        print(f"Successfully Resumed! Picked up at Stage {start_stage_idx+1}, Epoch {start_epoch_in_stage+1} (Global Epoch {global_epoch+1})")
    
    for stage_idx in range(start_stage_idx, len(curriculum_stages)):
        stage_config = curriculum_stages[stage_idx]
        print(f"\n{'='*50}\nStarting {stage_config['name']} (Tiers: {stage_config['tiers']})\n{'='*50}")
        
        manifest_path = os.path.join(base_dir, "final_manifest_v1_FROZEN.csv")
        dataset = AffectNetDataset(manifest_path, base_dir, mock_images=False, allowed_tiers=stage_config['tiers'])
        
        if args.disable_sampler:
            print(">>> WARNING: WeightedRandomSampler is DISABLED for this run! <<<")
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        else:
            weights = torch.tensor(dataset.df['class_sample_weight'].values, dtype=torch.double)
            sampler = WeightedRandomSampler(weights=weights, num_samples=len(dataset), replacement=True)
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)
        
        head_lr = BASE_LR_HEAD * stage_config['lr_mult']
        backbone_lr = BASE_LR_BACKBONE * stage_config['lr_mult']
        
        # For Stage 3 logit adjustment
        train_manifest = os.path.join(base_dir, "final_manifest_v1_FROZEN.csv")
        df_train = pd.read_csv(train_manifest, usecols=['human_label'], low_memory=False)
        counts = df_train['human_label'].value_counts().sort_index().values
        freqs_tensor = torch.tensor(counts / counts.sum(), dtype=torch.float32).to(device)
        is_stage3 = (stage_idx == 2)
        
        optimizer = torch.optim.AdamW([
            {'params': model.features.parameters(), 'lr': backbone_lr},
            {'params': model.classifier.parameters(), 'lr': head_lr}
        ], weight_decay=WEIGHT_DECAY)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage_config['epochs'])
        
        # Hydrate optimizer and scheduler states ONLY if we are actively resuming this specific stage
        if args.resume and os.path.exists(checkpoint_path) and stage_idx == start_stage_idx:
            optimizer.load_state_dict(ckpt['optimizer_state'])
            scheduler.load_state_dict(ckpt['scheduler_state'])
            print("=> Restored optimizer momentum and learning rate scheduler exact state.")
        
        model.train()
        
        for epoch in range(start_epoch_in_stage, stage_config['epochs']):
            print(f"\n[Global Epoch {global_epoch+1}] {stage_config['name']} Epoch {epoch+1}/{stage_config['epochs']}")
            
            optimizer.zero_grad()
            for step, (imgs, labels, softs, batch_weights) in enumerate(dataloader):
                imgs, labels = imgs.to(device), labels.to(device)
                softs, batch_weights = softs.to(device), batch_weights.to(device)
                
                with torch.amp.autocast('cuda'):
                    logits = model(imgs)
                    tot_loss, ce, kl = combined_loss(logits, labels, softs, batch_weights, LAMBDA_KL, is_stage3, freqs_tensor)
                    loss = tot_loss / GRADIENT_ACCUMULATION_STEPS
                    
                scaler.scale(loss).backward()
                
                if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                if step % 50 == 0:
                    print(f"Step {step}/{len(dataloader)} | Loss: {tot_loss.item():.4f} (CE: {ce:.4f}, KL: {kl:.4f})", flush=True)
                    
                if args.debug_fast_epoch and step >= 10:
                    break # Fast forward for tests
            
            scheduler.step()
            
            # Epoch Validation Hook
            val_acc = validate(model, device, global_epoch+1, stage_config['name'])
            
            # Save strictly after successful validation
            checkpoint = {
                'stage_idx': stage_idx,
                'epoch_in_stage': epoch + 1,
                'global_epoch': global_epoch + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'scaler_state': scaler.state_dict(),
                'best_acc': best_acc
            }
            torch.save(checkpoint, checkpoint_path)
            torch.save(checkpoint, checkpoint_path.replace("latest.pt", f"epoch_{global_epoch+1}.pt"))
            print(f"Checkpoint strictly saved to {checkpoint_path} and epoch_{global_epoch+1}.pt")
            
            # Save Best Model separately
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(checkpoint, r"d:\AffectNet+\AffectNet+\checkpoints\best_model.pt")
                with open(r"d:\AffectNet+\AffectNet+\checkpoints\best_log.txt", "w") as f:
                    f.write(f"Best Model: Global Epoch {global_epoch+1} | Acc: {best_acc:.4f}\n")
                print(">>> New Best Model Logged! <<<")
            
            global_epoch += 1
            
            if args.stop_after_epoch is not None and global_epoch >= args.stop_after_epoch:
                print(f"--- Terminating early at Global Epoch {global_epoch} as requested via test flag ---")
                sys.exit(0)
                
        # Reset inner start epoch for subsequent stages so they start at 0
        start_epoch_in_stage = 0

if __name__ == "__main__":
    main()
