import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
import subprocess
import threading

sys.path.append(r"d:\AffectNet+\AffectNet+")
from test_dataloader import AffectNetDataset

gpu_utils = []
def sample_gpu():
    while getattr(threading.current_thread(), "do_run", True):
        try:
            res = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits']).decode('utf-8')
            gpu_utils.append(int(res.strip().split('\n')[0]))
        except:
            pass
        time.sleep(3)

def main():
    print("Setting up Dataset and Dataloader...", flush=True)
    base_dir = r"d:\AffectNet+\AffectNet+\human_annotated\train_set"
    manifest_path = os.path.join(base_dir, "final_manifest_v1_FROZEN.csv")
    dataset = AffectNetDataset(manifest_path, base_dir, mock_images=False, allowed_tiers=["gold", "standard", "soft_weighted", "high_risk"])

    weights = torch.tensor(dataset.df['class_sample_weight'].values, dtype=torch.double)
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(dataset), replacement=True)

    # THE NEW CONFIG
    dataloader = DataLoader(
        dataset, 
        batch_size=32, 
        sampler=sampler, 
        num_workers=4, 
        pin_memory=True, 
        prefetch_factor=4,
        persistent_workers=True
    )

    print("Initializing Model...", flush=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 8)
    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.CrossEntropyLoss()

    print("Starting GPU Sampler Thread...", flush=True)
    sampler_thread = threading.Thread(target=sample_gpu)
    sampler_thread.start()

    print("Starting Smoke Test (300 steps)...", flush=True)
    start_time = time.time()
    try:
        for step, (imgs, labels, _, _) in enumerate(dataloader):
            if step >= 300:
                break
                
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits = model(imgs)
                loss = criterion(logits, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if step % 50 == 0:
                print(f"Step {step}/300 completed.", flush=True)
    except Exception as e:
        print(f"SMOKE TEST FAILED: {e}", flush=True)
        
    sampler_thread.do_run = False
    sampler_thread.join()

    end_time = time.time()
    print(f"\nSmoke Test Completed in {end_time - start_time:.2f} seconds.", flush=True)
    if gpu_utils:
        print(f"GPU Utilization: Min {min(gpu_utils)}%, Max {max(gpu_utils)}%, Avg {sum(gpu_utils)/len(gpu_utils):.1f}%", flush=True)
    else:
        print("Could not collect GPU stats.", flush=True)

if __name__ == '__main__':
    main()
