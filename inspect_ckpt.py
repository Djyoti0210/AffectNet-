import torch

ckpt_path = r"d:\AffectNet+\AffectNet+\checkpoints\best_model.pt"
ckpt = torch.load(ckpt_path, weights_only=False, map_location='cpu')

print(f"File: {ckpt_path}")
print("--- METADATA ---")
print(f"Global Epoch: {ckpt.get('global_epoch', 'MISSING')}")
print(f"Stage Index: {ckpt.get('stage_idx', 'MISSING')}")
print(f"Epoch in Stage: {ckpt.get('epoch_in_stage', 'MISSING')}")
print(f"Saved Best Acc: {ckpt.get('best_acc', 'MISSING')}")

print("\n--- MODEL WEIGHT CHECKSUM ---")
params = list(ckpt['model_state'].values())
param_sum = sum(p.sum().item() for p in params)
print(f"Model parameters checksum: {param_sum:.4f}")

ckpt_28_path = r"d:\AffectNet+\AffectNet+\checkpoints\epoch_28.pt"
ckpt_28 = torch.load(ckpt_28_path, weights_only=False, map_location='cpu')
params_28 = list(ckpt_28['model_state'].values())
param_sum_28 = sum(p.sum().item() for p in params_28)
print(f"\nepoch_28.pt checksum: {param_sum_28:.4f}")

if param_sum == param_sum_28:
    print("\nCONFIRMED: best_model.pt is an exact binary match for epoch_28.pt")
else:
    print("\nWARNING: Checksums do not match!")
