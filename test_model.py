import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

base_dir = r"d:\AffectNet+\AffectNet+\human_annotated\train_set"
sys.path.append(r"d:\AffectNet+\AffectNet+")
from test_dataloader import AffectNetDataset
from torch.utils.data import DataLoader

# Re-declare combined_loss here cleanly to avoid import mess if loss_test.py was modified
def combined_loss(logits, human_label, soft_label_normalized, tier_loss_weight, lambda_kl=1.0):
    ce_per_sample = F.cross_entropy(logits, human_label, reduction='none')
    weighted_ce = (ce_per_sample * tier_loss_weight).mean()
    log_probs = F.log_softmax(logits, dim=1)
    kl_per_sample = F.kl_div(log_probs, soft_label_normalized, reduction='none').sum(dim=1)
    kl_loss = kl_per_sample.mean()
    return weighted_ce + lambda_kl * kl_loss, weighted_ce.item(), kl_loss.item()

print("Loading Dataloader...")
manifest_path = os.path.join(base_dir, "final_manifest_v1_FROZEN.csv")
dataset = AffectNetDataset(manifest_path, base_dir, mock_images=True)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Fetch batch and replace mock zero-tensors with randn so gradients propagate fully
batch_imgs, batch_labels, batch_softs, batch_weights = next(iter(dataloader))
batch_imgs = torch.randn_like(batch_imgs) 

print("Loading Pretrained EfficientNet-B3...")
model = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)

# Replace head
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 8)

# Parameter Count
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params = total_params - trainable_params

print("\n--- PARAMETER REPORT ---")
print(f"Total Parameters     : {total_params:,}")
print(f"Trainable Parameters : {trainable_params:,}")
print(f"Frozen Parameters    : {frozen_params:,}")

# Forward Pass
print("\n--- FORWARD PASS ---")
logits = model(batch_imgs)
print(f"Batch input shape : {batch_imgs.shape}")
print(f"Logits shape      : {logits.shape}")

# Backward Pass
print("\n--- BACKWARD PASS ---")
tot, ce, kl = combined_loss(logits, batch_labels, batch_softs, batch_weights)
tot.backward()

# Check gradients on head
head_grad = model.classifier[1].weight.grad
head_mean = head_grad.abs().mean().item() if head_grad is not None else 0.0

# Check gradients on first conv layer (early backbone)
# EfficientNet features[0][0] is the first Conv2dNormActivation layer
first_conv_grad = model.features[0][0].weight.grad
first_conv_mean = first_conv_grad.abs().mean().item() if first_conv_grad is not None else 0.0

print(f"Head Layer (Linear) Mean Abs Grad : {head_mean:.8f}")
print(f"Early Layer (Conv2d) Mean Abs Grad: {first_conv_mean:.8f}")

if first_conv_mean > 0:
    print("\nSUCCESS: Gradients are successfully flowing through the entire EfficientNet backbone!")
else:
    print("\nERROR: Backbone gradients are zero or None. Fine-tuning is broken.")
