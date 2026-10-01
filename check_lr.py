import torch
from torchvision.models import efficientnet_b3

model = efficientnet_b3()
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 8)
optimizer = torch.optim.AdamW([
    {'params': model.features.parameters(), 'lr': 3e-5},
    {'params': model.classifier.parameters(), 'lr': 3e-4}
], weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6)

for epoch in range(1, 7):
    lrs = scheduler.get_last_lr()
    print(f"Start of Epoch {epoch}: Backbone LR = {lrs[0]:.7f} | Head LR = {lrs[1]:.7f}")
    optimizer.step()
    scheduler.step()
