import torch
import torch.nn.functional as F

# 1. Setup Synthetic Data
# 8 classes. Contempt (index 7) is a minority class (e.g. 1.3% frequency).
# Happy (index 1) is a majority class (e.g. 47% frequency).
freqs = torch.tensor([0.05, 0.47, 0.10, 0.10, 0.10, 0.10, 0.067, 0.013])
tau = 0.5

# Imagine a perfectly uniform logit output of 0.0 for all classes.
logits = torch.zeros(1, 8) 

# Example: The true label is Contempt (minority)
label_minority = torch.tensor([7]) 

# 2. Raw Loss (No Adjustment)
raw_ce = F.cross_entropy(logits, label_minority).item()

# 3. OLD Buggy Adjustment (-)
buggy_logits = logits - tau * torch.log(freqs)
buggy_ce = F.cross_entropy(buggy_logits, label_minority).item()

# 4. NEW Corrected Adjustment (+)
corrected_logits = logits + tau * torch.log(freqs)
corrected_ce = F.cross_entropy(corrected_logits, label_minority).item()

print(f"--- MINORITY CLASS CE LOSS (Contempt, True Label=7) ---")
print(f"Base Logit Output: {logits[0]}")
print(f"Log Freq Adjustment (tau * log(freq)): {tau * torch.log(freqs)}")
print(f"")
print(f"1. Raw CE Loss (Baseline)       : {raw_ce:.4f}")
print(f"2. Buggy Loss (logits - log(p)) : {buggy_ce:.4f}  <-- Loss incorrectly shrank, vanishing gradients!")
print(f"3. Corrected  (logits + log(p)) : {corrected_ce:.4f}  <-- Loss properly SPIKED, forcing network to learn!")
