import torch
import numpy as np

freq_dict = {
    0: 134415,
    1: 74874,
    2: 25459,
    3: 14090,
    4: 6378,
    5: 3803,
    6: 24882,
    7: 3752
}
total = sum(freq_dict.values())
freqs = {k: v/total for k, v in freq_dict.items()}

print("--- CLASS FREQUENCIES USED ---")
for k, v in freqs.items():
    print(f"Class {k}: {v:.6f} (Count: {freq_dict[k]})")

print("\n--- MANUAL CALCULATION FOR TAU = -0.20 ---")
# Example logit vector for a Contempt image
raw_logits = np.array([1.2, 0.5, -0.3, -1.5, -0.8, -0.2, 0.1, 0.7])
print(f"Assume raw logits from model: {raw_logits}")

tau = -0.20
adjusted_logits = np.zeros_like(raw_logits)
for i in range(8):
    log_freq = np.log(freqs[i])
    penalty = tau * log_freq
    adjusted_logits[i] = raw_logits[i] + penalty
    print(f"Class {i} | raw: {raw_logits[i]:>5.2f} | log(p): {log_freq:>6.3f} | {tau}*log(p) = {penalty:>6.3f} | adj: {adjusted_logits[i]:>6.3f}")

print(f"Argmax Raw: Class {np.argmax(raw_logits)}")
print(f"Argmax Adj: Class {np.argmax(adjusted_logits)}")
