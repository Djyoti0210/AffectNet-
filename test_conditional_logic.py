import pandas as pd
import os

base_dir = r"d:\AffectNet+\AffectNet+\human_annotated\train_set"
manifest_path = os.path.join(base_dir, "final_manifest_v1_FROZEN.csv")

df = pd.read_csv(manifest_path, low_memory=False)

# The EXACT logic defined in AffectNetDataset.__getitem__
minority_classes = {3, 4, 5, 7}

def check_logic(human_label, tier):
    is_heavy = (human_label in minority_classes) or (tier in ['soft_weighted', 'high_risk'])
    return 'heavy_transform' if is_heavy else 'std_transform'

tests = [
    ("1. gold-tier, Happy", 1, "gold", "std_transform"),
    ("2. gold-tier, Contempt", 7, "gold", "heavy_transform"),
    ("3. high_risk-tier, Happy", 1, "high_risk", "heavy_transform"),
    ("4. standard-tier, Disgust", 5, "standard", "heavy_transform")
]

print("--- EXACT CONDITIONAL CODE IN DATALOADER ---")
print("is_heavy = (human_label in {3, 4, 5, 7}) or (tier in ['soft_weighted', 'high_risk'])")
print("if is_heavy:")
print("    image_tensor = self.heavy_transform(img)")
print("else:")
print("    image_tensor = self.std_transform(img)")
print("-" * 65)

for desc, label, tier, expected in tests:
    # Validate the scenario literally exists in the dataset
    subset_df = df[(df['human_label'] == label) & (df['training_tier'] == tier)]
    
    if len(subset_df) == 0:
        print(f"{desc}: WARNING - 0 rows exist!")
    else:
        chosen = check_logic(label, tier)
        match_str = "PASS" if chosen == expected else "FAIL"
        print(f"{desc:<30} -> {chosen:<17} | Expected: {expected:<17} | {match_str} (Found {len(subset_df)} real rows)")
