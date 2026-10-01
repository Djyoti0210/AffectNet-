import os
import pandas as pd
import shutil

df = pd.read_csv(r"d:\AffectNet+\AffectNet+\human_annotated\validation_set\val_manifest.csv")
df_0 = df[df['human_label'] == 0].head(6)
df_1 = df[df['human_label'] == 1].head(6)

os.makedirs("temp_samples", exist_ok=True)

val_dir = r"d:\AffectNet+\AffectNet+\human_annotated\validation_set"
print("Label 0 Images:")
for i, row in df_0.iterrows():
    shutil.copy(os.path.join(val_dir, row['image_path']), f"temp_samples/label0_{i}.png")
    print(f"  Saved label0_{i}.png")

print("\nLabel 1 Images:")
for i, row in df_1.iterrows():
    shutil.copy(os.path.join(val_dir, row['image_path']), f"temp_samples/label1_{i}.png")
    print(f"  Saved label1_{i}.png")
