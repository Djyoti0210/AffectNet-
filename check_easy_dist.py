import pandas as pd

manifest_path = r"d:\AffectNet+\AffectNet+\human_annotated\validation_set\val_manifest.csv"
df = pd.read_csv(manifest_path)

easy_df = df[df['subset'] == 'Easy']
counts = easy_df['human_label'].value_counts().sort_index()
total = len(easy_df)

print(f"Total Easy Images: {total}")
class_names = ["Happy", "Neutral", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]

for idx in counts.index:
    print(f"{class_names[idx]} (Class {idx}): {counts[idx]} ({counts[idx]/total*100:.2f}%)")

happy_neutral_pct = (counts.get(0, 0) + counts.get(1, 0)) / total * 100
print(f"\nHappy+Neutral Combined: {happy_neutral_pct:.2f}%")
