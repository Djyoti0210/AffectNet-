import os
import json
import csv
from glob import glob
from multiprocessing import Pool, cpu_count
import pandas as pd

base_dir = r"d:\AffectNet+\AffectNet+\human_annotated\validation_set"
annotations_dir = os.path.join(base_dir, "annotations")
images_dir = os.path.join(base_dir, "images")
output_csv = os.path.join(base_dir, "val_manifest.csv")

json_files = glob(os.path.join(annotations_dir, "*.json"))
subset_mapping = {0: "Easy", 1: "Challenging", 2: "Difficult"}

def process_file(f):
    try:
        with open(f, 'r') as file:
            data = json.load(file)
            
        image_id = os.path.basename(f).replace('.json', '')
        image_path = f"images/{image_id}.jpg"
        
        human_label = data.get("human-label")
        soft_label = data.get("soft-label")
        if soft_label is None:
            soft_label = [None] * 8
        
        subset_val = data.get("subset")
        subset_str = subset_mapping.get(subset_val, "Unknown")
        if subset_val is None: subset_str = "None"
            
        meta = data.get("meta-data")
        if meta is None: meta = {}
            
        age = meta.get("age")
        valence = meta.get("valence")
        arousal = meta.get("arousal")
        
        gender = meta.get("gender")
        if gender is None: gender = {}
        
        race = meta.get("race")
        if race is None: race = {}
        
        pose = meta.get("pose")
        if pose is None: pose = {}
        
        landmarks_68 = json.dumps(meta.get("landmark-68", []))
        
        row = [
            image_id, image_path, human_label,
            soft_label[0] if len(soft_label) > 0 else None,
            soft_label[1] if len(soft_label) > 1 else None,
            soft_label[2] if len(soft_label) > 2 else None,
            soft_label[3] if len(soft_label) > 3 else None,
            soft_label[4] if len(soft_label) > 4 else None,
            soft_label[5] if len(soft_label) > 5 else None,
            soft_label[6] if len(soft_label) > 6 else None,
            soft_label[7] if len(soft_label) > 7 else None,
            subset_str, age,
            gender.get("male"), gender.get("female"),
            race.get("asian"), race.get("indian"), race.get("black"),
            race.get("white"), race.get("middle eastern"), race.get("latino hispanic"),
            pose.get("yaw"), pose.get("pitch"), pose.get("roll"),
            valence, arousal, landmarks_68
        ]
        return row
    except Exception as e:
        print(f"Error processing {f}: {e}")
        return None

if __name__ == '__main__':
    headers = [
        "image_id", "image_path", "human_label",
        "soft_label_0", "soft_label_1", "soft_label_2", "soft_label_3", 
        "soft_label_4", "soft_label_5", "soft_label_6", "soft_label_7",
        "subset", "age", "gender_male_prob", "gender_female_prob",
        "race_asian", "race_indian", "race_black", "race_white", 
        "race_mideast", "race_hispanic", "yaw", "pitch", "roll", 
        "valence", "arousal", "landmarks_68"
    ]

    print(f"Processing {len(json_files)} validation files...")
    with Pool(cpu_count()) as p:
        results = p.map(process_file, json_files)
        
    results = [r for r in results if r is not None]
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(results)
    print("Parsing complete.\n")

    print("--- VALIDATION CHECKS ---")
    df = pd.read_csv(output_csv, low_memory=False)
    
    # 1. Total rows before & after filter
    total_raw = len(df)
    print(f"1. Total rows before filter: {total_raw} (Expected: 5500)")
    
    df_filtered = df[df['human_label'].isin([0,1,2,3,4,5,6,7])].copy()
    total_filtered = len(df_filtered)
    print(f"1b. Total rows after filter: {total_filtered} (Expected: 4000)")
    
    # Overwrite with only the filtered 0-7 classes
    df_filtered.to_csv(output_csv, index=False)
    
    # 2. Exact class distribution
    counts = df_filtered['human_label'].value_counts().to_dict()
    print("\n2. Class distribution:")
    for c in range(8):
        print(f"   Class {c}: {counts.get(c, 0)}")
        
    # 3. Image-JSON Parity
    img_files = glob(os.path.join(images_dir, "*.jpg"))
    print("\n3. Parity Check:")
    print(f"   Total .json files: {len(json_files)}")
    print(f"   Total .jpg files:  {len(img_files)}")
    img_bases = set([os.path.basename(f).replace('.jpg', '') for f in img_files])
    json_bases = set([os.path.basename(f).replace('.json', '') for f in json_files])
    print(f"   Orphan images: {len(img_bases - json_bases)}")
    print(f"   Orphan JSONs : {len(json_bases - img_bases)}")
    
    # 4. Data Leakage (Train/Val overlap)
    train_manifest = r"d:\AffectNet+\AffectNet+\human_annotated\train_set\final_manifest_v1_FROZEN.csv"
    train_df = pd.read_csv(train_manifest, usecols=['image_id'], low_memory=False)
    train_ids = set(train_df['image_id'].astype(str).tolist())
    val_ids = set(df['image_id'].astype(str).tolist())
    overlap = train_ids.intersection(val_ids)
    print(f"\n4. Data Leakage Check (Train/Val image_id overlap): {len(overlap)} images")
    
    # 5. Columns check
    print(f"\n5. Columns saved: {list(df_filtered.columns)[:3]}... ({len(df_filtered.columns)} total, no tiers/weights computed)")
    print("Done!")
