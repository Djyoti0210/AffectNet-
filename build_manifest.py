import os
import json
import csv
from glob import glob
from multiprocessing import Pool, cpu_count

base_dir = r"d:\AffectNet+\AffectNet+\human_annotated\train_set"
annotations_dir = os.path.join(base_dir, "annotations")
images_dir = os.path.join(base_dir, "images")
output_csv = os.path.join(base_dir, "master_manifest.csv")

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
        if subset_val is None:
            subset_str = "None"
            
        meta = data.get("meta-data")
        if meta is None:
            meta = {}
            
        age = meta.get("age")
        valence = meta.get("valence")
        arousal = meta.get("arousal")
        
        gender = meta.get("gender")
        if gender is None:
            gender = {}
        gender_female_prob = gender.get("female")
        gender_male_prob = gender.get("male")
        
        race = meta.get("race")
        if race is None:
            race = {}
        race_asian = race.get("asian")
        race_indian = race.get("indian")
        race_black = race.get("black")
        race_white = race.get("white")
        race_mideast = race.get("middle eastern")
        race_hispanic = race.get("latino hispanic")
        
        pose = meta.get("pose")
        if pose is None:
            pose = {}
        yaw = pose.get("yaw")
        pitch = pose.get("pitch")
        roll = pose.get("roll")
        
        landmarks_68 = json.dumps(meta.get("landmark-68", []))
        
        row = [
            image_id,
            image_path,
            human_label,
            soft_label[0] if len(soft_label) > 0 else None,
            soft_label[1] if len(soft_label) > 1 else None,
            soft_label[2] if len(soft_label) > 2 else None,
            soft_label[3] if len(soft_label) > 3 else None,
            soft_label[4] if len(soft_label) > 4 else None,
            soft_label[5] if len(soft_label) > 5 else None,
            soft_label[6] if len(soft_label) > 6 else None,
            soft_label[7] if len(soft_label) > 7 else None,
            subset_str,
            age,
            gender_male_prob,
            gender_female_prob,
            race_asian,
            race_indian,
            race_black,
            race_white,
            race_mideast,
            race_hispanic,
            yaw,
            pitch,
            roll,
            valence,
            arousal,
            landmarks_68
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

    print(f"Processing {len(json_files)} files...")
    with Pool(cpu_count()) as p:
        results = p.map(process_file, json_files)
        
    results = [r for r in results if r is not None]
    
    print(f"Writing {len(results)} rows to {output_csv}...")
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(results)
    
    print("Done!")
