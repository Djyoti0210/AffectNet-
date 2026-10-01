import time
import os
import pandas as pd

log_path = r"d:\AffectNet+\AffectNet+\validation_logs.csv"

print("Listening for Epoch 5 and 6 validation drops...")

while True:
    try:
        df = pd.read_csv(log_path)
        real_runs = df.iloc[3:] # Skip the 3 test rows
        if len(real_runs) >= 6:
            print("\n--- EPOCH 5 AND 6 COMPLETE ---")
            for _, row in real_runs.tail(2).iterrows():
                print(f"Epoch {int(row['epoch'])}: Acc {row['acc']:.4f} | F1 {row['macro_f1']:.4f}")
                print(f"  C0:{row['class_0_acc']:.3f}, C1:{row['class_1_acc']:.3f}, C2:{row['class_2_acc']:.3f}, C3:{row['class_3_acc']:.3f}")
                print(f"  C4:{row['class_4_acc']:.3f}, C5:{row['class_5_acc']:.3f}, C6:{row['class_6_acc']:.3f}, C7:{row['class_7_acc']:.3f}")
            break
    except Exception as e:
        pass
    time.sleep(15)
