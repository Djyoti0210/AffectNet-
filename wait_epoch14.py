import time
import pandas as pd

log_path = r"d:\AffectNet+\AffectNet+\validation_logs.csv"
print("Waiting for Epoch 14 to print to validation_logs.csv...")

while True:
    try:
        df = pd.read_csv(log_path)
        if 14 in df['epoch'].values:
            print("\n--- EPOCH 14 DROPPED ---")
            ep13 = df[df['epoch'] == 13].iloc[0]
            ep14 = df[df['epoch'] == 14].iloc[0]
            
            print(f"Epoch 13: Acc {ep13['acc']:.4f} | F1 {ep13['macro_f1']:.4f}")
            print(f"Epoch 14: Acc {ep14['acc']:.4f} | F1 {ep14['macro_f1']:.4f}")
            
            if abs(ep14['acc'] - ep13['acc']) < 0.05:
                print("Continuity Verified: Resume is stable.")
            else:
                print("WARNING: Discontinuous jump detected!")
            break
    except Exception as e:
        pass
    time.sleep(30)
