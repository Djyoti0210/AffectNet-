import time
import pandas as pd

log_path = r"d:\AffectNet+\AffectNet+\validation_logs.csv"
print("Waiting for Epoch 24 to complete...", flush=True)

while True:
    try:
        df = pd.read_csv(log_path)
        if 24 in df['epoch'].values:
            print("\n--- EPOCH 24 DROPPED ---", flush=True)
            for ep in range(21, 25):
                if ep in df['epoch'].values:
                    row = df[df['epoch'] == ep].iloc[-1]
                    print(f"Epoch {ep:2d} | Acc: {row['acc']*100:.2f}% | F1: {row['macro_f1']:.4f} | C7(Cont): {row['class_7_acc']*100:.1f}% | C5(Disg): {row['class_5_acc']*100:.1f}% | C4(Fear): {row['class_4_acc']*100:.1f}% | C3(Surp): {row['class_3_acc']*100:.1f}%", flush=True)
            break
    except Exception:
        pass
    time.sleep(60)
