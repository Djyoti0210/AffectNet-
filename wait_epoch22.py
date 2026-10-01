import time
import pandas as pd

log_path = r"d:\AffectNet+\AffectNet+\validation_logs.csv"
print("Waiting for Epoch 22 to drop...")

while True:
    try:
        df = pd.read_csv(log_path)
        if 22 in df['epoch'].values:
            print("\n--- EPOCH 22 TREND REPORT ---")
            for ep in range(17, 23):
                if ep in df['epoch'].values:
                    row = df[df['epoch'] == ep].iloc[-1]
                    print(f"Epoch {ep:2d} | Acc: {row['acc']*100:.2f}% | F1: {row['macro_f1']:.4f} | C7(Cont): {row['class_7_acc']*100:.1f}% | C5(Disg): {row['class_5_acc']*100:.1f}% | C4(Fear): {row['class_4_acc']*100:.1f}% | C3(Surp): {row['class_3_acc']*100:.1f}%")
            break
    except Exception as e:
        pass
    time.sleep(60)
