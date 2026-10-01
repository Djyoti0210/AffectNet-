import time
import pandas as pd

log_path = r"d:\AffectNet+\AffectNet+\validation_logs.csv"
print("Waiting for Stage 3 Epoch 1 (Global Epoch 17) to finish...")

while True:
    try:
        df = pd.read_csv(log_path)
        if 17 in df['epoch'].values:
            print("\n--- STAGE 3 EPOCH 1 METRICS ---")
            ep16 = df[df['epoch'] == 16].iloc[0]
            ep17 = df[df['epoch'] == 17].iloc[0]
            
            print(f"Epoch 16 (Stage 2 Final): Acc {ep16['acc']:.4f} | F1 {ep16['macro_f1']:.4f}")
            print(f"  Contempt: {ep16['class_7_acc']:.3f}, Disgust: {ep16['class_5_acc']:.3f}, Fear: {ep16['class_4_acc']:.3f}, Surprise: {ep16['class_3_acc']:.3f}")
            
            print(f"\nEpoch 17 (Stage 3 First): Acc {ep17['acc']:.4f} | F1 {ep17['macro_f1']:.4f}")
            print(f"  Contempt: {ep17['class_7_acc']:.3f}, Disgust: {ep17['class_5_acc']:.3f}, Fear: {ep17['class_4_acc']:.3f}, Surprise: {ep17['class_3_acc']:.3f}")
            break
    except Exception as e:
        pass
    time.sleep(60)
