import time
import pandas as pd

log_path = r"d:\AffectNet+\AffectNet+\validation_logs.csv"
print("Waiting for Epoch 17 (Baseline Re-run) to finish...")

while True:
    try:
        df = pd.read_csv(log_path)
        if 17 in df['epoch'].values:
            print("\n--- EPOCH 17 DROPPED ---")
            ep16 = df[df['epoch'] == 16].iloc[-1]
            ep17 = df[df['epoch'] == 17].iloc[-1]
            
            report = f"""=========================================================
STAGE 3 EPOCH 17 BASELINE REPORT (RE-RUN)
=========================================================

--- PERFORMANCE COMPARISON ---

[Epoch 16 - Stage 2 Final (Raw CE Loss)]
Overall Acc : {ep16['acc']*100:.2f}%
Macro F1    : {ep16['macro_f1']:.4f}
Contempt    : {ep16['class_7_acc']*100:.2f}%
Disgust     : {ep16['class_5_acc']*100:.2f}%
Fear        : {ep16['class_4_acc']*100:.2f}%
Surprise    : {ep16['class_3_acc']*100:.2f}%

[Epoch 17 - Stage 3 First (Corrected Logit-Adjusted CE Loss)]
Overall Acc : {ep17['acc']*100:.2f}%
Macro F1    : {ep17['macro_f1']:.4f}
Contempt    : {ep17['class_7_acc']*100:.2f}%
Disgust     : {ep17['class_5_acc']*100:.2f}%
Fear        : {ep17['class_4_acc']*100:.2f}%
Surprise    : {ep17['class_3_acc']*100:.2f}%
"""
            with open(r"d:\AffectNet+\AffectNet+\STAGE_3_BASELINE_REPORT.txt", "w") as f:
                f.write(report)
            
            print("Report saved to STAGE_3_BASELINE_REPORT.txt")
            break
    except Exception as e:
        pass
    time.sleep(60)
