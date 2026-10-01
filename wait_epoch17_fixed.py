import time
import pandas as pd
import subprocess
import shutil
import os

log_path = r"d:\AffectNet+\AffectNet+\validation_logs.csv"
print("Waiting for Epoch 17 (Sampler ON) to finish...")

while True:
    try:
        df = pd.read_csv(log_path)
        if len(df[df['epoch'] == 17]) >= 1:
            print("\n--- FIRST EPOCH 17 (Sampler ON) DROPPED ---")
            break
    except Exception as e:
        pass
    time.sleep(60)

df = pd.read_csv(log_path)
ep16 = df[df['epoch'] == 16].iloc[-1]
ep17_on = df[df['epoch'] == 17].iloc[-1]

checkpoint_16 = r"d:\AffectNet+\AffectNet+\checkpoints\epoch_16.pt"
latest_ckpt = r"d:\AffectNet+\AffectNet+\checkpoints\latest.pt"

print("Copying epoch_16.pt to latest.pt for Sampler OFF run...")
shutil.copy(checkpoint_16, latest_ckpt)

print("Starting Sampler OFF run...")
proc = subprocess.Popen(["python", "train.py", "--resume", "--stop_after_epoch", "17", "--disable_sampler"], cwd=r"d:\AffectNet+\AffectNet+")

print("Waiting for SECOND Epoch 17 (Sampler OFF) to finish...")
while True:
    try:
        df = pd.read_csv(log_path)
        if len(df[df['epoch'] == 17]) >= 2:
            print("\n--- SECOND EPOCH 17 (Sampler OFF) DROPPED ---")
            break
    except Exception as e:
        pass
    time.sleep(60)

df = pd.read_csv(log_path)
ep17_off = df[df['epoch'] == 17].iloc[-1]

report = f"""=========================================================
STAGE 3 EPOCH 17 HYPOTHESIS TESTING REPORT
=========================================================

--- PERFORMANCE COMPARISON ---

[Epoch 16 - Stage 2 Final (Raw CE Loss + Sampler ON)]
Overall Acc : {ep16['acc']*100:.2f}%
Macro F1    : {ep16['macro_f1']:.4f}
Contempt    : {ep16['class_7_acc']*100:.2f}%
Disgust     : {ep16['class_5_acc']*100:.2f}%
Fear        : {ep16['class_4_acc']*100:.2f}%
Surprise    : {ep16['class_3_acc']*100:.2f}%

[Epoch 17 - Stage 3 (LA-loss Fixed + Sampler ON)]
Overall Acc : {ep17_on['acc']*100:.2f}%
Macro F1    : {ep17_on['macro_f1']:.4f}
Contempt    : {ep17_on['class_7_acc']*100:.2f}%
Disgust     : {ep17_on['class_5_acc']*100:.2f}%
Fear        : {ep17_on['class_4_acc']*100:.2f}%
Surprise    : {ep17_on['class_3_acc']*100:.2f}%

[Epoch 17 - Stage 3 (LA-loss Fixed + Sampler OFF)]
Overall Acc : {ep17_off['acc']*100:.2f}%
Macro F1    : {ep17_off['macro_f1']:.4f}
Contempt    : {ep17_off['class_7_acc']*100:.2f}%
Disgust     : {ep17_off['class_5_acc']*100:.2f}%
Fear        : {ep17_off['class_4_acc']*100:.2f}%
Surprise    : {ep17_off['class_3_acc']*100:.2f}%
"""
with open(r"d:\AffectNet+\AffectNet+\STAGE_3_FINAL_REPORT.txt", "w") as f:
    f.write(report)
print("All done!")
