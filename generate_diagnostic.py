import pandas as pd

df = pd.read_csv(r"d:\AffectNet+\AffectNet+\validation_logs.csv")
print("=== EPOCH OVER EPOCH DELTA (MACRO F1) ===")
# Last 5 epochs
last_5 = df.tail(6)
prev_f1 = None
for idx, row in last_5.iterrows():
    f1 = row['macro_f1']
    if prev_f1 is not None:
        delta = f1 - prev_f1
        print(f"Epoch {row['epoch']} F1: {f1:.4f} | Delta: {delta:+.4f}")
    prev_f1 = f1

print("\n=== PER CLASS TREND (EPOCHS 16-19) ===")
recent = df[df['epoch'] >= 16]
for idx, row in recent.iterrows():
    print(f"Epoch {row['epoch']:2d} | C0: {row['class_0_acc']*100:.1f} | C1: {row['class_1_acc']*100:.1f} | C2: {row['class_2_acc']*100:.1f} | C3: {row['class_3_acc']*100:.1f} | C4: {row['class_4_acc']*100:.1f} | C5: {row['class_5_acc']*100:.1f} | C6: {row['class_6_acc']*100:.1f} | C7: {row['class_7_acc']*100:.1f}")

# Read last few lines of task log to get training loss
with open(r"C:\Users\sanch\.gemini\antigravity-ide\brain\6250d43d-bcd0-4ede-8d50-087568f43d53\.system_generated\tasks\task-934.log", "r") as f:
    lines = f.readlines()[-20:]
print("\n=== RECENT TRAINING LOSS ===")
for line in lines:
    if "Loss:" in line:
        print(line.strip())
