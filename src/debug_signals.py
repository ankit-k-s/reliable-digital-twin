# src/debug_signals.py
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

val_residuals    = np.load('artifacts/val_residuals.npy')
attack_residuals = np.load('artifacts/attack_residuals.npy')
windowed_labels  = np.load('artifacts/windowed_labels.npy')

print("=== Signal Diagnostics ===\n")
print(f"Val residuals   — mean: {val_residuals.mean():.6f} "
      f"std: {val_residuals.std():.6f} "
      f"max: {val_residuals.max():.6f}")
print(f"Attack residuals — mean: {attack_residuals.mean():.6f} "
      f"std: {attack_residuals.std():.6f} "
      f"max: {attack_residuals.max():.6f}")
print(f"Separation ratio: {attack_residuals.mean()/val_residuals.mean():.2f}x\n")

# Test multiple percentile thresholds on RAW residuals
# No normalization — use raw values directly since separation is already 25x
print(f"{'Percentile':>12} | {'Threshold':>10} | {'F1':>8} | "
      f"{'Precision':>10} | {'Recall':>8} | {'FPR':>8}")
print("-" * 70)

best_f1 = 0
best_thresh = 0
best_pct = 0

for pct in [50, 60, 70, 75, 80, 85, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]:
    thresh = np.percentile(val_residuals, pct)
    preds  = (attack_residuals > thresh).astype(int)

    f1   = f1_score(windowed_labels,   preds, zero_division=0)
    prec = precision_score(windowed_labels, preds, zero_division=0)
    rec  = recall_score(windowed_labels,   preds, zero_division=0)
    fpr  = float((preds[windowed_labels == 0]).mean())

    print(f"{pct:>12}% | {thresh:>10.6f} | {f1:>8.4f} | "
          f"{prec:>10.4f} | {rec:>8.4f} | {fpr:>8.4f}")

    if f1 > best_f1:
        best_f1    = f1
        best_thresh = thresh
        best_pct   = pct

print("-" * 70)
print(f"\nBest: {best_pct}th percentile | "
      f"Threshold: {best_thresh:.6f} | F1: {best_f1:.4f}")

# Save the best threshold for evaluate.py
import json
with open('artifacts/training_artifacts.json', 'r+') as f:
    artifacts = json.load(f)
    artifacts['optimal_raw_threshold'] = float(best_thresh)
    artifacts['optimal_percentile']    = best_pct
    f.seek(0)
    json.dump(artifacts, f, indent=2)
    f.truncate()

print(f"\nSaved optimal threshold to training_artifacts.json")