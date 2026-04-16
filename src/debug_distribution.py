# src/debug_distribution.py
import numpy as np
import matplotlib.pyplot as plt

val_residuals    = np.load('artifacts/val_residuals.npy')
attack_residuals = np.load('artifacts/attack_residuals.npy')
windowed_labels  = np.load('artifacts/windowed_labels.npy')

# Split attack residuals into actual attack vs normal windows
true_attack_residuals  = attack_residuals[windowed_labels == 1]
false_attack_residuals = attack_residuals[windowed_labels == 0]

print("=== Distribution Analysis ===\n")
print(f"Val (normal) residuals:")
print(f"  mean: {val_residuals.mean():.4f} | "
      f"std: {val_residuals.std():.4f} | "
      f"p95: {np.percentile(val_residuals, 95):.4f} | "
      f"max: {val_residuals.max():.4f}")

print(f"\nAttack-file NORMAL windows (label=0):")
print(f"  count: {len(false_attack_residuals):,}")
print(f"  mean: {false_attack_residuals.mean():.4f} | "
      f"std: {false_attack_residuals.std():.4f} | "
      f"p95: {np.percentile(false_attack_residuals, 95):.4f} | "
      f"max: {false_attack_residuals.max():.4f}")

print(f"\nAttack-file ATTACK windows (label=1):")
print(f"  count: {len(true_attack_residuals):,}")
print(f"  mean: {true_attack_residuals.mean():.4f} | "
      f"std: {true_attack_residuals.std():.4f} | "
      f"p95: {np.percentile(true_attack_residuals, 95):.4f} | "
      f"max: {true_attack_residuals.max():.4f}")

# True separation ratio
print(f"\nTrue separation ratio "
      f"(attack mean / normal-in-attack-file mean): "
      f"{true_attack_residuals.mean()/false_attack_residuals.mean():.2f}x")

# What % of true attack windows exceed various thresholds
print(f"\n--- How detectable are TRUE attack windows? ---")
for pct in [90, 95, 99]:
    thresh = np.percentile(val_residuals, pct)
    detectable = (true_attack_residuals > thresh).mean() * 100
    fp_rate    = (false_attack_residuals > thresh).mean() * 100
    print(f"  Val {pct}th pct threshold ({thresh:.4f}): "
          f"catches {detectable:.1f}% of attacks, "
          f"{fp_rate:.1f}% FP on normal-in-attack-file")

# Save plot
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.hist(val_residuals, bins=100, alpha=0.7, label='Val normal', color='blue')
plt.title('Validation residuals')
plt.xlabel('Residual value')

plt.subplot(1, 3, 2)
plt.hist(false_attack_residuals, bins=100, alpha=0.7,
         label='Normal in attack file', color='green')
plt.title('Normal windows in attack file')
plt.xlabel('Residual value')

plt.subplot(1, 3, 3)
plt.hist(true_attack_residuals, bins=100, alpha=0.7,
         label='True attack windows', color='red')
plt.title('True attack windows')
plt.xlabel('Residual value')

plt.tight_layout()
plt.savefig('artifacts/residual_distributions.png', dpi=150)
print(f"\nPlot saved to artifacts/residual_distributions.png")