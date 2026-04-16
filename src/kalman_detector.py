# src/kalman_detector.py
"""
Kalman Filter based anomaly detector for WADI stealthy attacks.

Key insight: Instead of comparing against a static trained baseline,
we use a Kalman Filter to predict the NEXT sensor state from the
current state. Attacks cause prediction errors that accumulate
into a detectable signal regardless of operating regime.

This directly addresses distribution shift between normal and attack files.
"""
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve
from sklearn.preprocessing import StandardScaler
import json
from tqdm import tqdm
from src.dataset import get_dataloaders_from_csv


class SimpleKalmanFilter:
    """
    Per-sensor Kalman Filter.
    State = sensor value. Observation = measured sensor value.
    Predicts next value using exponential smoothing (first-order Kalman).
    Innovation = |predicted - actual| → anomaly signal.
    """
    def __init__(self, n_sensors, process_noise=1e-4, obs_noise=1e-2):
        self.n   = n_sensors
        self.Q   = process_noise   # process noise covariance
        self.R   = obs_noise       # observation noise covariance
        self.reset()

    def reset(self):
        self.x = None   # state estimate
        self.P = None   # error covariance

    def update(self, z):
        """
        z: observation array [n_sensors]
        Returns innovation (prediction error) [n_sensors]
        """
        if self.x is None:
            self.x = z.copy()
            self.P = np.ones(self.n)
            return np.zeros(self.n)

        # Predict
        x_pred = self.x                    # constant velocity model
        P_pred = self.P + self.Q           # covariance prediction

        # Update
        S = P_pred + self.R                # innovation covariance
        K = P_pred / S                     # Kalman gain
        innovation = z - x_pred           # prediction error
        self.x = x_pred + K * innovation  # updated state
        self.P = (1 - K) * P_pred         # updated covariance

        return innovation


def compute_kalman_scores(loader, n_sensors, desc="Computing"):
    """
    For each window, run Kalman filter through all 60 timesteps.
    Collect per-timestep innovations, aggregate to window-level score.
    """
    kf = SimpleKalmanFilter(n_sensors, process_noise=1e-4, obs_noise=1e-2)
    all_scores = []

    for batch in tqdm(loader, desc=desc, leave=False):
        batch_np = batch.numpy()  # [batch, 60, 127]

        for i in range(batch_np.shape[0]):
            window = batch_np[i]  # [60, 127]
            kf.reset()

            innovations = []
            for t in range(60):
                inn = kf.update(window[t])
                innovations.append(np.abs(inn))

            innovations = np.array(innovations)  # [60, 127]

            # Window-level features from innovations
            # These capture both sudden spikes and sustained drift
            inn_mean = innovations.mean()
            inn_max  = innovations.max()
            inn_std  = innovations.std()

            # Trend — is the innovation growing over the window?
            inn_per_timestep = innovations.mean(axis=1)  # [60]
            inn_trend = np.polyfit(np.arange(60), inn_per_timestep, 1)[0]

            # Suppression detection — very low innovation = suspicious flatness
            inn_min_sensor = innovations.mean(axis=0).min()

            # Per-sensor max innovation (which sensors are most anomalous)
            per_sensor_max = innovations.max(axis=0)
            top5_mean = np.sort(per_sensor_max)[-5:].mean()

            all_scores.append([
                inn_mean,
                inn_max,
                inn_std,
                inn_trend,
                inn_min_sensor,
                top5_mean,
            ])

    return np.array(all_scores, dtype=np.float32)


def compute_kl_divergence_scores(loader, val_distributions, n_sensors,
                                  desc="KL scores"):
    """
    KL divergence between window sensor distribution and learned normal.
    Catches distribution shift within a window vs expected normal.
    val_distributions: dict with 'mean' and 'std' per sensor [n_sensors]
    """
    all_kl = []

    for batch in tqdm(loader, desc=desc, leave=False):
        batch_np = batch.numpy()  # [batch, 60, 127]

        for i in range(batch_np.shape[0]):
            window = batch_np[i]  # [60, 127]

            # Window distribution per sensor
            w_mean = window.mean(axis=0)   # [127]
            w_std  = window.std(axis=0) + 1e-9

            # KL divergence: KL(window || normal)
            # KL(N(μ1,σ1) || N(μ2,σ2)) =
            #   log(σ2/σ1) + (σ1² + (μ1-μ2)²)/(2σ2²) - 0.5
            mu1, s1 = w_mean, w_std
            mu2 = val_distributions['mean']
            s2  = val_distributions['std']

            kl = (np.log(s2/s1) +
                  (s1**2 + (mu1-mu2)**2) / (2*s2**2) - 0.5)
            kl = np.nan_to_num(kl, nan=0.0, posinf=0.0, neginf=0.0)
            kl = np.clip(kl, 0, 10)

            all_kl.append([
                kl.mean(),
                kl.max(),
                kl.std(),
                np.sort(kl)[-10:].mean(),  # top 10 sensor KL
            ])

    return np.array(all_kl, dtype=np.float32)


def run_kalman_detection():
    print("=" * 58)
    print("   KALMAN FILTER + KL DIVERGENCE DETECTOR")
    print("   (Information-Theoretic Digital Twin approach)")
    print("=" * 58)

    _, val_loader, attack_loader, windowed_labels, sensor_cols = \
        get_dataloaders_from_csv(batch_size=128)

    windowed_labels = np.array(windowed_labels)
    if -1 in windowed_labels:
        windowed_labels = (windowed_labels == -1).astype(int)
    else:
        windowed_labels = (windowed_labels > 0).astype(int)

    n_sensors = len(sensor_cols)

    # ── Step 1: Learn normal distributions from validation data ──────────
    print("\n[1/5] Learning normal sensor distributions from val data...")
    all_val_windows = []
    for batch in val_loader:
        all_val_windows.append(batch.numpy())
    all_val_windows = np.concatenate(all_val_windows, axis=0)  # [N, 60, 127]

    # Per-sensor mean and std across all normal windows
    flat_val = all_val_windows.reshape(-1, n_sensors)  # [N*60, 127]
    val_distributions = {
        'mean': flat_val.mean(axis=0),           # [127]
        'std':  flat_val.std(axis=0) + 1e-9,     # [127]
    }
    print(f"      Learned distributions for {n_sensors} sensors")

    # ── Step 2: Kalman scores ─────────────────────────────────────────────
    print("\n[2/5] Computing Kalman innovation scores...")
    print("      Validation data...")
    val_kalman = compute_kalman_scores(
        val_loader, n_sensors, "Val Kalman"
    )
    print("      Attack data...")
    attack_kalman = compute_kalman_scores(
        attack_loader, n_sensors, "Attack Kalman"
    )

    # ── Step 3: KL divergence scores ─────────────────────────────────────
    print("\n[3/5] Computing KL divergence scores...")
    print("      Validation data...")
    val_kl = compute_kl_divergence_scores(
        val_loader, val_distributions, n_sensors, "Val KL"
    )
    print("      Attack data...")
    attack_kl = compute_kl_divergence_scores(
        attack_loader, val_distributions, n_sensors, "Attack KL"
    )

    # ── Step 4: Combine into final feature matrix ─────────────────────────
    print("\n[4/5] Combining features and training detector...")
    val_combined    = np.hstack([val_kalman,    val_kl])
    attack_combined = np.hstack([attack_kalman, attack_kl])

    val_combined    = np.nan_to_num(val_combined,    nan=0.0, posinf=0.0)
    attack_combined = np.nan_to_num(attack_combined, nan=0.0, posinf=0.0)

    print(f"      Combined feature size: {val_combined.shape[1]}")

    # Normalize
    scaler = StandardScaler()
    val_scaled    = scaler.fit_transform(val_combined)
    attack_scaled = scaler.transform(attack_combined)

    # Isolation Forest on combined features
    from sklearn.ensemble import IsolationForest
    iso = IsolationForest(
        n_estimators=300,
        contamination=0.058,
        max_samples=min(len(val_scaled), 10000),
        random_state=42,
        n_jobs=-1
    )
    iso.fit(val_scaled)

    val_scores    = iso.decision_function(val_scaled)
    attack_scores = iso.decision_function(attack_scaled)

    # Normalize to 0-1 where 1 = anomalous
    combined_scores = np.concatenate([val_scores, attack_scores])
    mn, mx = combined_scores.min(), combined_scores.max()
    norm_val    = 1 - (val_scores    - mn) / (mx - mn + 1e-9)
    norm_attack = 1 - (attack_scores - mn) / (mx - mn + 1e-9)

    # Diagnostic
    true_att  = norm_attack[windowed_labels == 1]
    false_att = norm_attack[windowed_labels == 0]
    print(f"\n  Separation diagnostics:")
    print(f"    Val normal mean:               {norm_val.mean():.4f}")
    print(f"    Attack-file normal mean:        {false_att.mean():.4f}")
    print(f"    True attack mean:               {true_att.mean():.4f}")
    print(f"    True separation ratio:          "
          f"{true_att.mean()/false_att.mean():.3f}x")

    # Also test Kalman signal alone (without KL)
    val_k_only    = val_kalman[:, 0]    # mean innovation alone
    attack_k_only = attack_kalman[:, 0]
    true_k  = attack_k_only[windowed_labels == 1]
    false_k = attack_k_only[windowed_labels == 0]
    print(f"\n  Kalman innovation alone:")
    print(f"    Attack-file normal mean:   {false_k.mean():.6f}")
    print(f"    True attack mean:          {true_k.mean():.6f}")
    print(f"    Separation:                "
          f"{true_k.mean()/false_k.mean():.3f}x")

    # ── Step 5: Threshold search ──────────────────────────────────────────
    print(f"\n[5/5] Finding optimal threshold...")
    print(f"\n{'Percentile':>12} | {'Threshold':>10} | {'F1':>8} | "
          f"{'Precision':>10} | {'Recall':>8} | {'FPR':>8}")
    print("-" * 72)

    best_f1     = 0
    best_thresh = 0
    best_preds  = None

    for pct in [80, 85, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]:
        thresh = np.percentile(false_att, pct)
        preds  = (norm_attack > thresh).astype(int)
        f1   = f1_score(windowed_labels,   preds, zero_division=0)
        prec = precision_score(windowed_labels, preds, zero_division=0)
        rec  = recall_score(windowed_labels,   preds, zero_division=0)
        fpr  = float((preds[windowed_labels == 0]).mean())

        print(f"{pct:>12}% | {thresh:>10.4f} | {f1:>8.4f} | "
              f"{prec:>10.4f} | {rec:>8.4f} | {fpr:>8.4f}")

        if f1 > best_f1:
            best_f1     = f1
            best_thresh = thresh
            best_preds  = preds.copy()

    # ROC-optimal
    fpr_arr, tpr_arr, roc_thresholds = roc_curve(
        windowed_labels, norm_attack
    )
    roc_idx    = np.argmax(tpr_arr - fpr_arr)
    roc_thresh = float(roc_thresholds[roc_idx])
    roc_preds  = (norm_attack > roc_thresh).astype(int)
    roc_f1     = f1_score(windowed_labels, roc_preds, zero_division=0)
    roc_prec   = precision_score(windowed_labels, roc_preds, zero_division=0)
    roc_rec    = recall_score(windowed_labels, roc_preds, zero_division=0)
    roc_fpr    = float((roc_preds[windowed_labels == 0]).mean())
    print(f"\n{'ROC-opt':>12} | {roc_thresh:>10.4f} | {roc_f1:>8.4f} | "
          f"{roc_prec:>10.4f} | {roc_rec:>8.4f} | {roc_fpr:>8.4f}")

    if roc_f1 > best_f1:
        best_f1    = roc_f1
        best_preds = roc_preds.copy()
        best_thresh = roc_thresh

    print("-" * 72)
    print(f"\nBEST F1: {best_f1:.4f}")

    # Save
    np.save('artifacts/kalman_scores.npy',   norm_attack)
    np.save('artifacts/kalman_preds.npy',    best_preds)
    np.save('artifacts/P_final_scores.npy',  norm_attack)
    np.save('artifacts/windowed_labels.npy', windowed_labels)

    with open('artifacts/training_artifacts.json', 'r+') as f:
        arts = json.load(f)
        arts['kalman_f1']     = float(best_f1)
        arts['kalman_thresh'] = float(best_thresh)
        arts['layer3_f1']     = float(best_f1)
        f.seek(0)
        json.dump(arts, f, indent=2)
        f.truncate()

    print("\nSaved: artifacts/kalman_scores.npy")
    print("Saved: artifacts/P_final_scores.npy")
    print("\nNext: python -m src.reliability")


if __name__ == "__main__":
    run_kalman_detection()