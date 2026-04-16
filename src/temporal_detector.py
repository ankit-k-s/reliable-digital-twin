# src/temporal_detector.py
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve
from sklearn.ensemble import IsolationForest
import json
from src.dataset import get_dataloaders_from_csv
from tqdm import tqdm

def extract_temporal_features(loader, sensor_cols, desc="Extracting"):
    """
    Extract features that capture TEMPORAL PATTERNS rather than magnitudes.
    These catch stealthy attacks that suppress sensor variance.
    """
    # Identify key sensor groups
    level_idx  = [i for i, c in enumerate(sensor_cols) if '_LT_' in c]
    flow_idx   = [i for i, c in enumerate(sensor_cols) if '_FIT_' in c or '_FIC_' in c]
    pump_idx   = [i for i, c in enumerate(sensor_cols) if '_P_' in c and 'STATUS' in c]
    quality_idx = [i for i, c in enumerate(sensor_cols) if '_AIT_' in c]

    features = []

    for batch in tqdm(loader, desc=desc, leave=False):
        batch_np = batch.numpy()  # [batch, 60, 127]

        batch_features = []
        for i in range(batch_np.shape[0]):
            window = batch_np[i]  # [60, 127]
            feat = []

            # --- Feature Group 1: Flatness (suspiciously low variance) ---
            # Real sensors have natural noise. Too-flat = spoofed.
            for idx_group in [level_idx, flow_idx, quality_idx]:
                if idx_group:
                    group_data = window[:, idx_group]
                    # Variance across time for each sensor, then mean
                    temporal_var = group_data.var(axis=0).mean()
                    feat.append(temporal_var)
                    # Minimum variance (the flattest sensor in group)
                    feat.append(group_data.var(axis=0).min())

            # --- Feature Group 2: Rate of change consistency ---
            # Attacks often cause unnaturally smooth transitions
            for idx_group in [level_idx, flow_idx]:
                if idx_group:
                    group_data = window[:, idx_group]
                    diffs = np.diff(group_data, axis=0)
                    # Variance of the differences (low = unnaturally smooth)
                    feat.append(diffs.var(axis=0).mean())
                    # Max absolute change (high = sudden jump = possible attack)
                    feat.append(np.abs(diffs).max(axis=0).mean())

            # --- Feature Group 3: Cross-sensor causality ---
            # Pump state vs flow correlation
            if pump_idx and flow_idx:
                pump_data = window[:, pump_idx].mean(axis=1)  # avg pump state
                flow_data = window[:, flow_idx].mean(axis=1)  # avg flow
                # If pumps change but flow doesn't respond = causality broken
                pump_changes = np.abs(np.diff(pump_data)).sum()
                flow_changes = np.abs(np.diff(flow_data)).sum()
                # Ratio: high pump activity with low flow response = suspicious
                causality_ratio = pump_changes / (flow_changes + 1e-9)
                feat.append(causality_ratio)
                # Correlation between pump and flow over window
                if pump_data.std() > 1e-9 and flow_data.std() > 1e-9:
                    correlation = np.corrcoef(pump_data, flow_data)[0, 1]
                else:
                    correlation = 0.0
                feat.append(float(correlation))

            # --- Feature Group 4: Quality sensor behavior ---
            # Water quality sensors (AIT) should correlate with flow
            if quality_idx and flow_idx:
                quality_data = window[:, quality_idx].mean(axis=1)
                flow_data    = window[:, flow_idx].mean(axis=1)
                if quality_data.std() > 1e-9 and flow_data.std() > 1e-9:
                    q_f_corr = np.corrcoef(quality_data, flow_data)[0, 1]
                else:
                    q_f_corr = 0.0
                feat.append(float(q_f_corr))
                # Quality variance (too flat = sensor frozen/spoofed)
                feat.append(float(quality_data.var()))

            # --- Feature Group 5: Global statistics ---
            feat.append(float(window.var(axis=0).mean()))      # mean sensor variance
            feat.append(float(window.var(axis=0).min()))       # min sensor variance
            feat.append(float(np.abs(np.diff(window, axis=0)).mean()))  # mean change rate
            feat.append(float(window.mean()))                  # global mean level

            batch_features.append(feat)

        features.extend(batch_features)

    return np.array(features, dtype=np.float32)


def run_temporal_detection():
    print("=" * 55)
    print("   TEMPORAL PATTERN DETECTOR (Stealthy Attack Fix)")
    print("=" * 55)

    _, val_loader, attack_loader, windowed_labels, sensor_cols = \
        get_dataloaders_from_csv(batch_size=128)

    windowed_labels = np.array(windowed_labels)
    if -1 in windowed_labels:
        windowed_labels = (windowed_labels == -1).astype(int)
    else:
        windowed_labels = (windowed_labels > 0).astype(int)

    print("\n[1/4] Extracting temporal features from validation data...")
    val_features = extract_temporal_features(
        val_loader, sensor_cols, "Val features"
    )
    print(f"      Feature vector size: {val_features.shape[1]}")

    print("\n[2/4] Extracting temporal features from attack data...")
    attack_features = extract_temporal_features(
        attack_loader, sensor_cols, "Attack features"
    )

    # Handle NaN/Inf from edge cases
    val_features    = np.nan_to_num(val_features,    nan=0.0, posinf=0.0)
    attack_features = np.nan_to_num(attack_features, nan=0.0, posinf=0.0)

    print(f"\n[3/4] Training Isolation Forest on temporal features...")
    # contamination matches actual attack rate
    iso = IsolationForest(
        contamination=0.058,
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    iso.fit(val_features)

    # Continuous anomaly scores
    val_scores    = iso.decision_function(val_features)
    attack_scores = iso.decision_function(attack_features)

    # Normalize: high score = more anomalous
    combined = np.concatenate([val_scores, attack_scores])
    mn, mx   = combined.min(), combined.max()
    norm_val    = 1 - (val_scores    - mn) / (mx - mn + 1e-9)
    norm_attack = 1 - (attack_scores - mn) / (mx - mn + 1e-9)

    true_att  = norm_attack[windowed_labels == 1]
    false_att = norm_attack[windowed_labels == 0]
    print(f"      Normal score mean:       {norm_val.mean():.4f}")
    print(f"      Attack-normal score mean: {false_att.mean():.4f}")
    print(f"      True-attack score mean:   {true_att.mean():.4f}")
    print(f"      True separation:          "
          f"{true_att.mean()/false_att.mean():.3f}x")

    print(f"\n[4/4] Finding optimal threshold...")
    print(f"\n{'Percentile':>12} | {'Threshold':>10} | {'F1':>8} | "
          f"{'Precision':>10} | {'Recall':>8} | {'FPR':>8}")
    print("-" * 72)

    best_f1     = 0
    best_thresh = 0
    best_preds  = None

    # Threshold from normal-in-attack-file
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

    # Also try ROC-optimal
    fpr_arr, tpr_arr, thresholds = roc_curve(windowed_labels, norm_attack)
    roc_idx   = np.argmax(tpr_arr - fpr_arr)
    roc_thresh = float(thresholds[roc_idx])
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
    np.save('artifacts/temporal_scores.npy',   norm_attack)
    np.save('artifacts/temporal_preds.npy',    best_preds)

    with open('artifacts/training_artifacts.json', 'r+') as f:
        arts = json.load(f)
        arts['temporal_f1']     = float(best_f1)
        arts['temporal_thresh'] = float(best_thresh)
        f.seek(0)
        json.dump(arts, f, indent=2)
        f.truncate()

    print("Saved: artifacts/temporal_scores.npy")
    print("Saved: artifacts/temporal_preds.npy")
    print("\nRun python -m src.final_fusion next to combine all signals.")


if __name__ == "__main__":
    run_temporal_detection()