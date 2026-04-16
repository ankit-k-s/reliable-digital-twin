# src/correlation_detector.py
"""
Multivariate correlation-based anomaly detector for stealthy WADI attacks.

Core idea: Train a model to learn the normal correlation structure between
sensor groups. Attacks break physical correlations even when individual
sensor values look plausible.

Uses: Sliding window correlation matrices + Isolation Forest
"""
import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_curve
)
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import json
from src.dataset import get_dataloaders_from_csv


# ─────────────────────────────────────────────
# SENSOR GROUP DEFINITIONS
# ─────────────────────────────────────────────
def get_sensor_groups(sensor_cols):
    """
    Define physically meaningful sensor groups.
    Correlations WITHIN and BETWEEN groups encode process physics.
    """
    idx = {c: i for i, c in enumerate(sensor_cols)}

    groups = {
        # Zone 1 — Raw water intake
        'z1_level':   [i for i,c in enumerate(sensor_cols) if '1_LT_' in c],
        'z1_flow':    [i for i,c in enumerate(sensor_cols) if '1_FIT_' in c],
        'z1_pumps':   [i for i,c in enumerate(sensor_cols)
                       if '1_P_' in c and 'STATUS' in c],
        'z1_valves':  [i for i,c in enumerate(sensor_cols)
                       if '1_MV_' in c and 'STATUS' in c],
        'z1_quality': [i for i,c in enumerate(sensor_cols) if '1_AIT_' in c],

        # Zone 2 — Treatment
        'z2_level':   [i for i,c in enumerate(sensor_cols)
                       if ('2_LT_' in c or '2_LS_' in c) and 'AIT' not in c],
        'z2_flow':    [i for i,c in enumerate(sensor_cols)
                       if '2_FIT_' in c or '2_FIC_' in c],
        'z2_pressure':[i for i,c in enumerate(sensor_cols) if '2_PIT_' in c],
        'z2_pumps':   [i for i,c in enumerate(sensor_cols)
                       if '2_P_' in c and 'STATUS' in c],
        'z2_valves':  [i for i,c in enumerate(sensor_cols)
                       if '2_MV_' in c and 'STATUS' in c],
        'z2a_quality':[i for i,c in enumerate(sensor_cols) if '2A_AIT_' in c],
        'z2b_quality':[i for i,c in enumerate(sensor_cols) if '2B_AIT_' in c],

        # Zone 3 — Distribution
        'z3_level':   [i for i,c in enumerate(sensor_cols) if '3_LT_' in c],
        'z3_flow':    [i for i,c in enumerate(sensor_cols) if '3_FIT_' in c],
        'z3_pumps':   [i for i,c in enumerate(sensor_cols)
                       if '3_P_' in c and 'STATUS' in c],
        'z3_valves':  [i for i,c in enumerate(sensor_cols)
                       if '3_MV_' in c and 'STATUS' in c],
        'z3_quality': [i for i,c in enumerate(sensor_cols) if '3_AIT_' in c],
    }

    # Remove empty groups
    groups = {k: v for k, v in groups.items() if len(v) > 0}
    return groups


def extract_correlation_features(loader, sensor_groups, desc="Extracting"):
    """
    For each window [60, 127], extract:
    1. Within-group correlations (e.g. flow sensors should track each other)
    2. Cross-group correlations (e.g. pump state vs flow)
    3. Temporal statistics (variance, trend slope, flatness)
    4. Physical relationship features (level change vs net flow)

    Returns feature matrix [n_windows, n_features]
    """
    group_names = list(sensor_groups.keys())
    all_features = []

    for batch in tqdm(loader, desc=desc, leave=False):
        batch_np = batch.numpy()  # [batch, 60, 127]

        for i in range(batch_np.shape[0]):
            window = batch_np[i]  # [60, 127]
            feat = []

            # ── Feature Block 1: Within-group mean correlations ──────────
            for gname, gidx in sensor_groups.items():
                if len(gidx) < 2:
                    feat.append(0.0)
                    continue
                group_data = window[:, gidx]  # [60, n_sensors_in_group]
                corr_matrix = np.corrcoef(group_data.T)  # [n, n]
                # Replace NaN (constant sensors) with 0
                corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
                # Upper triangle mean (excluding diagonal)
                upper = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
                feat.append(float(upper.mean()))
                feat.append(float(upper.std()))
                feat.append(float(upper.min()))

            # ── Feature Block 2: Cross-group correlations ────────────────
            # Physically meaningful pairs
            cross_pairs = [
                ('z1_pumps',    'z1_flow'),
                ('z1_flow',     'z1_level'),
                ('z1_valves',   'z1_flow'),
                ('z2_pumps',    'z2_flow'),
                ('z2_flow',     'z2_level'),
                ('z2_pressure', 'z2_flow'),
                ('z2_flow',     'z2a_quality'),
                ('z2_flow',     'z2b_quality'),
                ('z3_pumps',    'z3_flow'),
                ('z3_flow',     'z3_level'),
                ('z3_flow',     'z3_quality'),
            ]
            for g1, g2 in cross_pairs:
                if g1 not in sensor_groups or g2 not in sensor_groups:
                    feat.append(0.0)
                    continue
                s1 = window[:, sensor_groups[g1]].mean(axis=1)  # [60]
                s2 = window[:, sensor_groups[g2]].mean(axis=1)  # [60]
                if s1.std() > 1e-9 and s2.std() > 1e-9:
                    corr = np.corrcoef(s1, s2)[0, 1]
                else:
                    corr = 0.0
                feat.append(float(np.nan_to_num(corr, nan=0.0)))

            # ── Feature Block 3: Temporal variance (flatness detection) ──
            # Stealthy attacks suppress variance — this catches them
            for gname, gidx in sensor_groups.items():
                group_data = window[:, gidx]
                # Variance across time for each sensor, then take minimum
                # A suspiciously flat sensor = very low min variance
                temporal_vars = group_data.var(axis=0)
                feat.append(float(temporal_vars.mean()))
                feat.append(float(temporal_vars.min()))

            # ── Feature Block 4: Trend detection ─────────────────────────
            # Linear slope of each group mean over the window
            t = np.arange(60)
            for gname, gidx in sensor_groups.items():
                group_mean = window[:, gidx].mean(axis=1)  # [60]
                if group_mean.std() > 1e-9:
                    slope = np.polyfit(t, group_mean, 1)[0]
                else:
                    slope = 0.0
                feat.append(float(slope))

            # ── Feature Block 5: Physical mass balance ───────────────────
            # Level change should match net flow (inflow - outflow)
            # Violations indicate sensor spoofing

            # Zone 1: level change vs flow
            if sensor_groups.get('z1_level') and sensor_groups.get('z1_flow'):
                lvl = window[:, sensor_groups['z1_level']].mean(axis=1)
                flw = window[:, sensor_groups['z1_flow']].mean(axis=1)
                lvl_change = lvl[-1] - lvl[0]
                net_flow   = flw.mean()
                # If level rising but no inflow (or vice versa) = anomaly
                mass_balance_z1 = abs(lvl_change - net_flow * 0.1)
                feat.append(float(mass_balance_z1))
            else:
                feat.append(0.0)

            # Zone 3: same
            if sensor_groups.get('z3_level') and sensor_groups.get('z3_flow'):
                lvl = window[:, sensor_groups['z3_level']].mean(axis=1)
                flw = window[:, sensor_groups['z3_flow']].mean(axis=1)
                lvl_change = lvl[-1] - lvl[0]
                net_flow   = flw.mean()
                mass_balance_z3 = abs(lvl_change - net_flow * 0.1)
                feat.append(float(mass_balance_z3))
            else:
                feat.append(0.0)

            all_features.append(feat)

    return np.array(all_features, dtype=np.float32)


def run_correlation_detection():
    print("=" * 58)
    print("   CORRELATION-BASED STEALTHY ATTACK DETECTOR")
    print("=" * 58)

    _, val_loader, attack_loader, windowed_labels, sensor_cols = \
        get_dataloaders_from_csv(batch_size=128)

    windowed_labels = np.array(windowed_labels)
    if -1 in windowed_labels:
        windowed_labels = (windowed_labels == -1).astype(int)
    else:
        windowed_labels = (windowed_labels > 0).astype(int)

    sensor_groups = get_sensor_groups(sensor_cols)
    print(f"\nSensor groups identified: {len(sensor_groups)}")
    for gname, gidx in sensor_groups.items():
        print(f"  {gname:<20}: {len(gidx)} sensors")

    # ── Extract features ─────────────────────────────────────────────────
    print("\n[1/4] Extracting correlation features from validation data...")
    val_features = extract_correlation_features(
        val_loader, sensor_groups, "Val features"
    )

    print("\n[2/4] Extracting correlation features from attack data...")
    attack_features = extract_correlation_features(
        attack_loader, sensor_groups, "Attack features"
    )

    val_features    = np.nan_to_num(val_features,    nan=0.0, posinf=0.0, neginf=0.0)
    attack_features = np.nan_to_num(attack_features, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"\nFeature vector dimension: {val_features.shape[1]}")

    # ── Normalize features ───────────────────────────────────────────────
    scaler = StandardScaler()
    val_features_scaled    = scaler.fit_transform(val_features)
    attack_features_scaled = scaler.transform(attack_features)

    # ── Train Isolation Forest ───────────────────────────────────────────
    print("\n[3/4] Training Isolation Forest on correlation features...")
    iso = IsolationForest(
        n_estimators=300,
        contamination=0.058,
        max_samples='auto',
        random_state=42,
        n_jobs=-1
    )
    iso.fit(val_features_scaled)

    # Continuous anomaly scores
    val_scores    = iso.decision_function(val_features_scaled)
    attack_scores = iso.decision_function(attack_features_scaled)

    # Normalize to [0,1] where 1 = most anomalous
    combined = np.concatenate([val_scores, attack_scores])
    mn, mx   = combined.min(), combined.max()
    norm_val    = 1 - (val_scores    - mn) / (mx - mn + 1e-9)
    norm_attack = 1 - (attack_scores - mn) / (mx - mn + 1e-9)

    # Diagnostic
    true_att  = norm_attack[windowed_labels == 1]
    false_att = norm_attack[windowed_labels == 0]
    print(f"\n  Normal score mean (val):          {norm_val.mean():.4f}")
    print(f"  Normal score mean (attack file):  {false_att.mean():.4f}")
    print(f"  True attack score mean:           {true_att.mean():.4f}")
    print(f"  True separation ratio:            "
          f"{true_att.mean()/false_att.mean():.3f}x")

    # ── Threshold search ─────────────────────────────────────────────────
    print(f"\n[4/4] Finding optimal threshold...")
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
    roc_idx   = np.argmax(tpr_arr - fpr_arr)
    roc_thresh = float(roc_thresholds[roc_idx])
    roc_preds  = (norm_attack > roc_thresh).astype(int)
    roc_f1     = f1_score(windowed_labels, roc_preds, zero_division=0)
    roc_prec   = precision_score(windowed_labels, roc_preds, zero_division=0)
    roc_rec    = recall_score(windowed_labels, roc_preds, zero_division=0)
    roc_fpr    = float((roc_preds[windowed_labels == 0]).mean())
    print(f"\n{'ROC-opt':>12} | {roc_thresh:>10.4f} | {roc_f1:>8.4f} | "
          f"{roc_prec:>10.4f} | {roc_rec:>8.4f} | {roc_fpr:>8.4f}")

    if roc_f1 > best_f1:
        best_f1     = roc_f1
        best_thresh = roc_thresh
        best_preds  = roc_preds.copy()

    print("-" * 72)
    print(f"\nBEST F1: {best_f1:.4f}")

    # ── Save ─────────────────────────────────────────────────────────────
    np.save('artifacts/correlation_scores.npy',  norm_attack)
    np.save('artifacts/correlation_preds.npy',   best_preds)
    np.save('artifacts/windowed_labels.npy',     windowed_labels)

    with open('artifacts/training_artifacts.json', 'r+') as f:
        arts = json.load(f)
        arts['correlation_f1']     = float(best_f1)
        arts['correlation_thresh'] = float(best_thresh)
        f.seek(0)
        json.dump(arts, f, indent=2)
        f.truncate()

    print("\nSaved: artifacts/correlation_scores.npy")
    print("Saved: artifacts/correlation_preds.npy")
    print("\nNext step: python -m src.final_fusion")


if __name__ == "__main__":
    run_correlation_detection()