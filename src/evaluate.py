# src/evaluate.py
import numpy as np
import os
import json
import torch
from tqdm import tqdm
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    roc_curve, precision_recall_curve
)
from sklearn.preprocessing import MinMaxScaler
from src.dataset import get_dataloaders_from_csv
from src.constraints import WADIConstraintChecker

device = torch.device('cpu')

def normalize_invert(val_scores, atk_scores):
    """
    Normalize scores using normal baseline range,
    then invert so that HIGH score = MORE anomalous.
    This preserves the continuous magnitude signal
    instead of throwing it away with binary prediction.
    """
    combined = np.concatenate([val_scores, atk_scores])
    mn, mx = combined.min(), combined.max()
    norm_atk = (atk_scores - mn) / (mx - mn + 1e-9)
    return 1.0 - norm_atk


def evaluate_layer_3():
    print("=" * 52)
    print("   LAYER 3: HYBRID DEVIATION DETECTION (EVAL)")
    print("=" * 52)

    # ---------------------------------------------------------
    # 1. LOAD DATA AND PRE-COMPUTED RESIDUALS
    # ---------------------------------------------------------
    print("\n[1/6] Loading datasets and pre-computed residuals...")
    _, val_loader, attack_loader, windowed_labels, sensor_cols = \
        get_dataloaders_from_csv(batch_size=128)

    val_residuals    = np.load('artifacts/val_residuals.npy')
    attack_residuals = np.load('artifacts/attack_residuals.npy')

    # Fix label encoding — handle both -1/1 and 0/1 formats
    windowed_labels = np.array(windowed_labels)
    if -1 in windowed_labels:
        windowed_labels = (windowed_labels == -1).astype(int)
    else:
        windowed_labels = (windowed_labels > 0).astype(int)

    # Diagnostic — critical to understand your data
    total     = len(windowed_labels)
    n_attack  = windowed_labels.sum()
    n_normal  = total - n_attack
    print(f"      Total windows:  {total:,}")
    print(f"      Attack windows: {n_attack:,} ({n_attack/total*100:.1f}%)")
    print(f"      Normal windows: {n_normal:,} ({n_normal/total*100:.1f}%)")

    # ---------------------------------------------------------
    # 2. CYBER MODULE — Isolation Forest on Actuator Sequences
    # ---------------------------------------------------------
    print("\n[2/6] Training Cyber Module (Isolation Forest)...")

    actuator_indices = [
        i for i, col in enumerate(sensor_cols)
        if 'STATUS' in col or '_CO' in col or 'SPEED' in col
    ]
    print(f"      Using {len(actuator_indices)} actuator/controller columns.")

    def extract_actuators(loader, desc):
        act_data = []
        for batch in tqdm(loader, desc=desc, leave=False):
            act_data.append(batch[:, -1, actuator_indices].numpy())
        return np.concatenate(act_data, axis=0)

    val_actuators    = extract_actuators(val_loader,    "Extracting Val Actuators")
    attack_actuators = extract_actuators(attack_loader, "Extracting Attack Actuators")

    # Contamination matched to actual imbalance rate
    iso_forest = IsolationForest(
        contamination=0.058, random_state=42, n_jobs=-1
    )
    iso_forest.fit(val_actuators)

    # Use decision_function — continuous score, not binary
    # More negative = more anomalous
    val_iso_scores = iso_forest.decision_function(val_actuators)
    atk_iso_scores = iso_forest.decision_function(attack_actuators)
    P_cyber = normalize_invert(val_iso_scores, atk_iso_scores)

    print(f"      P_cyber  — mean normal: {normalize_invert(val_iso_scores, val_iso_scores).mean():.4f} | "
          f"mean attack: {P_cyber.mean():.4f}")

    # ---------------------------------------------------------
    # 3. PHYSICAL MODULE — One-Class SVM on LSTM Residuals
    # ---------------------------------------------------------
    print("\n[3/6] Training Physical Module (One-Class SVM)...")

    svm = OneClassSVM(kernel='rbf', nu=0.01, gamma=0.1)
    svm.fit(val_residuals.reshape(-1, 1))

    # Continuous decision scores — not binary predictions
    val_svm_scores = svm.decision_function(val_residuals.reshape(-1, 1))
    atk_svm_scores = svm.decision_function(attack_residuals.reshape(-1, 1))
    P_physical_lstm = normalize_invert(val_svm_scores, atk_svm_scores)

    print(f"      P_physical_lstm — mean normal: "
          f"{normalize_invert(val_svm_scores, val_svm_scores).mean():.4f} | "
          f"mean attack: {P_physical_lstm.mean():.4f}")

    # ---------------------------------------------------------
    # 4. PHYSICAL MODULE — Hard Constraint Rules
    # ---------------------------------------------------------
    print("\n[4/6] Evaluating Hard Physical Constraints...")

    constraint_checker = WADIConstraintChecker(sensor_cols)
    P_physical_constraints = []

    for batch in tqdm(attack_loader, desc="Checking Constraints", leave=False):
        batch_np = batch.numpy()
        for i in range(batch_np.shape[0]):
            _, score, _ = constraint_checker.check(batch_np[i])
            P_physical_constraints.append(score)

    P_physical_constraints = np.array(P_physical_constraints)
    print(f"      Constraints mean score: {P_physical_constraints.mean():.4f} | "
          f"max: {P_physical_constraints.max():.4f}")

    # Combine physical signals — take maximum (if either fires, physics violated)
    P_physical = np.maximum(P_physical_lstm, P_physical_constraints)

    # ---------------------------------------------------------
    # 5. DECISION FUSION — PR-Optimal Threshold
    # ---------------------------------------------------------
    print("\n[5/6] Decision Fusion Grid Search (PR-Optimal Threshold)...")
    print(f"\n{'Alpha':>5} | {'F1':>6} | {'Prec':>6} | {'Recall':>6} "
          f"| {'FPR':>6} | {'Bal-F1':>6} | {'Thresh':>6}")
    print("-" * 70)

    best_f1       = 0
    best_alpha    = 0
    best_thresh   = 0
    best_P_final  = None
    results_table = []

    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        P_final = alpha * P_cyber + (1 - alpha) * P_physical

        # PR-optimal threshold
        precision_arr, recall_arr, pr_thresholds = precision_recall_curve(
            windowed_labels, P_final
        )
        pr_f1 = (2 * precision_arr * recall_arr / 
                 (precision_arr + recall_arr + 1e-9))
        best_pr_idx = np.argmax(pr_f1)
        thresh = float(pr_thresholds[best_pr_idx]) \
                 if best_pr_idx < len(pr_thresholds) else 0.5

        predictions = (P_final > thresh).astype(int)

        f1   = f1_score(windowed_labels,   predictions, zero_division=0)
        prec = precision_score(windowed_labels, predictions, zero_division=0)
        rec  = recall_score(windowed_labels,  predictions, zero_division=0)
        fpr  = float((predictions[windowed_labels == 0]).mean())
        
        # Weighted F1 for context
        balanced = f1_score(windowed_labels, predictions, average='weighted', zero_division=0)

        marker = " <- Sayghe" if alpha == 0.3 else ""
        print(f"{alpha:>5.1f} | {f1:>6.4f} | {prec:>6.4f} | {rec:>6.4f} "
              f"| {fpr:>6.4f} | {balanced:>6.4f} | {thresh:>6.4f}{marker}")

        results_table.append({
            'alpha': alpha, 'f1': f1, 'precision': prec,
            'recall': rec, 'fpr': fpr, 'balanced_f1': balanced, 'threshold': thresh
        })

        if f1 > best_f1:
            best_f1      = f1
            best_alpha   = alpha
            best_thresh  = thresh
            best_P_final = P_final.copy()

    print("-" * 70)
    print(f"\nOPTIMAL: Alpha = {best_alpha} | "
          f"F1 = {best_f1:.4f} | Threshold = {best_thresh:.4f}")

    # ---------------------------------------------------------
    # 6. SAVE ARTIFACTS FOR LAYER 4
    # ---------------------------------------------------------
    print("\n[6/6] Saving artifacts for Layer 4 (Reliability Translation)...")

    np.save('artifacts/P_final_scores.npy',        best_P_final)
    np.save('artifacts/P_cyber_scores.npy',         P_cyber)
    np.save('artifacts/P_physical_scores.npy',      P_physical)
    np.save('artifacts/P_constraints_scores.npy',   P_physical_constraints)
    np.save('artifacts/windowed_labels.npy',         windowed_labels)

    # Update training artifacts JSON
    with open('artifacts/training_artifacts.json', 'r+') as f:
        artifacts = json.load(f)
        artifacts['optimal_alpha']     = best_alpha
        artifacts['optimal_threshold'] = best_thresh
        artifacts['layer3_f1']         = best_f1
        artifacts['layer3_results']    = results_table
        f.seek(0)
        json.dump(artifacts, f, indent=2)
        f.truncate()

    print("\nArtifacts saved:")
    print("  artifacts/P_final_scores.npy")
    print("  artifacts/P_cyber_scores.npy")
    print("  artifacts/P_physical_scores.npy")
    print("  artifacts/P_constraints_scores.npy")
    print("  artifacts/windowed_labels.npy")
    print("  artifacts/training_artifacts.json  (updated)")
    print("\nReady for Layer 4 — reliability.py")


if __name__ == "__main__":
    evaluate_layer_3()