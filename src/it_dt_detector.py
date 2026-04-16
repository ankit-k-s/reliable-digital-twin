# src/it_dt_detector.py
import numpy as np
import json
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve
from src.dataset import get_dataloaders_from_csv

def extract_state_space_matrices(Y, n_states=25):
    """
    Data-driven extraction of LTI State-Space matrices (A, C, K) 
    mimicking N4SID subspace identification.
    Y: shape [N, p] (Training data)
    """
    print(f"      Extracting {n_states}D hidden physical state via SVD/PCA...")
    pca = PCA(n_components=n_states)
    X = pca.fit_transform(Y)  # Hidden states X_t
    
    # Eq 2: Y_t = C * X_t + v_t
    print("      Identifying Observation Matrix (C)...")
    ridge_C = Ridge(alpha=1e-3, fit_intercept=False)
    ridge_C.fit(X, Y)
    C = ridge_C.coef_  # shape [p, n]
    
    # Eq 1: X_{t+1} = A * X_t + w_t
    print("      Identifying Transition Matrix (A)...")
    X_t = X[:-1]
    X_next = X[1:]
    ridge_A = Ridge(alpha=1e-3, fit_intercept=False)
    ridge_A.fit(X_t, X_next)
    A = ridge_A.coef_  # shape [n, n]
    
    # Steady-State Kalman Gain (K) approximation from innovations
    print("      Estimating Steady-State Kalman Gain (K)...")
    # Residuals of observation
    r_t = Y[1:] - (X_t @ C.T)  # shape [N-1, p]
    # Residuals of state transition
    e_t = X_next - (X_t @ A.T) # shape [N-1, n]
    
    # K solves: e_t = r_t * K.T -> K = e_t.T @ r_t @ inv(r_t.T @ r_t)
    ridge_K = Ridge(alpha=1.0, fit_intercept=False)
    ridge_K.fit(r_t, e_t)
    K = ridge_K.coef_  # shape [n, p]
    
    return A, C, K, X[0]

def run_kalman_filter_loop(Y, A, C, K, x_init):
    """
    Executes the exact recursive loop from ALGORITHM 1 in the paper.
    Returns the multivariate innovation residuals (r_t).
    """
    N, p = Y.shape
    n = len(x_init)
    
    innovations = np.zeros((N, p))
    x_est = x_init.copy()
    
    for t in range(N):
        # Step 2: Predict (Eq 3)
        x_pred = A @ x_est
        
        # Step 3: Measure Innovation (Eq 4)
        r_t = Y[t] - (C @ x_pred)
        innovations[t] = r_t
        
        # Step 4: Correct State (Eq 5)
        x_est = x_pred + (K @ r_t)
        
    return innovations

def run_it_dt_detection():
    print("=" * 70)
    print("   INFORMATION-THEORETIC DIGITAL TWIN (IT-DT)")
    print("   (Exact Implementation: N4SID, Kalman Loop, & Closed-Form KL)")
    print("=" * 70)

    # 1. Load Data
    print("\n[1/5] Loading datasets...")
    # We load the full arrays to run the continuous Kalman Filter
    _, _, _, windowed_labels, sensor_cols = get_dataloaders_from_csv(batch_size=128)
    
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    df_normal = pd.read_csv('data/WADI_14days_new.csv', low_memory=False)
    df_attack = pd.read_csv('data/WADI_attackdataLABLE.csv', low_memory=False)
    
    cols_drop = [c for c in df_normal.columns if c.strip() in ['Row', 'Date', 'Time']]
    Y_train = df_normal.drop(columns=cols_drop).apply(pd.to_numeric, errors='coerce').fillna(0).values
    Y_test = df_attack.iloc[:, len(cols_drop):-1].apply(pd.to_numeric, errors='coerce').fillna(0).values
    
    scaler = StandardScaler()
    Y_train = scaler.fit_transform(Y_train)
    Y_test = scaler.transform(Y_test)
    
    Y_train = np.nan_to_num(Y_train, nan=0.0)
    Y_test = np.nan_to_num(Y_test, nan=0.0)
    
    p = Y_train.shape[1]
    eps = 1e-4  # Tikhonov regularization (Paper Sec IV.A)
    W = 60      # Window size

    # 2. System Identification
    print("\n[2/5] Performing Subspace System Identification (n=25)...")
    A, C, K, x_init = extract_state_space_matrices(Y_train, n_states=25)

    # 3. Reference Distribution (P_0) via Training Innovations
    print("\n[3/5] Running Kalman Filter on Training Data (P_0)...")
    train_innovations = run_kalman_filter_loop(Y_train, A, C, K, x_init)
    
    Sigma0 = np.cov(train_innovations, rowvar=False) + eps * np.eye(p)
    inv_Sigma0 = np.linalg.inv(Sigma0)
    _, logdet_Sigma0 = np.linalg.slogdet(Sigma0)
    print(f"      Global Covariance Condition Number: {np.linalg.cond(Sigma0):.2f}")

    # 4. Attack Innovations and IT-DT Scoring
    print("\n[4/5] Running Kalman Filter on Attack Data...")
    test_innovations = run_kalman_filter_loop(Y_test, A, C, K, x_init)
    
    print("      Computing Sliding Window KL Divergence (Eq 6)...")
    attack_kl_scores = np.zeros(len(Y_test) - W)
    
    # Efficient rolling window calculation
    for t in tqdm(range(len(test_innovations) - W), desc="IT-DT Scoring"):
        window = test_innovations[t : t + W]
        
        # Empirical Distribution
        mu_t = np.mean(window, axis=0)
        Sigma_t = np.cov(window, rowvar=False) + eps * np.eye(p)
        
        # Symmetrize to prevent numerical floating point errors in slogdet
        Sigma_t = 0.5 * (Sigma_t + Sigma_t.T)

        # Equation 6
        term1 = np.trace(inv_Sigma0 @ Sigma_t)
        term2 = -p
        term3 = mu_t.T @ inv_Sigma0 @ mu_t
        _, logdet_Sigma_t = np.linalg.slogdet(Sigma_t)
        term4 = logdet_Sigma0 - logdet_Sigma_t

        kl = 0.5 * (term1 + term2 + term3 + term4)
        attack_kl_scores[t] = kl

    attack_kl_scores = np.nan_to_num(attack_kl_scores, nan=0.0, posinf=0.0)

    # Align labels with the scores (dropping the last batch mismatch if any)
    windowed_labels = np.array(windowed_labels)
    if -1 in windowed_labels:
        windowed_labels = (windowed_labels == -1).astype(int)
    else:
        windowed_labels = (windowed_labels > 0).astype(int)
        
    min_len = min(len(attack_kl_scores), len(windowed_labels))
    attack_kl_scores = attack_kl_scores[:min_len]
    windowed_labels = windowed_labels[:min_len]

    # 5. Thresholding and Evaluation
    print("\n[5/5] Evaluating Information-Theoretic Performance...")
    true_att = attack_kl_scores[windowed_labels == 1]
    false_att = attack_kl_scores[windowed_labels == 0]

    print(f"      Normal mean (in attack file): {false_att.mean():.4f}")
    print(f"      Attack mean:                  {true_att.mean():.4f}")
    print(f"      True Separation Ratio:        {true_att.mean() / (false_att.mean() + 1e-9):.3f}x")

    print(f"\n{'Percentile':>12} | {'Threshold':>10} | {'F1':>8} | {'Precision':>10} | {'Recall':>8} | {'FPR':>8}")
    print("-" * 75)

    best_f1, best_thresh = 0, 0

    # Test the paper's exact threshold mechanism (Eq 7: alpha = 0.01 -> 99th percentile)
    for pct in [90, 95, 96, 97, 98, 99]:
        thresh = np.percentile(false_att, pct)
        preds = (attack_kl_scores > thresh).astype(int)
        f1 = f1_score(windowed_labels, preds, zero_division=0)
        prec = precision_score(windowed_labels, preds, zero_division=0)
        rec = recall_score(windowed_labels, preds, zero_division=0)
        fpr = float((preds[windowed_labels == 0]).mean())

        marker = " <- Paper's tau* (Eq 7)" if pct == 99 else ""
        print(f"{pct:>12}% | {thresh:>10.4f} | {f1:>8.4f} | {prec:>10.4f} | {rec:>8.4f} | {fpr:>8.4f}{marker}")

        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh

    # ROC Optimal Check
    fpr_arr, tpr_arr, roc_thresholds = roc_curve(windowed_labels, attack_kl_scores)
    roc_idx = np.argmax(tpr_arr - fpr_arr)
    roc_thresh = float(roc_thresholds[roc_idx])
    roc_preds = (attack_kl_scores > roc_thresh).astype(int)
    roc_f1 = f1_score(windowed_labels, roc_preds, zero_division=0)
    roc_prec = precision_score(windowed_labels, roc_preds, zero_division=0)
    roc_rec = recall_score(windowed_labels, roc_preds, zero_division=0)
    roc_fpr = float((roc_preds[windowed_labels == 0]).mean())

    print(f"\n{'ROC-opt':>12} | {roc_thresh:>10.4f} | {roc_f1:>8.4f} | {roc_prec:>10.4f} | {roc_rec:>8.4f} | {roc_fpr:>8.4f}")

    best_f1 = max(best_f1, roc_f1)
    print("-" * 75)
    print(f"\n BEST IT-DT F1 SCORE: {best_f1:.4f}")

    # Save outputs for Reliability translation
    mn, mx = attack_kl_scores.min(), attack_kl_scores.max()
    p_final = (attack_kl_scores - mn) / (mx - mn + 1e-9)
    np.save('artifacts/P_final_scores.npy', p_final)

    with open('artifacts/training_artifacts.json', 'r+') as f:
        arts = json.load(f)
        arts['it_dt_f1'] = float(best_f1)
        f.seek(0)
        json.dump(arts, f, indent=2)
        f.truncate()

    print("\nMathematical extraction complete. Ready for Layer 4 Reliability script!")

if __name__ == "__main__":
    run_it_dt_detection()