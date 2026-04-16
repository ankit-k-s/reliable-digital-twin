import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

def preprocess_wadi_data():
    # Define paths
    data_dir = 'data'
    normal_file = os.path.join(data_dir, 'WADI_14days_new.csv')
    attack_file = os.path.join(data_dir, 'WADI_attackdataLABLE.csv')

    print("--- Starting WADI Preprocessing ---")

    # ---------------------------------------------------------
    # 1. PROCESS NORMAL DATA
    # ---------------------------------------------------------
    print(f"Loading normal data from {normal_file}...")
    # low_memory=False stops Pandas from guessing data types chunk-by-chunk
    df_normal = pd.read_csv(normal_file, low_memory=False)

    print("Cleaning normal data (dropping empty columns, filling NaNs)...")
    df_normal = df_normal.dropna(axis=1, how='all')
    df_normal = df_normal.ffill().bfill()

    cols_to_drop = [c for c in df_normal.columns if c.strip() in ['Row', 'Date', 'Time']]
    sensor_normal = df_normal.drop(columns=cols_to_drop)

    # BULLETPROOFING: Force all data to be numeric. Any weird strings become NaNs, 
    # which we then forward-fill away.
    sensor_normal = sensor_normal.apply(pd.to_numeric, errors='coerce')
    sensor_normal = sensor_normal.ffill().bfill()

    print("Scaling normal data...")
    scaler = MinMaxScaler()
    sensor_normal_scaled = scaler.fit_transform(sensor_normal)

    scaler_path = os.path.join(data_dir, 'wadi_scaler.pkl')
    joblib.dump(scaler, scaler_path)
    
    # ---------------------------------------------------------
    # 2. PROCESS ATTACK DATA
    # ---------------------------------------------------------
    print(f"\nLoading attack data from {attack_file}...")
    df_attack = pd.read_csv(attack_file, low_memory=False)

    # BULLETPROOFING: Smarter label column finder
    # Looks for 'attack' or 'label' anywhere in the header, regardless of case
    label_candidates = [c for c in df_attack.columns if 'attack' in c.lower() or 'label' in c.lower()]
    
    if not label_candidates:
        print("\nWARNING: Could not find standard label column. Using the last column.")
        label_col = df_attack.columns[-1]
    else:
        label_col = label_candidates[0]
        
    print(f"Found label column: '{label_col}'")
    attack_labels = df_attack[label_col].apply(pd.to_numeric, errors='coerce').fillna(1).values

    print("Cleaning attack data and forcing feature alignment...")
    sensor_attack = df_attack[sensor_normal.columns].copy()

    # Force numeric and fill
    sensor_attack = sensor_attack.apply(pd.to_numeric, errors='coerce')
    sensor_attack = sensor_attack.ffill().bfill()

    print("Scaling attack data...")
    sensor_attack_scaled = scaler.transform(sensor_attack)

    # ---------------------------------------------------------
    # 3. SAVE ARTIFACTS
    # ---------------------------------------------------------
    print("\nSaving processed data as high-speed NumPy arrays...")
    np.save(os.path.join(data_dir, 'normal_scaled.npy'), sensor_normal_scaled)
    np.save(os.path.join(data_dir, 'attack_scaled.npy'), sensor_attack_scaled)
    np.save(os.path.join(data_dir, 'attack_labels.npy'), attack_labels)

    print("\n--- Preprocessing Complete ---")
    print(f"Normal data shape: {sensor_normal_scaled.shape}")
    print(f"Attack data shape: {sensor_attack_scaled.shape}")
    print("Files successfully saved to the 'data/' directory.")

if __name__ == "__main__":
    preprocess_wadi_data()