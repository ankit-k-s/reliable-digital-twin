# src/dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
import warnings

class WADIDataset(Dataset):
    def __init__(self, data_array, window_size=60):
        """
        Args:
            data_array (numpy.ndarray): The scaled sensor data.
            window_size (int): How many timesteps to include in one window.
        """
        self.data = torch.FloatTensor(data_array)
        self.window_size = window_size
        self.num_samples = len(self.data) - self.window_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        window = self.data[idx : idx + self.window_size]
        return window

def get_dataloaders_from_csv(data_dir='data', window_size=60, batch_size=128, train_split=0.8):
    """
    Loads CSVs, handles temporal windowing, forces label alignment, and exports metadata.
    """
    print("Loading CSV files directly into memory...")
    normal_file = os.path.join(data_dir, 'WADI_14days_new.csv')
    attack_file = os.path.join(data_dir, 'WADI_attackdataLABLE.csv')

    df_normal = pd.read_csv(normal_file, low_memory=False)
    df_attack = pd.read_csv(attack_file, low_memory=False)

    print("Cleaning and aligning data...")
    cols_to_drop = [c for c in df_normal.columns if c.strip() in ['Row', 'Date', 'Time']]
    
    # Isolate pure sensor data
    sensor_normal_df = df_normal.drop(columns=cols_to_drop)
    sensor_cols = sensor_normal_df.columns.tolist() # Save for Layer 6 Dashboard
    sensor_normal = sensor_normal_df.apply(pd.to_numeric, errors='coerce').fillna(0).values
    
    # Attack data processing
    attack_labels_raw = pd.to_numeric(df_attack.iloc[:, -1], errors='coerce').fillna(1).values
    num_meta_cols = len(cols_to_drop)
    sensor_attack = df_attack.iloc[:, num_meta_cols:-1].apply(pd.to_numeric, errors='coerce').fillna(0).values

    # Bulletproof check: Ensure physical topologies match before scaling
    assert sensor_normal.shape[1] == sensor_attack.shape[1], \
        f"Column mismatch: normal={sensor_normal.shape[1]}, attack={sensor_attack.shape[1]}"

    # --- THE CRITICAL FIX: Z-Score Standardization ---
    print("Scaling data to standard normal (Z-score)...")
    scaler = StandardScaler()
    normal_scaled = scaler.fit_transform(sensor_normal)
    attack_scaled = scaler.transform(sensor_attack)
    # -------------------------------------------------

    # Safety net for any dirty data coercion
    normal_scaled = np.nan_to_num(normal_scaled, nan=0.0)
    attack_scaled = np.nan_to_num(attack_scaled, nan=0.0)

    # Export scaler to artifacts folder
    os.makedirs('artifacts', exist_ok=True)
    joblib.dump(scaler, 'artifacts/wadi_scaler.pkl')

    # Train/Val split
    split_idx = int(len(normal_scaled) * train_split)
    train_data = normal_scaled[:split_idx]
    val_data = normal_scaled[split_idx:]

    print(f"Creating datasets with sliding window of {window_size}...")
    train_dataset = WADIDataset(train_data, window_size)
    val_dataset = WADIDataset(val_data, window_size)
    attack_dataset = WADIDataset(attack_scaled, window_size)

    # Align labels with sliding windows (causal alignment)
    windowed_labels = np.array([
        attack_labels_raw[i + window_size - 1] 
        for i in range(len(attack_labels_raw) - window_size)
    ])

    # Sync windowed_labels with DataLoader drop_last=True
    n_complete_batches = len(attack_dataset) // batch_size
    n_used = n_complete_batches * batch_size
    windowed_labels = windowed_labels[:n_used]

    # Initialize loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    attack_loader = DataLoader(attack_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    print("DataLoaders ready.")
    return train_loader, val_loader, attack_loader, windowed_labels, sensor_cols

if __name__ == "__main__":
    train_loader, val_loader, attack_loader, labels, cols = get_dataloaders_from_csv(window_size=60, batch_size=128)
    
    for batch in train_loader:
        print(f"Batch shape: {batch.shape}") 
        break
    print(f"Synced label shape: {labels.shape}")