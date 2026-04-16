import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from tqdm import tqdm  # <-- NEW IMPORT
from src.dataset import get_dataloaders_from_csv
from src.model import WADILSTMAutoencoder, compute_residuals

# Force PyTorch to use your Intel performance cores
torch.set_num_threads(16)

def train(
    data_dir='data',
    save_dir='artifacts',
    window_size=60,
    batch_size=128,
    epochs=30,
    learning_rate=1e-3,
    hidden_dim=64,
    latent_dim=32,
):
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    train_loader, val_loader, attack_loader, windowed_labels, sensor_cols = \
        get_dataloaders_from_csv(
            data_dir=data_dir,
            window_size=window_size,
            batch_size=batch_size
        )

    model = WADILSTMAutoencoder(
        num_sensors=len(sensor_cols),
        hidden_dim=hidden_dim,
        latent_dim=latent_dim
    ).to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    print(f"\nStarting training for {epochs} epochs...")
    print("-" * 50)

    for epoch in range(epochs):

        # --- Training phase ---
        model.train()
        epoch_train_loss = 0
        
        # <-- NEW: Wrap train_loader with tqdm progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1:02d}/{epochs}] [Train]", leave=False)
        
        for batch in train_pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstruction = model(batch)
            loss = criterion(reconstruction, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_train_loss += loss.item()
            
            # <-- NEW: Update the progress bar with the current batch loss
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- Validation phase ---
        model.eval()
        epoch_val_loss = 0
        
        # <-- NEW: Wrap val_loader with tqdm progress bar
        val_pbar = tqdm(val_loader, desc=f"Epoch [{epoch+1:02d}/{epochs}] [Val]", leave=False)
        
        with torch.no_grad():
            for batch in val_pbar:
                batch = batch.to(device)
                reconstruction = model(batch)
                loss = criterion(reconstruction, batch)
                epoch_val_loss += loss.item()
                val_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        # Print the final summary for the epoch
        print(f"Epoch [{epoch+1:02d}/{epochs}] "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))
            print(f"  -> Best model saved (val_loss={best_val_loss:.6f})")

    # --- Derive anomaly threshold from validation residuals ---
    print("\nDeriving anomaly threshold from validation set...")
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pt')))
    model.eval()

    val_residuals = []
    
    # <-- NEW: Add progress bar here too, as it takes a moment
    thresh_pbar = tqdm(val_loader, desc="Calculating Threshold")
    with torch.no_grad():
        for batch in thresh_pbar:
            batch = batch.to(device)
            r_t = compute_residuals(model, batch)
            val_residuals.extend(r_t.max(dim=1).values.cpu().numpy())

    val_residuals = np.array(val_residuals)
    threshold = float(np.percentile(val_residuals, 95))
    print(f"\nAnomaly threshold (95th percentile): {threshold:.6f}")

    # --- Per-sensor reconstruction error profile ---
    print("Computing per-sensor error profile...")
    sensor_errors = []
    
    sensor_pbar = tqdm(val_loader, desc="Profiling Sensors")
    with torch.no_grad():
        for batch in sensor_pbar:
            batch = batch.to(device)
            reconstruction = model(batch)
            err = torch.abs(batch - reconstruction).mean(dim=(0, 1))
            sensor_errors.append(err.cpu().numpy())

    avg_sensor_errors = np.mean(sensor_errors, axis=0)
    top10_sensors = np.argsort(avg_sensor_errors)[-10:][::-1]
    print("\nTop 10 most sensitive sensors:")
    for rank, idx in enumerate(top10_sensors):
        print(f"  {rank+1}. {sensor_cols[idx]} "
              f"(avg error: {avg_sensor_errors[idx]:.6f})")

    # --- Save all artifacts ---
    artifacts = {
        'threshold': threshold,
        'best_val_loss': best_val_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'top10_sensor_indices': top10_sensors.tolist(),
        'top10_sensor_names': [sensor_cols[i] for i in top10_sensors],
        'sensor_cols': sensor_cols,
        'config': {
            'window_size': window_size,
            'batch_size': batch_size,
            'epochs': epochs,
            'hidden_dim': hidden_dim,
            'latent_dim': latent_dim,
        }
    }
    with open(os.path.join(save_dir, 'training_artifacts.json'), 'w') as f:
        json.dump(artifacts, f, indent=2)

    # --- Plot training curves ---
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.axhline(y=threshold, color='red', linestyle='--',
                label=f'Anomaly threshold ({threshold:.4f})')
    plt.xlabel('Epoch')
    plt.ylabel('L1 Loss (MAE)') 
    plt.title('LSTM Autoencoder Training — WADI Normal Behaviour')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    plt.close()

    print("\nTraining complete. Artifacts saved:")
    print(f"  {save_dir}/best_model.pt")
    print(f"  {save_dir}/training_artifacts.json")
    print(f"  {save_dir}/training_curves.png")

    return model, threshold, artifacts


if __name__ == "__main__":
    train(
        data_dir='data',
        save_dir='artifacts',
        window_size=60,
        batch_size=128,
        epochs=30,
        learning_rate=1e-3,
    )