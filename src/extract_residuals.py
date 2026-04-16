# src/extract_residuals.py
import torch
import numpy as np
from src.dataset import get_dataloaders_from_csv
from src.model import WADILSTMAutoencoder, compute_residuals

device = torch.device('cpu')
_, val_loader, attack_loader, windowed_labels, sensor_cols = \
    get_dataloaders_from_csv(batch_size=128)

model = WADILSTMAutoencoder(num_sensors=len(sensor_cols))
model.load_state_dict(
    torch.load('artifacts/best_model.pt', map_location=device)
)
model.eval()

print("Extracting validation residuals...")
val_residuals = []
for batch in val_loader:
    r_t = compute_residuals(model, batch)
    val_residuals.extend(r_t.max(dim=1).values.numpy())

print("Extracting attack residuals...")
attack_residuals = []
for batch in attack_loader:
    r_t = compute_residuals(model, batch)
    attack_residuals.extend(r_t.max(dim=1).values.numpy())

val_residuals = np.array(val_residuals)
attack_residuals = np.array(attack_residuals)

np.save('artifacts/val_residuals.npy', val_residuals)
np.save('artifacts/attack_residuals.npy', attack_residuals)

print(f"Val residuals:    {val_residuals.shape}")
print(f"Attack residuals: {attack_residuals.shape}")
print(f"Val mean:         {val_residuals.mean():.6f}")
print(f"Attack mean:      {attack_residuals.mean():.6f}")
print(f"Separation ratio: {attack_residuals.mean()/val_residuals.mean():.2f}x")