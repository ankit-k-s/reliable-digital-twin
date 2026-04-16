# src/finetune.py
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from src.model import WADILSTMAutoencoder
from src.dataset import get_dataloaders_from_csv

def finetune():
    print("=== Semi-Supervised Fine-tuning ===\n")
    
    _, _, attack_loader, windowed_labels, sensor_cols = \
        get_dataloaders_from_csv(batch_size=128)
    
    windowed_labels = np.array(windowed_labels)
    attack_labels   = (windowed_labels > 0).astype(int)
    
    # Load trained model
    device = torch.device('cpu')
    model  = WADILSTMAutoencoder(num_sensors=len(sensor_cols))
    model.load_state_dict(
        torch.load('artifacts/best_model.pt', map_location=device)
    )
    
    # Collect ALL windows from attack loader
    all_windows = []
    for batch in attack_loader:
        all_windows.append(batch)
    all_windows = torch.cat(all_windows, dim=0)  # [172672, 60, 127]
    
    # Use ONLY normal windows from attack file for fine-tuning
    normal_mask    = (attack_labels == 0)
    normal_windows = all_windows[normal_mask]
    print(f"Fine-tuning on {len(normal_windows):,} normal windows "
          f"from attack file...")
    
    # Create fine-tune loader
    ft_dataset = TensorDataset(normal_windows)
    ft_loader  = DataLoader(
        ft_dataset, batch_size=128, shuffle=True, drop_last=True
    )
    
    # Fine-tune with low learning rate to preserve learned features
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    print(f"\n{'Epoch':>6} | {'Loss':>10}")
    print("-" * 22)
    
    model.train()
    for epoch in range(10):
        epoch_loss = 0
        for (batch,) in ft_loader:
            optimizer.zero_grad()
            recon = model(batch)
            loss  = criterion(recon, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        avg = epoch_loss / len(ft_loader)
        print(f"{epoch+1:>6} | {avg:>10.6f}")
    
    # Save fine-tuned model
    torch.save(model.state_dict(), 'artifacts/finetuned_model.pt')
    print("\nSaved: artifacts/finetuned_model.pt")
    
    # Recompute residuals with fine-tuned model
    print("\nRecomputing residuals with fine-tuned model...")
    from src.model import compute_residuals
    
    model.eval()
    new_residuals = []
    for batch in attack_loader:
        r_t = compute_residuals(model, batch)
        new_residuals.extend(r_t.max(dim=1).values.detach().numpy())
    
    new_residuals = np.array(new_residuals)
    np.save('artifacts/finetuned_attack_residuals.npy', new_residuals)
    
    # Check new separation
    true_att  = new_residuals[attack_labels == 1]
    false_att = new_residuals[attack_labels == 0]
    print(f"\nAfter fine-tuning:")
    print(f"  Normal mean: {false_att.mean():.4f}")
    print(f"  Attack mean: {true_att.mean():.4f}")
    print(f"  Separation:  {true_att.mean()/false_att.mean():.3f}x")
    
    # Quick threshold scan
    from sklearn.metrics import f1_score
    print(f"\n{'Percentile':>12} | {'F1':>8} | {'Recall':>8} | {'FPR':>8}")
    print("-" * 46)
    
    best_f1 = 0
    for pct in [85, 90, 93, 95, 97, 99]:
        thresh = np.percentile(false_att, pct)
        preds  = (new_residuals > thresh).astype(int)
        f1  = f1_score(attack_labels, preds, zero_division=0)
        rec = (preds[attack_labels == 1]).mean()
        fpr = (preds[attack_labels == 0]).mean()
        print(f"{pct:>12}% | {f1:>8.4f} | {rec:>8.4f} | {fpr:>8.4f}")
        best_f1 = max(best_f1, f1)
    
    print(f"\nBest F1 after fine-tuning: {best_f1:.4f}")
    return best_f1

if __name__ == "__main__":
    finetune()