from src.dataset import get_dataloaders_from_csv
import numpy as np

print("Loading data to check label distribution...")
_, _, _, windowed_labels, _ = get_dataloaders_from_csv(batch_size=128)

wl = np.array(windowed_labels)

# Clean labels just in case they are -1
if -1 in wl:
    wl = (wl == -1).astype(int)
else:
    wl = (wl > 0).astype(int)

print(f"\nLabel distribution:")
print(f"  Total windows: {len(wl)}")
print(f"  Attack windows: {wl.sum()}")
print(f"  Normal windows: {len(wl) - wl.sum()}")
print(f"  Attack %: {(wl.mean()) * 100:.2f}%")