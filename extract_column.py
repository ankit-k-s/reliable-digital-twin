import pandas as pd
df = pd.read_csv('data/WADI_14days_new.csv', nrows=2, low_memory=False)
cols = [c for c in df.columns if c.strip() not in ['Row','Date','Time']]
for i, c in enumerate(cols):
    print(f"{i:3d}: {c}")