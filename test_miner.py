import pandas as pd
import re
import numpy as np

csv_path = "data/Virtue Foundation Ghana v0.3 - Sheet1.csv"
df = pd.read_csv(csv_path, encoding='latin1')
df.columns = [c.strip() for c in df.columns]

lat_pat = r"latitude\s*[:\s]*([-+]?[0-9]*\.?[0-9]+)"
lon_pat = r"longitude\s*[:\s]*([-+]?[0-9]*\.?[0-9]+)"

def extract_coord(text, pattern):
    if pd.isna(text): return None
    match = re.search(pattern, str(text), re.IGNORECASE)
    return float(match.group(1)) if match else None

df['latitude'] = np.nan
df['longitude'] = np.nan

text_cols = [c for c in df.columns if df[c].dtype == object]
for col in text_cols:
    mask = df['latitude'].isna()
    df.loc[mask, 'latitude'] = df.loc[mask, col].apply(lambda x: extract_coord(x, lat_pat))
    df.loc[mask, 'longitude'] = df.loc[mask, col].apply(lambda x: extract_coord(x, lon_pat))

valid = df.dropna(subset=['latitude', 'longitude'])
print(f"Total rows: {len(df)}")
print(f"Mined coordinates: {len(valid)}")
if not valid.empty:
    print(f"Sample mined data:\n{valid[['name', 'latitude', 'longitude']].head()}")
else:
    print("Miner failed to find any coordinates in text fields.")
