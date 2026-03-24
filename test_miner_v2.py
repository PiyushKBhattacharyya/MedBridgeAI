import pandas as pd
import re
import numpy as np

csv_path = "data/Virtue Foundation Ghana v0.3 - Sheet1.csv"
df = pd.read_csv(csv_path, encoding='latin1')
df.columns = [c.strip() for c in df.columns]

def extract_coord(text, is_lat=True):
    if pd.isna(text): return None
    text_str = str(text)
    pat1 = r"(?:latitude|lat)\s*[:\s]*([-+]?[0-9]*\.?[0-9]+)" if is_lat else r"(?:longitude|long|lon)\s*[:\s]*([-+]?[0-9]*\.?[0-9]+)"
    match1 = re.search(pat1, text_str, re.IGNORECASE)
    if match1: return float(match1.group(1))
    pat2 = r"([-+]?[0-9]*\.?[0-9]+)\s*(?:latitude|lat)" if is_lat else r"([-+]?[0-9]*\.?[0-9]+)\s*(?:longitude|long|lon)"
    match2 = re.search(pat2, text_str, re.IGNORECASE)
    if match2: return float(match2.group(1))
    return None

df['latitude'] = np.nan
df['longitude'] = np.nan

text_cols = [c for c in df.columns if df[c].dtype == object]
for col in text_cols:
    mask = df['latitude'].isna()
    df.loc[mask, 'latitude'] = df.loc[mask, col].apply(lambda x: extract_coord(x, is_lat=True))
    df.loc[mask, 'longitude'] = df.loc[mask, col].apply(lambda x: extract_coord(x, is_lat=False))

valid = df.dropna(subset=['latitude', 'longitude'])
print(f"Total rows: {len(df)}")
print(f"Mined coordinates: {len(valid)}")
if not valid.empty:
    print(f"Sample mined data:\n{valid[['name', 'latitude', 'longitude']].head(20)}")
else:
    print("Miner failed to find any coordinates in text fields.")
