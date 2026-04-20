import os
import sys
import json
import pandas as pd

# Add project folder to sys.path so 'src' can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.synthesis.database import MedBridgeStore

def create_snapshot():
    print("[SNAPSHOT] Creating a lightweight snapshot of the analyzed data...")
    store = MedBridgeStore()
    
    f_df = store.get_all_facilities()
    n_df = store.get_all_ngos()
    
    if f_df.empty and n_df.empty:
        print("[ERROR] No analyzed data found in the local database. Please analyze a document first!")
        return

    # Convert to list of dicts for JSON serialization
    # We drop the 'vector' column to keep the file small
    # We also ensure all numpy types are converted to native Python types
    def sanitize_for_json(df):
        if df.empty: return []
        # Convert df to dict and then handle any non-serializable objects
        records = df.drop(columns=['vector']).to_dict(orient='records')
        return json.loads(pd.Series(records).to_json(orient='records'))

    data = {
        "facilities": sanitize_for_json(f_df),
        "ngos": sanitize_for_json(n_df)
    }
    
    output_path = os.path.join("data", "seed_data.json")
    os.makedirs("data", exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
        
    print(f"[SUCCESS] Snapshot saved to {output_path} ({len(data['facilities'])} facilities, {len(data['ngos'])} NGOs).")
    print("[INFO] You can now push this file to GitHub, and the map will be populated on Streamlit Cloud!")

if __name__ == "__main__":
    create_snapshot()
