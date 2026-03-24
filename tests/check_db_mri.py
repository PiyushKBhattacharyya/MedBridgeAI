import os
import sys
import pandas as pd
from src.synthesis.database import MedBridgeStore

def check_mri():
    store = MedBridgeStore()
    
    # Try a simple text search if possible, or just vector search
    query = "MRI"
    print(f"Searching for: {query}")
    
    results = store.search_facilities(query, limit=10)
    if not results.empty:
        print("\nFound Facilities:")
        for _, row in results.iterrows():
            print(f"- {row['name']}: {row.get('description', '')}")
            print(f"  Equipment: {row.get('equipment', '')}")
            print(f"  Capability: {row.get('capability', '')}")
            print("-" * 20)
    else:
        print("No facilities found.")

    ngo_results = store.search_ngos(query, limit=10)
    if not ngo_results.empty:
        print("\nFound NGOs:")
        for _, row in ngo_results.iterrows():
            print(f"- {row['name']}: {row.get('organizationDescription', '')}")
            print("-" * 20)

if __name__ == "__main__":
    check_mri()
