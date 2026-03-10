import pandas as pd
import random
import os

def create_sample_dataset(csv_path: str, output_dir: str):
    """
    Reads the real CSV dataset and creates markdown files for all relevant rows
    containing the raw unstructured text fields we want the IDP agent to extract from.
    We will feed these into our agent to test extraction.
    """
    df = pd.read_csv(csv_path)
    
    # We want rows that actually have some descriptive text to extract facts from
    # Filter for rows where any of these fields are not null
    samples = df[df['description'].notna() | df['capability'].notna() | df['equipment'].notna() | df['procedure'].notna() | df['missionStatement'].notna()]
    
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, row in samples.iterrows():
        # Create a safe filename using the organization name and ID
        safe_name = str(row['name']).replace("/", "_").replace("\\", "_").replace(" ", "_").replace('"', '').replace("'", "")
        filename = os.path.join(output_dir, f"{safe_name}_{row['pk_unique_id']}.md")
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# Medical Report: {row['name']}\n\n")
            
            if pd.notna(row['source_url']):
                f.write(f"Source: {row['source_url']}\n\n")
                
            f.write("## Overview / Description\n")
            if pd.notna(row['description']):
                f.write(f"{row['description']}\n\n")
            else:
                f.write("No description provided.\n\n")
                
            f.write("## Capabilities noted in raw data:\n")
            if pd.notna(row['capability']):
                f.write(f"{row['capability']}\n\n")
            
            f.write("## Equipment noted in raw data:\n")
            if pd.notna(row['equipment']):
                f.write(f"{row['equipment']}\n\n")
                
            f.write("## Procedures noted in raw data:\n")
            if pd.notna(row['procedure']):
                f.write(f"{row['procedure']}\n\n")

            f.write("## Mission Statement:\n")
            if pd.notna(row['missionStatement']):
                f.write(f"{row['missionStatement']}\n\n")
                
            f.write("---\nEND OF REPORT\n")
            
    print(f"Created {len(samples)} sample files in {output_dir}")

if __name__ == "__main__":
    csv_file = os.path.join(os.path.dirname(__file__), "..", "data", "Virtue Foundation Ghana v0.3 - Sheet1.csv")
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "samples")
    create_sample_dataset(csv_file, out_dir)
