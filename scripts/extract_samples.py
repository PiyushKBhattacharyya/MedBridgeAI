import pandas as pd
import os
import shutil

def create_sample_dataset(csv_path: str, output_dir: str):
    """
    Reads the real CSV dataset and creates HIGH-FIDELITY markdown files for all 987 rows.
    Uses explicit [FIELD_...] tags for 100% retrieval parity.
    """
    df = pd.read_csv(csv_path)
    
    # Clear existing samples to prevent stale/duplicate data
    if os.path.exists(output_dir):
        print(f"Clearing old samples in {output_dir}...")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Exporting {len(df)} reports to high-fidelity markdown...")
    
    for idx, row in df.iterrows():
        # Use INDEX for absolute 987-record uniqueness
        safe_name = str(row['name']).replace("/", "_").replace("\\", "_").replace(" ", "_").replace('"', '').replace("'", "")
        filename = os.path.join(output_dir, f"{safe_name}_{idx}.md")
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# Medical Report: {row['name']}\n\n")
            
            # Explicit Ingestion Tags for 1:1 Parity
            f.write(f"[ORG_TYPE]: {row['organization_type']}\n")
            f.write(f"[ROW_IDX]: {idx}\n")
            f.write(f"[SOURCE_URL]: {row['source_url'] if pd.notna(row['source_url']) else ''}\n")
            
            f.write("\n## Contact Data (Direct Mapping)\n")
            f.write(f"[FIELD_PHONE]: {row['phone_numbers'] if pd.notna(row['phone_numbers']) else ''}\n")
            f.write(f"[FIELD_EMAIL]: {row['email'] if pd.notna(row['email']) else ''}\n")
            f.write(f"[FIELD_WEB]: {row['websites'] if pd.notna(row['websites']) else ''}\n")
            
            f.write("\n## Metadata\n")
            f.write(f"[FIELD_YEAR]: {row['yearEstablished'] if pd.notna(row['yearEstablished']) else ''}\n")
            f.write(f"[FIELD_DOCS]: {row['numberDoctors'] if pd.notna(row['numberDoctors']) else ''}\n")
            f.write(f"[FIELD_CAPACITY]: {row['capacity'] if pd.notna(row['capacity']) else ''}\n")
            f.write(f"[FIELD_AREA]: {row['area'] if pd.notna(row['area']) else ''}\n")
            f.write(f"[FIELD_VOLUNTEERS]: {row['acceptsVolunteers'] if pd.notna(row['acceptsVolunteers']) else ''}\n")
            f.write(f"[FIELD_FB]: {row['facebookLink'] if pd.notna(row['facebookLink']) else ''}\n")
            f.write(f"[FIELD_TW]: {row['twitterLink'] if pd.notna(row['twitterLink']) else ''}\n")
            f.write(f"[FIELD_LI]: {row['linkedinLink'] if pd.notna(row['linkedinLink']) else ''}\n")
            f.write(f"[FIELD_IG]: {row['instagramLink'] if pd.notna(row['instagramLink']) else ''}\n")
            f.write(f"[FIELD_LOGO]: {row['logo'] if pd.notna(row['logo']) else ''}\n")
            
            f.write("\n## Location\n")
            f.write(f"[FIELD_ADDR1]: {row['address_line1'] if pd.notna(row['address_line1']) else ''}\n")
            f.write(f"[FIELD_CITY]: {row['address_city'] if pd.notna(row['address_city']) else ''}\n")
            f.write(f"[FIELD_REGION]: {row['address_stateOrRegion'] if pd.notna(row['address_stateOrRegion']) else ''}\n")
            
            f.write("\n## Text Content\n")
            if pd.notna(row['description']): f.write(f"Description: {row['description']}\n\n")
            if pd.notna(row['capability']): f.write(f"Capabilities: {row['capability']}\n\n")
            if pd.notna(row['equipment']): f.write(f"Equipment: {row['equipment']}\n\n")
            if pd.notna(row['procedure']): f.write(f"Procedures: {row['procedure']}\n\n")
            if pd.notna(row['missionStatement']): f.write(f"Mission: {row['missionStatement']}\n\n")
                
            f.write("---\nEND OF REPORT\n")
            
    print(f"Total: Exported {len(df)} reports.")

if __name__ == "__main__":
    csv_file = os.path.join(os.path.dirname(__file__), "..", "data", "Virtue Foundation Ghana v0.3 - Sheet1.csv")
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "samples")
    create_sample_dataset(csv_file, out_dir)
