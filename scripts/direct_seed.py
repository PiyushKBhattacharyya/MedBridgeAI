import os
import pandas as pd
import shutil
import ast
from src.synthesis.database import MedBridgeStore
from src.schema.models import Facility, NGO

def clean_val(v):
    if pd.isna(v) or str(v).lower() in ["nan", "none", "", "[]", "[,]"]:
        return None
    s = str(v).strip()
    if s in ["[]", "[,]"]: return None
    return s

def parse_list(v):
    if not v or pd.isna(v): return []
    if isinstance(v, list): return [i for i in v if i]
    s = str(v).strip()
    if s == "[]" or s == "[,]" or not s: return []
    
    if s.startswith("[") and s.endswith("]"):
        try: 
            import ast
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(i).strip() for i in parsed if i]
        except: pass
    return [i.strip() for i in s.split(",") if i.strip()]

def merge_models(target, source_data: dict, model_type):
    """
    Smart merges source_data into target model.
    Fills None fields in target with data from source.
    Merges and deduplicates lists.
    """
    for field, value in source_data.items():
        current_val = getattr(target, field, None)
        
        # Merge Lists
        if isinstance(current_val, list) and isinstance(value, list):
            combined = list(dict.fromkeys(current_val + value)) # deduplicate while preserving order
            setattr(target, field, combined)
        
        # Fill None/Empty fields
        elif (current_val is None or current_val == "") and value is not None:
            setattr(target, field, value)

def direct_seed():
    print("Starting SMART High-Fidelity Seeding...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "data", "Virtue Foundation Ghana v0.3 - Sheet1.csv")
    df = pd.read_csv(csv_path)
    
    store = MedBridgeStore()
    
    # Store keys to models
    facility_map = {} # key -> Facility
    ngo_map = {} # key -> NGO
    
    def get_clean_k(s):
        if not s: return ""
        return "".join(e for e in str(s) if e.isalnum()).lower()

    print(f"Processing {len(df)} rows from CSV with Smart Merge...")
    
    for idx, row in df.iterrows():
        city = clean_val(row.get('address_city', ''))
        region = clean_val(row.get('address_stateOrRegion', ''))
        lat, lon = None, None
        
        if city:
            c_lower = city.lower()
            if "accra" in c_lower: lat, lon = 5.6037, -0.1870
            elif "kumasi" in c_lower: lat, lon = 6.6666, -1.6163
            elif "tamale" in c_lower: lat, lon = 9.4072, -0.8533

        org_type = str(row['organization_type']).lower()
        name = str(row['name']).strip()
        address1 = clean_val(row.get('address_line1', ''))
        
        name_k = get_clean_k(name)
        city_k = get_clean_k(city)
        addr_k = get_clean_k(address1)
        key = (name_k, addr_k, city_k or "unknown")

        if org_type == "ngo":
            data = {
                "name": name,
                "organizationDescription": clean_val(row.get('description')),
                "missionStatement": clean_val(row.get('missionStatement')),
                "countries": ["Ghana"],
                "phone_numbers": parse_list(row.get('phone_numbers')),
                "email": clean_val(row.get('email')),
                "websites": parse_list(row.get('websites')),
                "facebookLink": clean_val(row.get('facebookLink')),
                "twitterLink": clean_val(row.get('twitterLink')),
                "linkedinLink": clean_val(row.get('linkedinLink')),
                "instagramLink": clean_val(row.get('instagramLink')),
                "logo": clean_val(row.get('logo')),
                "address_line1": address1,
                "address_city": city or "Unknown",
                "address_stateOrRegion": region or "Ghana",
                "address_country": "Ghana",
                "address_countryCode": "GH",
                "latitude": lat,
                "longitude": lon,
                "yearEstablished": int(float(row['yearEstablished'])) if pd.notna(row['yearEstablished']) else None
            }
            if key in ngo_map:
                merge_models(ngo_map[key], data, NGO)
            else:
                ngo_map[key] = NGO(**data)
        else:
            data = {
                "name": name,
                "description": clean_val(row.get('description')),
                "capability": parse_list(row.get('capability')),
                "equipment": parse_list(row.get('equipment')),
                "procedure": parse_list(row.get('procedure')),
                "latitude": lat,
                "longitude": lon,
                "address_line1": address1,
                "address_city": city or "Unknown",
                "address_stateOrRegion": region or "Ghana",
                "address_country": "Ghana",
                "address_countryCode": "GH",
                "phone_numbers": parse_list(row.get('phone_numbers')),
                "email": clean_val(row.get('email')),
                "websites": parse_list(row.get('websites')),
                "yearEstablished": int(float(row['yearEstablished'])) if pd.notna(row['yearEstablished']) else None,
                "numberDoctors": clean_val(row.get('numberDoctors')),
                "capacity": clean_val(row.get('capacity')),
                "area": clean_val(row.get('area')),
                "facebookLink": clean_val(row.get('facebookLink')),
                "twitterLink": clean_val(row.get('twitterLink')),
                "linkedinLink": clean_val(row.get('linkedinLink')),
                "instagramLink": clean_val(row.get('instagramLink')),
                "logo": clean_val(row.get('logo')),
                "acceptsVolunteers": clean_val(row.get('acceptsVolunteers'))
            }
            if key in facility_map:
                merge_models(facility_map[key], data, Facility)
            else:
                facility_map[key] = Facility(**data)

    facilities = list(facility_map.values())
    ngos = list(ngo_map.values())

    # Bulk Create
    if facilities:
        print(f"Seeding {len(facilities)} smart-merged facilities...")
        data = store._prepare_facility_data(facilities, "CSV Smart Merge v1")
        store.db.create_table(store.facility_table_name, data=data, mode="overwrite")
    
    if ngos:
        print(f"Seeding {len(ngos)} smart-merged NGOs...")
        data = store._prepare_ngo_data(ngos, "CSV Smart Merge v1")
        store.db.create_table(store.ngo_table_name, data=data, mode="overwrite")
        
    print(f"\nSUCCESS: Seeded {len(facilities)} Facilities and {len(ngos)} NGOs. Total {len(facilities)+len(ngos)} unique entities.")

if __name__ == "__main__":
    direct_seed()


if __name__ == "__main__":
    direct_seed()
