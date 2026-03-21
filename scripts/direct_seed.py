import os
import sys
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import json
import time

# Fix: Add project folder to sys.path so 'src' can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.synthesis.database import MedBridgeStore
from src.schema.models import Facility, NGO

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def clean_val(v):
    if pd.isna(v) or str(v).lower() in ["nan", "none", "unknown", "", "[]", "[,]"]:
        return None
    s = str(v).strip()
    if s.replace(",", "").strip() == "" or s in ["[]", "[,]"]: return None
    return s

def parse_list(v):
    if not v or pd.isna(v): return []
    if isinstance(v, list): return [i for i in v if i]
    s = str(v).strip()
    if s.replace(",", "").strip() == "" or s in ["[]", "[,]"]: return []
    if s.startswith("[") and s.endswith("]"):
        try: 
            import ast
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(i).strip() for i in parsed if i and str(i).strip()]
        except: pass
    return [i.strip() for i in s.split(",") if i.strip() and i.strip().lower() not in ["", "nan", "none"]]

def safe_float(v):
    if v is None or v == "" or pd.isna(v): return None
    try: return float(v)
    except: return None

def safe_int(v):
    if v is None or v == "" or pd.isna(v): return None
    try: return int(float(v))
    except: return None

def merge_models(target, source_data: dict, model_type):
    allowed = set(model_type.model_fields.keys())
    for field, value in source_data.items():
        if field not in allowed: continue
        try:
            current_val = getattr(target, field, None)
            if isinstance(current_val, list) and isinstance(value, list):
                combined = list(dict.fromkeys(current_val + value))
                setattr(target, field, combined)
            elif (current_val is None or current_val == "") and value is not None:
                setattr(target, field, value)
        except: pass

from src.utils.gemini_utils import rotator

class ExtremeMatcher:
    def __init__(self):
        self.model_id = "gemini-2.0-flash"
        rotator.configure_genai()
        self.model = genai.GenerativeModel(self.model_id)


    def cluster_all(self, all_candidates: list) -> dict:
        """Process ALL 80 groups in ONE request to save quota."""
        prompt = """Task: Group medical facility records into physical entities.
For each ENTITY_NAME group, cluster the IDs that belong to the SAME building or branch.
Note: Accra suburbs (Dansoman, Kasoa, East Legon) are identical to 'Accra'.

Groups:
"""
        for g in all_candidates:
            prompt += f"\n- {g['name_k']}:\n"
            for r in g['rows']:
                prompt += f"  ID {r['id']}: {r['name']} at {r['addr']}, {r['city']}\n"
        
        prompt += "\nOutput ONLY a JSON dict: {'name_k': [[id1, id2], [id3]], ...}"
        
        # Try multiple keys if 429 occurs
        for _ in range(max(1, len(rotator.keys))):
            try:
                response = self.model.generate_content(prompt)
                text = response.text
                if "```json" in text: text = text.split("```json")[1].split("```")[0]
                elif "{" in text: text = text[text.find("{"):text.rfind("}")+1]
                return json.loads(text)
            except Exception as e:
                err_msg = str(e).lower()
                if "429" in err_msg or "quota" in err_msg:
                    print(f"ExtremeMatcher Key Rotated due to 429. Retrying...")
                    rotator.rotate_key()
                    self.model = genai.GenerativeModel(self.model_id)
                    continue
                print(f"Gemini Extreme Error: {e}")
                return {g['name_k']: [[r['id'] for r in g['rows']]] for g in all_candidates}
        return {g['name_k']: [[r['id'] for r in g['rows']]] for g in all_candidates}

def direct_seed():
    print("Starting GEMINI EXTREME-BATCH Seeding (v8)...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "data", "Virtue Foundation Ghana v0.3 - Sheet1.csv")
    
    df = pd.read_csv(csv_path)
    store = MedBridgeStore()
    df['name_k'] = df['name'].apply(lambda x: "".join(e for e in str(x) if e.isalnum()).lower())
    
    candidate_groups = []
    final_clusters = {}
    
    for name_k, group in df.groupby('name_k'):
        if len(group) > 1:
            rows = []
            for idx, row in group.iterrows():
                rows.append({"id": int(idx), "name": str(row['name']), "addr": clean_val(row.get('address_line1', '')), "city": clean_val(row.get('address_city', ''))})
            candidate_groups.append({"name_k": name_k, "rows": rows})
        else:
            final_clusters[name_k] = [[int(group.index[0])]]

    if candidate_groups:
        print(f"Sending {len(candidate_groups)} groups to Gemini in ONE request...")
        matcher = ExtremeMatcher()
        results = matcher.cluster_all(candidate_groups)
        final_clusters.update(results)

    unified_facilities = []
    unified_ngos = []
    for name_k, clusters in final_clusters.items():
        for cluster in clusters:
            primary_obj = None
            is_ngo = False
            for row_idx in cluster:
                row = df.loc[row_idx]
                is_ngo = (str(row['organization_type']).lower() == "ngo")
                data = {
                    "name": str(row['name']).strip(), "description": clean_val(row.get('description')),
                    "organizationDescription": clean_val(row.get('description')), "missionStatement": clean_val(row.get('missionStatement')),
                    "phone_numbers": parse_list(row.get('phone_numbers')), "email": clean_val(row.get('email')),
                    "websites": parse_list(row.get('websites')), "logo": clean_val(row.get('logo')),
                    "address_line1": clean_val(row.get('address_line1', '')), "address_city": clean_val(row.get('address_city', '')) or "Unknown",
                    "latitude": safe_float(row.get('latitude')), "longitude": safe_float(row.get('longitude')),
                    "capability": parse_list(row.get('capability')), "equipment": parse_list(row.get('equipment')),
                    "procedure": parse_list(row.get('procedure')), "yearEstablished": safe_int(row.get('yearEstablished')),
                    "numberDoctors": clean_val(row.get('numberDoctors')), "capacity": safe_int(row.get('capacity')),
                    "area": safe_int(row.get('area')), "acceptsVolunteers": clean_val(row.get('acceptsVolunteers')),
                    "facebookLink": clean_val(row.get('facebookLink')), "twitterLink": clean_val(row.get('twitterLink')),
                    "linkedinLink": clean_val(row.get('linkedinLink')), "instagramLink": clean_val(row.get('instagramLink')),
                    "address_stateOrRegion": clean_val(row.get('address_stateOrRegion', '')) or "Ghana"
                }
                model_cls = NGO if is_ngo else Facility
                allowed = set(model_cls.model_fields.keys())
                clean_data = {k: v for k, v in data.items() if k in allowed}
                if primary_obj is None:
                    primary_obj = model_cls(**clean_data)
                else:
                    merge_models(primary_obj, clean_data, type(primary_obj))
            if isinstance(primary_obj, NGO): unified_ngos.append(primary_obj)
            else: unified_facilities.append(primary_obj)

    print(f"\nSUCCESS: Unified 987 rows into {len(unified_facilities)} Facilities and {len(unified_ngos)} NGOs.")
    store.clear_database()
    store.add_facilities(unified_facilities)
    store.add_ngos(unified_ngos)

if __name__ == "__main__":
    direct_seed()
