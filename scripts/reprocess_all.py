import os
import sys
import glob
import json
from typing import Any

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from json_repair import repair_json
from src.extraction.agent import IDPExtractor
from src.synthesis.database import MedBridgeStore
from src.utils.fallback_llm import fallback_llm
from src.schema.models import DocumentExtraction

def pre_clean_extracted_dict(data: Any):
    """
    ULTRA-ROBUST CLEANING: 
    1. Ensures top-level is a dict with 'facilities' and 'ngos'.
    2. Forces list-only fields to be lists of strings.
    3. Drops hallucinated nested dictionaries inside string lists.
    """
    if isinstance(data, list):
        # LLM returned a list directly? Try to extract the first dict.
        for item in data:
            if isinstance(item, dict):
                data = item
                break
    
    if not isinstance(data, dict):
        return {"facilities": [], "ngos": [], "other_organizations": []}

    list_fields = {
        "phone_numbers", "websites", "affiliationTypeIds", 
        "procedure", "equipment", "capability", "specialties", "countries"
    }
    
    for category in ["facilities", "ngos", "other_organizations"]:
        if category in data and isinstance(data[category], list):
            for org in data[category]:
                if not isinstance(org, dict): continue
                for k in list_fields:
                    if k in org:
                        val = org[k]
                        if not isinstance(val, list):
                            # Convert string/None/int to list
                            org[k] = [str(val)] if val else []
                        else:
                            # If it IS a list, ensure elements are strings, not dicts
                            clean_list = []
                            for item in val:
                                if isinstance(item, (str, int, float, bool)):
                                    clean_list.append(str(item))
                                # If it's a dict, the LLM hallucinated nesting. Drop it.
                            org[k] = clean_list
    return data

def reprocess_all():
    extractor = IDPExtractor()
    store = MedBridgeStore()
    
    # Resume logic: Check what's already in the database
    print("Checking database for existing records to resume...")
    processed_set = set()
    try:
        fac_df = store.get_all_facilities()
        if not fac_df.empty:
            processed_set.update(fac_df['source_doc'].unique().tolist())
        ngo_df = store.get_all_ngos()
        if not ngo_df.empty:
            processed_set.update(ngo_df['source_doc'].unique().tolist())
    except Exception as e:
        print(f"Could not load existing records: {e}")

    # Get all markdown files in data/samples
    sample_files = glob.glob("data/samples/*.md")
    print(f"Found {len(sample_files)} sample files. Skip {len(processed_set)} already processed.")
    
    total_added = 0
    batch_facilities = []
    batch_ngos = []
    BATCH_SIZE = 50
    
    for i, file_path in enumerate(sample_files):
        fname = os.path.basename(file_path)
        if fname in processed_set:
            continue
            
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        progress_pct = (i + 1) / len(sample_files) * 100
        print(f"\n>>> PROGRESS: {i+1}/{len(sample_files)} ({progress_pct:.1f}%) | Current: {fname}")
        try:
            # Use the high-fidelity Gemini extractor with 10-key rotation
            extraction_model = extractor.extract_from_text(text)
            
            if not extraction_model or (not extraction_model.facilities and not extraction_model.ngos):
                print(f"No insights extracted from {fname}")
                continue
            
            # Label with provenance and aggregate
            for fac in extraction_model.facilities:
                fac.source_doc = fname
                batch_facilities.append(fac)
            for ngo in extraction_model.ngos:
                ngo.source_doc = fname
                batch_ngos.append(ngo)
                
            total_added += 1
            processed_set.add(fname)
            
            # Flush batch to DB
            if (len(batch_facilities) >= BATCH_SIZE) or (len(batch_ngos) >= BATCH_SIZE) or (i == len(sample_files) - 1):
                if batch_facilities:
                    store.add_facilities(batch_facilities)
                    batch_facilities = []
                if batch_ngos:
                    store.add_ngos(batch_ngos)
                    batch_ngos = []
                print(f"--- Flushed accumulated batch to database! ---")
                
        except Exception as e:
            print(f"Error processing {fname}: {e}")
            
    print(f"Done! Extracted insights from {total_added} NEW files.")

if __name__ == "__main__":
    reprocess_all()
