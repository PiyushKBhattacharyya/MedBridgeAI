from src.extraction.agent import IDPExtractor
from src.extraction.loader import load_text_documents
from src.synthesis.database import MedBridgeStore
from src.synthesis.agent import SynthesisAgent
import os
import shutil

def main():
    # 1. Setup
    print("--- Phase 3 Verification ---")
    db_path = "data/test_medbridge.lancedb"
    if os.path.exists(db_path):
        shutil.rmtree(db_path) # Clean start for verification
        
    store = MedBridgeStore(db_path=db_path)
    extractor = IDPExtractor()
    agent = SynthesisAgent(store)

    # 2. Pick a sample file
    sample_dir = "data/samples"
    if not os.path.exists(sample_dir):
        print(f"Error: {sample_dir} not found. Run scripts/extract_samples.py first.")
        return
        
    sample_files = [f for f in os.listdir(sample_dir) if f.endswith(".md")]
    if not sample_files:
        print(f"Error: No samples found in {sample_dir}.")
        return
        
    sample_path = os.path.join(sample_dir, sample_files[0])
    print(f"Processing sample: {sample_path}")
    
    # 3. Extract
    docs = load_text_documents(sample_path)
    if not docs:
        print("Error: Could not load document.")
        return
        
    try:
        extraction = extractor.extract_from_text(docs[0].page_content)
        if extraction is None:
            print("Error: Extraction returned None (likely a parsing failure).")
            return
    except Exception as e:
        print(f"Extraction failed with error: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"Extracted {len(extraction.facilities)} facilities and {len(extraction.ngos)} NGOs.")

    # 4. Store
    print("Storing extractions in LanceDB...")
    store.add_extractions(extraction)

    # 5. Query
    query = "Find healthcare facilities and describe their capabilities."
    print(f"\nAsking Agent: '{query}'")
    answer = agent.answer_question(query)
    
    print("\n--- AGENT ANSWER ---")
    print(answer)
    print("--------------------")

if __name__ == "__main__":
    main()
