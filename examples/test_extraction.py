import os
import json
from src.extraction.agent import IDPExtractor
from src.extraction.loader import load_text_documents

def run_test():
    # Initialize the extractor
    extractor = IDPExtractor(model_name="gemini-1.5-flash", temperature=0.0)
    
    # Path to our sample data
    samples_dir = os.path.join(os.path.dirname(__file__), "..", "data", "samples")
    
    if not os.path.exists(samples_dir):
        print(f"Samples directory {samples_dir} not found. Please run scripts/extract_samples.py first.")
        return
        
    documents = load_text_documents(samples_dir)
    print(f"Loaded {len(documents)} sample documents.")
    
    # Just process the first 2 documents to save time/tokens during a test run
    test_docs = documents[:2]
    
    for idx, doc in enumerate(test_docs):
        print(f"\\n--- Extracting Document {idx+1}/{len(test_docs)} ---")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\\n")
        
        # Run extraction
        try:
            extraction_result = extractor.extract_from_text(doc.page_content)
            
            # Print the extracted JSON
            print(extraction_result.model_dump_json(indent=2, exclude_none=True))
        except Exception as e:
            print(f"Extraction failed: {str(e)}")

if __name__ == "__main__":
    run_test()
