import os
import sys
import warnings
import json
import re
import google.generativeai as genai
from typing import List
from dotenv import load_dotenv

# Add project folder to sys.path so 'src' can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.schema.models import DocumentExtraction, Facility, NGO

from src.utils.gemini_utils import rotator

load_dotenv()

class IDPExtractor:
    def __init__(self, model_id: str = "gemini-2.0-flash", temperature: float = 0.1):
        """
        Initializes the IDP extractor using the Gemini API.
        """
        self.model_id = model_id
        rotator.configure_genai()
        self.model = genai.GenerativeModel(model_id)
        self.temperature = temperature
        print(f"IDPExtractor initialized with Gemini ({model_id}) and multiple keys.")
        
        # System instructions
        self.system_prompt = """You are an expert AI medical intelligence extraction system for the Virtue Foundation.
Your task is to analyze medical reports, Facebook snippets, and facility surveys and extract a JSON object.

EXTRACTION RULES:
- NAME DISCOVERY: The organization name is CRITICAL and is usually in the document title or header (e.g., "# Medical Report: [NAME]"). If you see "3E Medical Center", that is the name.
- ENTITY CLASSIFICATION: 
    - [ORG_TYPE]: facility -> Use 'facilities' list.
    - [ORG_TYPE]: ngo -> Use 'ngos' list.
    - If unsure, use 'other_organizations'.
- BE COMPLETELY EXHAUSTIVE: If an organization is mentioned in the title or text, extract it. Do not return empty lists if there is a header title!
- SCHEMA CONFORMANCE: Output ONLY 'facilities', 'ngos', and 'other_organizations'.
- FORMATTING: Return ONLY a valid JSON object.

Output ONLY this schema:
{
  "facilities": [{"name": "string", "description": "string", "capability": ["string"], "address_line1": "string", "address_city": "string", "address_country": "string"}],
  "ngos": [{"name": "string", "organizationDescription": "string", "countries": ["string"]}],
  "other_organizations": [{"name": "string", "address_city": "string"}]
}
"""
        
    def extract_from_text(self, text: str) -> DocumentExtraction:
        """
        Runs the extraction via Gemini and parses the JSON output.
        """
        prompt = f"{self.system_prompt}\n\nDocument Text:\n{text}"
        
        # Try multiple keys if 429 occurs
        for _ in range(max(1, len(rotator.keys))):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=self.temperature,
                        response_mime_type="application/json"
                    )
                )
                
                data = json.loads(response.text)
                # Ensure facilities and ngos keys exist if not present in Gemini output
                if "facilities" not in data: data["facilities"] = []
                if "ngos" not in data: data["ngos"] = []
                if "other_organizations" not in data: data["other_organizations"] = []
                
                return DocumentExtraction(**data)
                
            except Exception as e:
                err_msg = str(e).lower()
                if "429" in err_msg or "quota" in err_msg:
                    print(f"Extraction Key Rotated due to 429. Retrying...")
                    rotator.rotate_key()
                    self.model = genai.GenerativeModel(self.model_id) # Re-instantiate model with new key config
                    continue
                print(f"Gemini Extraction Error: {e}")
                
                # Fallback to manual regex if JSON fails
                try:
                    match = re.search(r'\{.*\}', response.text, re.DOTALL)
                    if match:
                        return DocumentExtraction(**json.loads(match.group(0)))
                except: pass
                break
                
        return DocumentExtraction(facilities=[], ngos=[])

    def extract_from_documents(self, documents: List) -> List[DocumentExtraction]:
        """
        Helper method to extract from multiple LangChain Document objects.
        """
        results = []
        for doc in documents:
            res = self.extract_from_text(doc.page_content)
            results.append(res)
        return results
