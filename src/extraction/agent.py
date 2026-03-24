import os
import sys
import warnings
import json
import time
import re
import google.generativeai as genai
from typing import List
from dotenv import load_dotenv

# Add project folder to sys.path so 'src' can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.schema.models import DocumentExtraction, Facility, NGO

from src.utils.gemini_utils import rotator
from src.utils.fallback_llm import fallback_llm

load_dotenv()

class IDPExtractor:
    def __init__(self, model_id: str = "gemini-2.5-flash", temperature: float = 0.1):
        """
        Initializes the IDP extractor using the Gemini API.
        """
        self.model_id = model_id
        self.temperature = temperature
        
        # Configure the initial Gemini key
        rotator.configure_genai()
        self.model = genai.GenerativeModel(self.model_id)
        
        print(f"IDPExtractor initialized with Gemini ({model_id}) and multiple keys.")
        
        # System instructions
        self.system_prompt = """You are an expert AI medical intelligence extraction system for the Virtue Foundation.
Your task is to analyze medical reports, Facebook snippets, and facility surveys and extract a JSON object.

THE DOCUMENTS OFTEN USE TAGS EXPLICITLY. MAP THEM AS FOLLOWS:
- [FIELD_PHONE] contents => 'phone_numbers' (list of strings)
- [FIELD_EMAIL] contents => 'email' (string)
- [FIELD_WEB] or [FIELD_WEB_OFFICIAL] => 'websites' (list of strings)
- [FIELD_ADDR1] contents => 'address_line1'
- [FIELD_CITY] contents => 'address_city'
- [FIELD_REGION] contents => 'address_stateOrRegion'
- [FIELD_YEAR] contents => 'yearEstablished'
- [FIELD_DOCS] contents => 'numberDoctors'
- [FIELD_CAPACITY] contents => 'capacity'
- [FIELD_VOLUNTEERS] contents => 'acceptsVolunteers' (boolean)
- [FIELD_FB], [FIELD_TW], [FIELD_LI], [FIELD_IG] => the respective social media link fields.

EXTRACTION RULES:
- NAME DISCOVERY: The organization name is CRITICAL and is usually in the document title or header.
- ENTITY CLASSIFICATION: 
    - [ORG_TYPE]: facility -> Use 'facilities' list.
    - [ORG_TYPE]: ngo -> Use 'ngos' list.
- BE COMPLETELY EXHAUSTIVE: If 'Capabilities' or 'Text Content' sections mention equipment (MRI, CT, X-Ray), procedures, or specialties, extract them into 'equipment', 'procedure', and 'specialties' (list of strings).
- DO NOT RETURN NULL: Use empty lists [] or empty strings "" if a field is absolutely not found. Output ONLY a valid JSON object.

Output ONLY this schema:
{
  "facilities": [{"name": "string", "description": "string", "capability": ["string"], "equipment": ["string"], "procedure": ["string"], "specialties": ["string"], "address_line1": "string", "address_city": "string", "phone_numbers": ["string"], "email": "string", "websites": ["string"]}],
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
                
                rotator.reset_backoff()
                time.sleep(0.5) # Gentle pacing
                
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
                    self.model = genai.GenerativeModel(self.model_id)
                    continue
                
                print(f"Gemini Extraction failed: {e}. Attempting local fallback...")
                break
        
        # FINAL FALLBACK to local LLM
        print("Using Local LLM Fallback for Extraction...")
        try:
            fallback_prompt = f"EXTRACT JSON for Facility/NGO from this text:\n{text}\n\nReturn ONLY raw JSON with keys 'facilities', 'ngos', 'other_organizations'."
            res_text = fallback_llm.generate(fallback_prompt)
            # Find JSON block
            match = re.search(r"\{.*\}", res_text, re.DOTALL)
            if match:
                data = json.loads(match.group())
                # Ensure keys exist
                if "facilities" not in data: data["facilities"] = []
                if "ngos" not in data: data["ngos"] = []
                if "other_organizations" not in data: data["other_organizations"] = []
                return DocumentExtraction(**data)
        except Exception as fe:
            print(f"Local fallback extraction failed: {fe}")

        return DocumentExtraction(facilities=[], ngos=[], other_organizations=[])

    def extract_from_documents(self, documents: List) -> List[DocumentExtraction]:
        """
        Helper method to extract from multiple LangChain Document objects.
        """
        results = []
        for doc in documents:
            res = self.extract_from_text(doc.page_content)
            results.append(res)
        return results
