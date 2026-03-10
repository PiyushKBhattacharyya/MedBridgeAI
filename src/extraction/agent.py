import os
from typing import List
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from src.schema.models import DocumentExtraction

# Load environment variables (e.g., GEMINI_API_KEY)
load_dotenv()

class IDPExtractor:
    def __init__(self, model_name: str = "gemini-2.5-flash", temperature: float = 0.0):
        """
        Initializes the IDP extractor using LangChain's structured output with Gemini.
        """
        # Ensure you have your GEMINI_API_KEY set in .env
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("API Key not found. Please set GEMINI_API_KEY or GOOGLE_API_KEY in .env")

        # Initialize Gemini Model
        self.llm = ChatGoogleGenerativeAI(
            model=model_name, 
            temperature=temperature,
            google_api_key=api_key
        )
        
        # We bind our Pydantic schema to the LLM to force structured JSON output
        self.structured_llm = self.llm.with_structured_output(DocumentExtraction)
        
        # Define the system prompt guiding the extraction logic
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert AI medical intelligence extraction system for the Virtue Foundation. 
Your task is to analyze medical notes, surveys, and facility reports, and extract structured facts about healthcare facilities and NGOs.

DEFINITIONS:
- NGOs: Any non-profit organization that delivers tangible, on-the-ground healthcare services in low/lower-middle-income settings. Include medical foundations, non-profit research institutes, and professional medical societies that provide direct patient care. Exclude advocacy-only, government agencies.
- Facilities: Any physical, currently operating site delivering in-person medical diagnosis or treatment (hospitals, clinics, centers). Exclude administrative offices or supply warehouses.

ORGANIZATION EXTRACTION RULES:
- Only extract organizations explicitly mentioned by NAME. Do NOT infer names.
- Always use the complete, unabbreviated form. Do not include suffixes like "Ltd" or "LLC".
- If multiple variations appear, extract the most complete version.

CONTACT & LOCATION RULES:
- Phone numbers MUST be in exactly E164 format (e.g., '+233392022664').
- Address: Address lines 1-3 are for STREET address only. City, State, Country go in their specific fields.
- Country extraction is MANDATORY. If a country can be inferred from context (like the city or URL domain), provide its full name and the 2-letter ISO code.

FACILITY FACTS RULES (procedure, equipment, capability):
- Use clear, declarative sentences (e.g., "Hospital offers hemodialysis treatment 3 times weekly.", "Facility has a Siemens SOMATOM Force dual-source CT scanner.").
- Do not extract single words or nouns. Include specific quantities or dates if available.

MEDICAL SPECIALTIES RULES:
- Extract all medical specialties, matching exactly to standard CamelCase forms (e.g., "internalMedicine", "familyMedicine", "dentistry", "emergencyMedicine", "generalSurgery", "pediatrics", "gynecologyAndObstetrics").
- Parse facility name (e.g. "Dental Clinic" -> "dentistry", "Eye Center" -> "ophthalmology").

CRITICAL REQUIREMENT:
- Be conservative. If a value safely cannot be determined from the text, omit it (leave as null). Do not hallucinate.

Analyze the following document and output the structured JSON targeting the exact Schema."""),
            ("user", "Document Text:\n{document_text}")
        ])
        
        self.extraction_chain = self.prompt | self.structured_llm

    def extract_from_text(self, text: str) -> DocumentExtraction:
        """
        Runs the extraction pipeline on a single document text.
        """
        return self.extraction_chain.invoke({"document_text": text})

    def extract_from_documents(self, documents: List) -> List[DocumentExtraction]:
        """
        Helper method to extract from multiple LangChain Document objects.
        """
        results = []
        for doc in documents:
            res = self.extract_from_text(doc.page_content)
            results.append(res)
        return results
