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
Your goal is to parse unstructured medical notes, surveys, and facility reports.
You must extract all relevant NGOs, healthcare facilities, and facts specifically as requested by the provided schema.

Key extraction instructions:
- 'procedure', 'equipment', and 'capability' facts must be written as clear, declarative sentences (e.g., "Hospital offers hemodialysis treatment 3 times weekly.", "Facility has a Siemens SOMATOM Force dual-source CT scanner."). Do not extract single words or nouns.
- Use exact phrase matches when mapping to Medical Specialties (consult the schema definition).
- Use exact phrasing when mapping to Enums (e.g., 'hospital', 'public', 'faith-tradition').
- Phone numbers MUST be formatted exactly in E164 format (e.g., '+233392022664') regardless of how they appear in the raw text. Remove spaces, dashes, or local prefixes.
- If a country isn't explicitly stated but can be inferred from context (like the city or domain name), provide its full name ('address_country') and the 2-letter ISO code ('address_countryCode').
- If a value safely cannot be determined from the text, omit it (leave as null) rather than guessing. Do not hallucinate data.

Analyze the following document and output the structured JSON targeting the Schema."""),
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
