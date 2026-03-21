import os
import sys
import warnings
import google.generativeai as genai
import pandas as pd
from typing import List, Optional
from dotenv import load_dotenv

# Add project folder to sys.path so 'src' can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.synthesis.database import MedBridgeStore

from src.utils.gemini_utils import rotator

load_dotenv()

class SynthesisAgent:
    def __init__(self, store: MedBridgeStore, model_id: str = "gemini-2.0-flash"):
        """
        Initializes the Synthesis Assistant using the Gemini API.
        """
        self.store = store
        self.model_id = model_id
        rotator.configure_genai()
        self.model = genai.GenerativeModel(model_id)
        print(f"SynthesisAgent initialized with Gemini ({model_id}) and multiple keys.")
        
        self.system_prompt = "You are the MedBridge AI Medical Assistant. Your role is to answer questions about healthcare facilities and NGOs in Ghana based on the provided search results from our database. Use ONLY the information provided. Be concise and professional."

    def answer_question(self, query: str) -> str:
        """
        Retrieves relevant data from the store and synthesizes an answer with citations using Gemini.
        """
        # Step 1: Search facilities and NGOs
        facilities_df = self.store.search_facilities(query, limit=5)
        ngos_df = self.store.search_ngos(query, limit=3)
        
        # Combine results into context
        context = "--- FACILITIES ---\n"
        if not facilities_df.empty:
            context += facilities_df.drop(columns=['vector', 'latitude', 'longitude'], errors='ignore').to_string(index=False)
        else:
            context += "No matching facilities found."
            
        context += "\n\n--- NGOs ---\n"
        if not ngos_df.empty:
            context += ngos_df.drop(columns=['vector', 'latitude', 'longitude'], errors='ignore').to_string(index=False)
        else:
            context += "No matching NGOs found."

        # Step 2: Generate synthesized answer via Gemini with key rotation
        prompt = f"""System: {self.system_prompt}
 
 SEARCH RESULTS CONTEXT:
 {context}
 
 User Question: {query}
 Assistant:"""
        
        # Try multiple keys if 429 occurs
        for _ in range(max(1, len(rotator.keys))):
            try:
                response = self.model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                err_msg = str(e).lower()
                if "429" in err_msg or "quota" in err_msg:
                    print(f"Synthesis Key Rotated due to 429. Retrying...")
                    rotator.rotate_key()
                    self.model = genai.GenerativeModel(self.model_id) # Re-instantiate model with new key config
                    continue
                print(f"Gemini Synthesis Error: {e}")
                return f"I encountered an error while synthesizing an answer: {e}"
                
        return "I exhausted all available Gemini API keys and still hit quota limits. Please try again later or add more API keys."
