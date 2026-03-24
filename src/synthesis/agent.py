import os
import sys
import time
import warnings
import google.generativeai as genai
import pandas as pd
from typing import List, Optional
from dotenv import load_dotenv

# Add project folder to sys.path so 'src' can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.synthesis.database import MedBridgeStore

from src.utils.gemini_utils import rotator
from src.utils.fallback_llm import fallback_llm

load_dotenv()

class SynthesisAgent:
    def __init__(self, store: MedBridgeStore, model_id: str = "gemini-2.5-flash"):
        """
        Initializes the synthesis agent using Gemini.
        """
        self.store = store
        self.model_id = model_id
        
        # Configure the initial Gemini key
        rotator.configure_genai()
        self.model = genai.GenerativeModel(self.model_id)
        
        print(f"SynthesisAgent initialized with Gemini ({model_id}) and multiple keys.")

    def answer_question(self, query: str) -> str:
        """
        Answers a user question based on the retrieved data from LanceDB.
        """
        # Search the database for relevant context
        facilities_df = self.store.search_facilities(query, limit=5)
        ngos_df = self.store.search_ngos(query, limit=5)
        
        # Build context string
        context = "Relevant Facilities:\n"
        if not facilities_df.empty:
            for _, row in facilities_df.iterrows():
                context += f"- {row['name']}: {row.get('description', 'No description')} (Capibilities: {row.get('capability', 'None')})\n"
        else:
            context += "No relevant facilities found.\n"
            
        context += "\nRelevant NGOs:\n"
        if not ngos_df.empty:
            for _, row in ngos_df.iterrows():
                context += f"- {row['name']}: {row.get('organizationDescription', 'No description')} (Countries: {row.get('countries', 'None')})\n"
        else:
            context += "No relevant NGOs found.\n"
            
        prompt = f"Context from database:\n{context}\n\nQuestion: {query}\n\nAnswer based ONLY on the context above. If unsure, say you don't know."
        
        # Try multiple keys if 429 occurs
        for _ in range(max(1, len(rotator.keys))):
            try:
                response = self.model.generate_content(prompt)
                rotator.reset_backoff()
                time.sleep(0.5) # Gentle pacing
                return response.text.strip()
            except Exception as e:
                err_msg = str(e).lower()
                if "429" in err_msg or "quota" in err_msg:
                    print(f"Synthesis Key Rotated due to 429. Retrying...")
                    rotator.rotate_key()
                    self.model = genai.GenerativeModel(self.model_id) # Re-instantiate model with new key config
                    continue
                
                print(f"Gemini Synthesis failed: {e}. Attempting local fallback...")
                break
        
        # FINAL FALLBACK to local LLM
        print("Using Local LLM Fallback for Synthesis...")
        try:
            return fallback_llm.generate(prompt)
        except Exception as fe:
            return f"I encountered an error while synthesizing an answer: {fe}"
