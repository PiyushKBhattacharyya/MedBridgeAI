import os
import sys
import google.generativeai as genai
from dotenv import load_dotenv

# Add project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.gemini_utils import rotator
from src.synthesis.database import MedBridgeStore
from src.synthesis.agent import SynthesisAgent

def test_rotation():
    load_dotenv(override=True)
    print(f"Keys found in .env: {len(rotator.keys)}")
    
    store = MedBridgeStore()
    agent = SynthesisAgent(store=store)
    
    query = "Who has MRI?"
    print(f"Testing query: {query}")
    
    try:
        response = agent.answer_question(query)
        print("\n--- RESPONSE ---")
        print(response)
        print("----------------")
        if "I exhausted all available Gemini API keys" in response:
            print("\nRESULT: FAILED (All keys exhausted)")
        elif "error while synthesizing" in response:
            print("\nRESULT: FAILED (Other error)")
        else:
            print("\nRESULT: SUCCESS!")
    except Exception as e:
        print(f"Unexpected Error: {e}")

if __name__ == "__main__":
    test_rotation()
