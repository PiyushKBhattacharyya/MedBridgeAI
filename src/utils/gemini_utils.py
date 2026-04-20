import os
import time
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv(override=True)

def _load_from_streamlit_secrets():
    """
    On Streamlit Cloud there is no .env file — secrets live in st.secrets.
    This helper injects them into os.environ so the rest of the codebase
    (which reads os.environ / load_dotenv) works unchanged.
    Silently skips if streamlit is not available or secrets are not set.
    """
    try:
        import streamlit as st
        for key, value in st.secrets.items():
            if isinstance(value, str) and key not in os.environ:
                os.environ[key] = value
    except Exception:
        pass  # Running locally without st.secrets — .env handles it

_load_from_streamlit_secrets()

class GeminiRotator:
    def __init__(self):
        self._load_keys()
        self.current_index = 0
        self.backoff_time = 2.0 # Start with 2s backoff
        if not self.keys:
            print("WARNING: No Gemini API keys found in environment variables.")
        else:
            print(f"GeminiRotator: Successfully loaded {len(self.keys)} API keys.")

    def _load_keys(self):
        """Discovers all keys from environment variables (populated from .env or st.secrets)."""
        # 1. Start with dedicated comma-separated list
        keys_str = os.getenv("GEMINI_API_KEYS", "")
        self.keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        
        # 2. Add individual numbered keys (GEMINI_API_KEY_1 … GEMINI_API_KEY_N)
        for key, value in os.environ.items():
            if key.startswith("GEMINI_API_KEY") and key != "GEMINI_API_KEYS":
                if value and value.strip() not in self.keys:
                    self.keys.append(value.strip())

    def get_random_key(self):
        if not self.keys: return None
        import random
        return random.choice(self.keys)

    def rotate_key(self):
        # Refresh from environment
        load_dotenv(override=True)
        self._load_keys()
        
        if not self.keys: return None
        
        # Exponential backoff: Sleep longer if 429 persists
        print(f"GeminiRotator: 429 detected. Sleeping for {self.backoff_time:.1f}s before rotating...")
        time.sleep(self.backoff_time)
        
        # Increase backoff for next time, up to 10s
        self.backoff_time = min(self.backoff_time * 1.5, 10.0)
        
        # Advance index
        self.current_index = (self.current_index + 1) % len(self.keys)
        
        # If we just went through the WHOLE pool and still hitting 429, 
        # it's likely a shared project quota. Take a longer break.
        if self.current_index == 0:
            print(f"GeminiRotator: WHOLE POOL EXHAUSTED (all {len(self.keys)} keys rate-limited).")
            print(f"GeminiRotator: Entering deep sleep for 20s to reset global quota...")
            time.sleep(18) # Total ~20s with the previous sleep
            self.backoff_time = 2.0 # Reset backoff for a fresh start

        key = self.keys[self.current_index]
        print(f"GeminiRotator: Trying API Key {self.current_index + 1}/{len(self.keys)} (Key: {key[:8]}...)")
        
        genai.configure(api_key=key)
        return key

    def reset_backoff(self):
        """Call this after a successful API call to reset the backoff timer."""
        self.backoff_time = 2.0

    def configure_genai(self):
        """Initial configuration with a random key."""
        key = self.get_random_key()
        if key:
            genai.configure(api_key=key)
        return key

# Global instance
rotator = GeminiRotator()
