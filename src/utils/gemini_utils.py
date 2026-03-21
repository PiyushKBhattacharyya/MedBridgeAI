import os
import time
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv(override=True)

class GeminiRotator:
    def __init__(self):
        # Support both GEMINI_API_KEY (single) and GEMINI_API_KEYS (comma separated)
        keys_str = os.getenv("GEMINI_API_KEYS", os.getenv("GEMINI_API_KEY", ""))
        self.keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        self.current_index = 0
        
        if not self.keys:
            print("WARNING: No Gemini API keys found in environment variables.")
        else:
            print(f"GeminiRotator: Successfully loaded {len(self.keys)} API keys.")

    def get_random_key(self):
        if not self.keys: return None
        import random
        return random.choice(self.keys)

    def rotate_key(self):
        # Reload keys in case .env was updated
        load_dotenv(override=True)
        keys_str = os.getenv("GEMINI_API_KEYS", os.getenv("GEMINI_API_KEY", ""))
        self.keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        if not self.keys: return None
        
        # Add a small delay to help with RPM limits
        print(f"GeminiRotator: 429 detected. Sleeping for 2s before rotating...")
        time.sleep(2)
        
        # Advance index
        self.current_index = (self.current_index + 1) % len(self.keys)
        key = self.keys[self.current_index]
        print(f"GeminiRotator: Switched to API Key {self.current_index + 1}/{len(self.keys)} (Key: {key[:8]}...)")
        
        genai.configure(api_key=key)
        return key

    def configure_genai(self):
        """Initial configuration with a random key."""
        key = self.get_random_key()
        if key:
            genai.configure(api_key=key)
        return key

# Global instance
rotator = GeminiRotator()
