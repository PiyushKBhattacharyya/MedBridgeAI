import os
import sys

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.synthesis.database import MedBridgeStore

def reset():
    store = MedBridgeStore()
    store.clear_database()
    print("Database tables dropped. Ready for schema-enforced reprocessing.")

if __name__ == "__main__":
    reset()
