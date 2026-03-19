from typing import List, Optional
import warnings
import torch
from transformers import AutoTokenizer, logging
from optimum.pipelines import pipeline
from optimum.onnxruntime import ORTModelForCausalLM
# Removed langchain imports to speed up local inference
from src.synthesis.database import MedBridgeStore
import os
import onnxruntime as ort
import pandas as pd

# Suppress noisy hardware warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

def get_best_provider():
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return "CUDAExecutionProvider"
    if "DmlExecutionProvider" in available:
        return "DmlExecutionProvider"
    return "CPUExecutionProvider"

class SynthesisAgent:
    def __init__(self, store: MedBridgeStore, model_id: str = "Qwen/Qwen2.5-0.5B-Instruct", model=None, tokenizer=None):
        """
        Initializes the Synthesis Assistant. Can accept an existing model/tokenizer to save VRAM.
        """
        self.store = store
        
        if model and tokenizer:
            self.model = model
            self.tokenizer = tokenizer
            print("SynthesisAgent: Using shared model instance.")
        else:
            # Localize model check
            local_model_path = os.path.join("models", "qwen2.5-0.5b-onnx")
            export_needed = True
            if os.path.exists(local_model_path):
                model_id = local_model_path
                export_needed = False
                print(f"Using localized model: {model_id}")

            # Auto-detect best provider
            provider = get_best_provider()
            print(f"Loading local SLM for synthesis on {provider}: {model_id}...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = ORTModelForCausalLM.from_pretrained(
                model_id,
                export=export_needed,
                provider=provider
            )
            print(f"SynthesisAgent Active Backend: {provider}")
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            repetition_penalty=1.1,
            return_full_text=False,
            device="cpu"
        )
        print("Synthesis engine running via DirectML for AMD GPU")
        
        self.system_prompt = "You are the MedBridge AI Synthesis Assistant. Your role is to answer questions about healthcare facilities and NGOs based on the provided search results from our database. Use ONLY the information provided. Be concise."
        
        # Using raw pipeline for faster local inference

    def answer_question(self, query: str):
        """
        Retrieves relevant data from the store and synthesizes an answer.
        """
        # Step 1: Search facilities
        facilities_df = self.store.search_facilities(query, limit=5)
        
        # Step 2: Search NGOs
        ngos_df = self.store.search_ngos(query, limit=3)
        
        # Combine results for prompt
        results_text = "--- FACILITIES ---\n"
        if not facilities_df.empty:
            results_text += facilities_df.drop(columns=['vector']).to_string()
        else:
            results_text += "No matching facilities found."
            
        results_text += "\n\n--- NGOs ---\n"
        if not ngos_df.empty:
            results_text += ngos_df.drop(columns=['vector']).to_string()
        else:
            results_text += "No matching NGOs found."
            
        # Step 3: Generate synthesized answer
        prompt = f"<|im_start|>system\n{self.system_prompt}\n\nSEARCH RESULTS:\n{results_text}<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
        
        response = self.pipe(prompt)[0]['generated_text']
        return response
