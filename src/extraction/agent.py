import os
import warnings
import json
import re
import torch
from typing import List
from dotenv import load_dotenv
from transformers import AutoTokenizer, logging
from optimum.pipelines import pipeline
from optimum.onnxruntime import ORTModelForCausalLM
import onnxruntime as ort
from src.schema.models import DocumentExtraction, Facility, NGO

# Suppress non-critical hardware and tracing warnings globally
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["ORT_LOGGING_LEVEL"] = "3" 
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

def get_best_provider():
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return "CUDAExecutionProvider"
    if "DmlExecutionProvider" in available:
        return "DmlExecutionProvider"
    return "CPUExecutionProvider"

# Load environment variables
load_dotenv()


class IDPExtractor:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-0.5B-Instruct", temperature: float = 0.1, model=None, tokenizer=None):
        """
        Initializes the IDP extractor. Can accept an existing model/tokenizer to save VRAM.
        """
        if model and tokenizer:
            self.model = model
            self.tokenizer = tokenizer
            print("IDPExtractor: Using shared model instance.")
        else:
            # Localize model check
            local_model_path = os.path.join("models", "qwen2.5-0.5b-onnx")
            export_needed = True
            if os.path.exists(local_model_path):
                model_id = local_model_path
                export_needed = False
                print(f"Using localized model: {model_id}")

            # Auto-detect best provider (CUDA > DirectML > CPU)
            provider = get_best_provider()
            print(f"Loading local SLM: {model_id} on {provider}...")
            
            # Load ONNX model with appropriate provider
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = ORTModelForCausalLM.from_pretrained(
                model_id,
                export=export_needed, 
                provider=provider
            )
            print(f"IDPExtractor Active Backend: {provider}")
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=1024,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            repetition_penalty=1.1,
            return_full_text=False,
            device="cpu" 
        )
        print("Extraction engine started via DirectML for AMD GPU")
        
        # System instructions
        self.system_prompt = """You are an expert AI medical intelligence extraction system for the Virtue Foundation.
Your task is to analyze medical notes, surveys, and facility reports and extract a JSON object.

EXTRACTION RULES:
- Entities MUST be mentioned by NAME.
- Be conservative. If a value cannot be determined, omit it (leave as null).
- Output ONLY a JSON object matching this schema:
{
  "facilities": [{"name": "string", "description": "string", "capability": ["string"], "address_country": "string", "address_countryCode": "string"}],
  "ngos": [{"name": "string", "organizationDescription": "string", "countries": ["string"]}]
}
"""
        
    def extract_from_text(self, text: str) -> DocumentExtraction:
        """
        Runs the extraction pipeline and parses the JSON output.
        """
        prompt = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n<|im_start|>user\nDocument Text:\n{text}<|im_end|>\n<|im_start|>assistant\n"
        
        output = self.pipe(prompt)[0]['generated_text']
        
        # Attempt to find JSON block in the output
        try:
            # Look for everything between the first '{' and the last '}'
            match = re.search(r'\{.*\}', output, re.DOTALL)
            if match:
                json_str = match.group(0)
                data = json.loads(json_str)
                return DocumentExtraction(**data)
            else:
                print("No JSON found in model output. Raw output:")
                print(output)
                return DocumentExtraction(facilities=[], ngos=[])
        except Exception as e:
            print(f"Error parsing model output: {e}. Raw output:")
            print(output)
            return DocumentExtraction(facilities=[], ngos=[])

    def extract_from_documents(self, documents: List) -> List[DocumentExtraction]:
        """
        Helper method to extract from multiple LangChain Document objects.
        """
        results = []
        for doc in documents:
            res = self.extract_from_text(doc.page_content)
            results.append(res)
        return results
