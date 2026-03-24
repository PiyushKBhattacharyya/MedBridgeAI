import os
import torch
from transformers import pipeline

class FallbackLLM:
    _instance = None
    _pipeline = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FallbackLLM, cls).__new__(cls)
        return cls._instance

    def _get_pipeline(self):
        if self._pipeline is None:
            print("Loading Fallback LLM (Qwen2.5-0.5B) on CPU... this may take a moment.")
            try:
                # Use a very small, high-quality model for CPU fallback
                model_id = "Qwen/Qwen2.5-0.5B-Instruct"
                self._pipeline = pipeline(
                    "text-generation",
                    model=model_id,
                    torch_dtype=torch.float32,
                    device=-1  # Use device=-1 for explicit CPU without needing accelerate
                )
                print("Fallback LLM loaded successfully.")
            except Exception as e:
                print(f"Error loading fallback LLM: {e}")
                return None
        return self._pipeline

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        pipe = self._get_pipeline()
        if not pipe:
            return "Error: Fallback LLM failed to load."

        messages = [
            {"role": "system", "content": "You are a helpful medical assistant fallback. Provide concise, factual answers based on the provided context."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            from transformers import GenerationConfig
            # Format using the model's chat template
            formatted_prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            gen_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                pad_token_id=pipe.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                truncation=True
            )
            
            outputs = pipe(
                formatted_prompt,
                generation_config=gen_config
            )
            return outputs[0]["generated_text"][len(formatted_prompt):].strip()
        except Exception as e:
            return f"Error during local generation: {e}"

# Global singleton
fallback_llm = FallbackLLM()
