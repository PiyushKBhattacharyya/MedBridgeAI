import os
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

def localize_model(model_id: str, local_dir: str):
    """
    Downloads and exports a model to a local directory for offline/portable use.
    """
    print(f"Localizing model: {model_id} to {local_dir}...")
    
    if not os.path.exists(local_dir):
        os.makedirs(local_dir, exist_ok=True)
    
    # Download and export
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(local_dir)
    
    model = ORTModelForCausalLM.from_pretrained(
        model_id,
        export=True,
        provider="CPUExecutionProvider" # Export on CPU for compatibility
    )
    model.save_pretrained(local_dir)
    
    print(f"Model localized successfully at {local_dir}")

if __name__ == "__main__":
    # Localize Qwen 0.5B
    localize_model(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        local_dir="models/qwen2.5-0.5b-onnx"
    )
    
    # FastEmbed handles its own local caching, but we can point it to a local dir later if needed.
    print("\nNext: Update agents to use './models/qwen2.5-0.5b-onnx' instead of the HuggingFace ID.")
