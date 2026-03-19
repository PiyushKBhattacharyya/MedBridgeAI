import onnxruntime as ort
import torch
print("--- Hardware Check ---")
print(f"ONNX Runtime Execution Providers: {ort.get_available_providers()}")
print(f"PyTorch CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"PyTorch Device Name: {torch.cuda.get_device_name(0)}")
else:
    print("PyTorch is running on CPU. For AMD GPUs on Windows, specialized setup is needed.")
