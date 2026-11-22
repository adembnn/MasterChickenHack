import os
os.environ["HF_HOME"]="/home/adem/eac/t2/hf"
from huggingface_hub import snapshot_download
print(snapshot_download("HuggingFaceTB/SmolLM2-1.7B-Instruct", allow_patterns=["*.json","*.safetensors","*.txt","*.model","*.jinja"]))
