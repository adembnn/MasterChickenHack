import os
os.environ["HF_HOME"]="/home/adem/eac/t2/hf"
from huggingface_hub import snapshot_download
for m in ["allenai/OLMo-2-0425-1B-Instruct","Qwen/Qwen2.5-1.5B-Instruct"]:
    p=snapshot_download(m, allow_patterns=["*.json","*.safetensors","*.txt","*.model","*.jinja"])
    print(m,p,flush=True)
