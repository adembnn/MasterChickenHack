import sys, os
os.environ.setdefault("HF_HOME", "/home/adem/eac/t2/hf")
from huggingface_hub import snapshot_download
for m in sys.argv[1:]:
    try:
        snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])
        print("OK", m, flush=True)
    except Exception as e:
        print("ECHEC", m, type(e).__name__, str(e)[:120], flush=True)
