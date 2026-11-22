# Télécharge les poids nécessaires (tous sans jeton, licences ouvertes). Journal : logs/dl.log
from huggingface_hub import snapshot_download, HfApi
import sys
jobs = [
 ("stabilityai/stable-diffusion-xl-base-1.0", ["model_index.json","scheduler/*","tokenizer/*","tokenizer_2/*","text_encoder/config.json","text_encoder/model.fp16.safetensors","text_encoder_2/config.json","text_encoder_2/model.fp16.safetensors","unet/config.json","unet/diffusion_pytorch_model.fp16.safetensors","vae/config.json","vae/diffusion_pytorch_model.fp16.safetensors","vae_1_0/config.json","vae_1_0/diffusion_pytorch_model.fp16.safetensors"]),
 ("madebyollin/sdxl-vae-fp16-fix", ["config.json","diffusion_pytorch_model.safetensors"]),
 ("florence-community/Florence-2-large", None),
 ("h94/IP-Adapter", ["sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors","sdxl_models/ip-adapter_sdxl_vit-h.safetensors","models/image_encoder/*"]),
 ("city96/FLUX.1-schnell-gguf", ["flux1-schnell-Q8_0.gguf"]),
]
api = HfApi()
try:
    files = [s.rfilename for s in api.model_info("ostris/OpenFLUX.1").siblings]
    print("OpenFLUX files:", files, flush=True)
    jobs.append(("ostris/OpenFLUX.1", ["model_index.json","scheduler/*","tokenizer/*","tokenizer_2/*","text_encoder/*","text_encoder_2/*","vae/*"]))
except Exception as e:
    print("OpenFLUX ERR", e, flush=True)
for repo, pats in jobs:
    try:
        p = snapshot_download(repo, allow_patterns=pats)
        print("OK", repo, p, flush=True)
    except Exception as e:
        print("ERR", repo, type(e).__name__, str(e)[:300], flush=True)
