import torch, os, time
from huggingface_hub import hf_hub_download
from diffusers import FluxTransformer2DModel, GGUFQuantizationConfig
t=time.time(); gg = hf_hub_download("city96/FLUX.1-schnell-gguf", "flux1-schnell-Q8_0.gguf")
tr = FluxTransformer2DModel.from_single_file(gg, quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16), torch_dtype=torch.bfloat16, config=os.path.abspath("flux_schnell"), subfolder="transformer")
print("FLUX_TR_OK", type(tr).__name__, tr.config.guidance_embeds, sum(p.numel() for p in tr.parameters())/1e9, round(time.time()-t))
