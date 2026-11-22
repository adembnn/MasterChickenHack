"""Génère des non-membres avec SDXL base 1.0 (OpenRAIL++-M).
usage : gen_sdxl.py MODE TAG [strength|scale] [all|subset] [seed]
MODE = t2i (légende -> image) | i2i (image-vers-image, force de bruit = strength) | ipa (IP-Adapter : variation guidée par l'image, sans image initiale)
Sorties : img/TAG/full/<src>.jpg (768 px) et img/TAG.npz (images 224 uint8, src, y)."""
import nm, sys, os, json, time, re, numpy as np, torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, AutoencoderKL
mode, tag = sys.argv[1], sys.argv[2]; par = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
which = sys.argv[4] if len(sys.argv) > 4 else 'subset'; seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
R = int(os.environ.get("RES", 768)); BS = int(os.environ.get("BS", 6))
caps = json.load(open(nm.MEAS + '/captions.json')); y = np.array(caps['pred']); X = nm.forget()
idx = np.load(nm.MEAS + '/subset.npy') if which.startswith('sub') else np.arange(len(X))
if which.startswith('sub') and which[3:].isdigit(): idx = np.sort(np.concatenate([idx[y[idx] == c][:int(which[3:])] for c in np.unique(y[idx])]))   # sub5 = 5 images par classe
def clean(t):
    t = re.sub(r'^(The image is|The image shows|The image features|The image depicts|This is|This image is|This image shows)\s+', '', t.strip())
    return ' '.join(re.split(r'(?<=\.)\s+', t)[:3])[:400]
vae = AutoencoderKL.from_pretrained('madebyollin/sdxl-vae-fp16-fix', torch_dtype=torch.float16)
pipe = StableDiffusionXLPipeline.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0', vae=vae, torch_dtype=torch.float16, variant='fp16')
LOW = os.environ.get('LOWVRAM') == '1'
if not LOW: pipe = pipe.to('cuda')
pipe.set_progress_bar_config(disable=True)
if mode == 'i2i': pipe = StableDiffusionXLImg2ImgPipeline(**pipe.components); pipe.set_progress_bar_config(disable=True)
if LOW: pipe.enable_model_cpu_offload(); pipe.enable_vae_tiling()
if mode == 'ipa':
    pipe.load_ip_adapter('h94/IP-Adapter', subfolder='sdxl_models', weight_name='ip-adapter-plus_sdxl_vit-h.safetensors', image_encoder_folder='models/image_encoder')
    pipe.set_ip_adapter_scale(par)
out = f'{nm.IMG}/{tag}'; os.makedirs(out + '/full', exist_ok=True); t0 = time.time(); small = []
for b in range(0, len(idx), BS):
    ii = idx[b:b + BS]; todo = [i for i in ii if not os.path.exists(f'{out}/full/{i}.jpg')]
    if todo:
        g = torch.Generator('cpu' if LOW else 'cuda').manual_seed(seed * 100000 + int(todo[0]))
        if mode == 't2i':
            ims = pipe(prompt=[clean(caps['detailed'][i]) for i in todo], height=R, width=R, num_inference_steps=24, guidance_scale=5.0, generator=g).images
        elif mode == 'i2i':
            init = [Image.fromarray(X[i]).resize((R, R), Image.BICUBIC) for i in todo]
            ims = pipe(prompt=[clean(caps['detailed'][i]) for i in todo], image=init, strength=par, num_inference_steps=30, guidance_scale=5.0, generator=g).images
        else:
            ims = []
            for i in todo:   # IP-Adapter : une image de référence à la fois
                ims += pipe(prompt=caps['short'][i], ip_adapter_image=Image.fromarray(X[i]).resize((448, 448), Image.BICUBIC), height=R, width=R, num_inference_steps=24, guidance_scale=5.0, generator=g).images
        for i, im in zip(todo, ims): im.save(f'{out}/full/{i}.jpg', quality=95)
    if b % (BS * 10) == 0: print(tag, b, len(idx), round(time.time() - t0), 's', flush=True)
small = np.stack([np.array(Image.open(f'{out}/full/{i}.jpg').convert('RGB').resize((224, 224), Image.LANCZOS)) for i in idx])
np.savez(f'{nm.IMG}/{tag}.npz', images=small, src=idx, y=y[idx]); print('fini', tag, small.shape, round(time.time() - t0), 's', flush=True)
