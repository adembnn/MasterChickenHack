"""Non-membres avec FLUX.1-schnell (Apache-2.0) : transformeur GGUF Q8 (city96), encodeurs de texte et VAE pris dans ostris/OpenFLUX.1 (Apache-2.0).
usage : gen_flux.py MODE TAG [strength] [all|subset] [seed]   MODE = t2i | i2i"""
import nm, sys, os, json, time, re, numpy as np, torch
from PIL import Image
from huggingface_hub import hf_hub_download, snapshot_download
from diffusers import FluxPipeline, FluxImg2ImgPipeline, FluxTransformer2DModel, GGUFQuantizationConfig, FlowMatchEulerDiscreteScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
mode, tag = sys.argv[1], sys.argv[2]; par = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
which = sys.argv[4] if len(sys.argv) > 4 else 'subset'; seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
R = 768; BS = int(os.environ.get('BS', 4))
caps = json.load(open(nm.MEAS + '/captions.json')); y = np.array(caps['pred']); X = nm.forget()
idx = np.load(nm.MEAS + '/subset.npy') if which == 'subset' else np.arange(len(X))
def clean(t):
    t = re.sub(r'^(The image is|The image shows|The image features|The image depicts|This is|This image is|This image shows)\s+', '', t.strip())
    return ' '.join(re.split(r'(?<=\.)\s+', t)[:4])[:600]
cfg = os.path.dirname(os.path.abspath(__file__)) + '/flux_schnell'
gg = hf_hub_download('city96/FLUX.1-schnell-gguf', 'flux1-schnell-Q8_0.gguf')
tr = FluxTransformer2DModel.from_single_file(gg, quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16), torch_dtype=torch.bfloat16, config=cfg, subfolder='transformer')
of = snapshot_download('ostris/OpenFLUX.1', allow_patterns=['model_index.json', 'scheduler/*', 'tokenizer/*', 'tokenizer_2/*', 'text_encoder/*', 'text_encoder_2/*', 'vae/*'])
sch = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0, use_dynamic_shifting=False, base_shift=0.5, max_shift=1.15, base_image_seq_len=256, max_image_seq_len=4096)
cls = FluxPipeline if mode == 't2i' else FluxImg2ImgPipeline
pipe = cls(scheduler=sch, vae=AutoencoderKL.from_pretrained(of, subfolder='vae', torch_dtype=torch.bfloat16),
           text_encoder=CLIPTextModel.from_pretrained(of, subfolder='text_encoder', torch_dtype=torch.bfloat16), tokenizer=CLIPTokenizer.from_pretrained(of, subfolder='tokenizer'),
           text_encoder_2=T5EncoderModel.from_pretrained(of, subfolder='text_encoder_2', torch_dtype=torch.bfloat16), tokenizer_2=T5TokenizerFast.from_pretrained(of, subfolder='tokenizer_2'), transformer=tr).to('cuda')
pipe.set_progress_bar_config(disable=True)
out = f'{nm.IMG}/{tag}'; os.makedirs(out + '/full', exist_ok=True); t0 = time.time()
for b in range(0, len(idx), BS):
    ii = idx[b:b + BS]; todo = [i for i in ii if not os.path.exists(f'{out}/full/{i}.jpg')]
    if todo:
        g = torch.Generator('cuda').manual_seed(seed * 100000 + int(todo[0])); pr = [clean(caps['detailed'][i]) for i in todo]
        if mode == 't2i':
            ims = pipe(prompt=pr, height=R, width=R, num_inference_steps=4, guidance_scale=0.0, max_sequence_length=256, generator=g).images
        else:
            init = [Image.fromarray(X[i]).resize((R, R), Image.BICUBIC) for i in todo]
            ims = pipe(prompt=pr, image=init, strength=par, height=R, width=R, num_inference_steps=max(4, round(4 / par)), guidance_scale=0.0, max_sequence_length=256, generator=g).images
        for i, im in zip(todo, ims): im.save(f'{out}/full/{i}.jpg', quality=95)
    if b % (BS * 10) == 0: print(tag, b, len(idx), round(time.time() - t0), 's', flush=True)
small = np.stack([np.array(Image.open(f'{out}/full/{i}.jpg').convert('RGB').resize((224, 224), Image.LANCZOS)) for i in idx])
np.savez(f'{nm.IMG}/{tag}.npz', images=small, src=idx, y=y[idx]); print('fini', tag, small.shape, round(time.time() - t0), 's', flush=True)
