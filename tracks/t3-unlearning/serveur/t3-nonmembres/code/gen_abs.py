"""Images des classes ABSENTES du forget set, générées par SDXL texte-vers-image à partir de noms de classes devinés
(devinés d'après les images les plus activantes, meas/topact_*.jpg). Sert d'instrument de mesure de la rétention sur ces classes (55 % du jeu de contrôle).
usage : gen_abs.py [n_par_classe=20]  -> img/abs_t2i.npz (images 224, y = classe visée, src = -1)"""
import nm, sys, os, time, numpy as np, torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
n = int(sys.argv[1]) if len(sys.argv) > 1 else 20; BS = int(os.environ.get('BS', 4)); R = 768
NAMES = {9: 'a brown clay teapot', 10: 'a black cast iron teapot', 11: 'a white porcelain teapot', 13: 'a red fly agaric mushroom', 14: 'a brown bolete mushroom',
         16: 'a white mushroom', 17: 'a cluster of oyster mushrooms growing on a log', 20: 'a green potted plant', 24: 'an old wooden barn', 26: 'a red and white striped lighthouse',
         27: 'a water tower', 34: 'a wooden rocking chair', 46: 'a deer', 47: 'a beagle dog', 50: 'a fox standing in a field', 55: 'a round loaf of bread', 56: 'a baguette',
         58: 'a pretzel', 61: 'a hot air balloon', 62: 'a brass barometer', 64: 'a wall clock', 70: 'an arctic fox', 74: 'a pocket watch', 75: 'a cello', 77: 'a lit candle',
         89: 'stage spotlights', 91: 'a carved pumpkin jack-o-lantern', 94: 'wooden beehives in a meadow', 95: 'a park bench', 96: 'a stained glass window', 99: 'a hound dog'}
CTX = ['on a wooden table near a window', 'outdoors at sunset', 'in a garden, close-up', 'in a cozy room with warm lamps', 'on a foggy morning']
STYLE = 'photograph, shallow depth of field, soft warm cinematic lighting, blurred background, film look'
vae = AutoencoderKL.from_pretrained('madebyollin/sdxl-vae-fp16-fix', torch_dtype=torch.float16)
pipe = StableDiffusionXLPipeline.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0', vae=vae, torch_dtype=torch.float16, variant='fp16').to('cuda'); pipe.set_progress_bar_config(disable=True)
out = nm.IMG + '/abs_t2i'; os.makedirs(out + '/full', exist_ok=True); jobs = [(c, j) for c in NAMES for j in range(n)]; t0 = time.time()
for b in range(0, len(jobs), BS):
    todo = [(c, j) for c, j in jobs[b:b + BS] if not os.path.exists(f'{out}/full/{c}_{j}.jpg')]
    if todo:
        g = torch.Generator('cuda').manual_seed(1000 * todo[0][0] + todo[0][1])
        ims = pipe(prompt=[f'{NAMES[c]} {CTX[j % len(CTX)]}, {STYLE}' for c, j in todo], height=R, width=R, num_inference_steps=24, guidance_scale=5.0, generator=g).images
        for (c, j), im in zip(todo, ims): im.save(f'{out}/full/{c}_{j}.jpg', quality=95)
    if b % (BS * 10) == 0: print('abs_t2i', b, len(jobs), round(time.time() - t0), 's', flush=True)
X = np.stack([np.array(Image.open(f'{out}/full/{c}_{j}.jpg').convert('RGB').resize((224, 224), Image.LANCZOS)) for c, j in jobs])
np.savez(nm.IMG + '/abs_t2i.npz', images=X, src=-np.ones(len(jobs), int), y=np.array([c for c, j in jobs])); json_names = {int(k): v for k, v in NAMES.items()}
import json; json.dump(json_names, open(nm.MEAS + '/noms_classes_absentes.json', 'w'), indent=1); print('fini abs_t2i', X.shape, round(time.time() - t0), 's', flush=True)
