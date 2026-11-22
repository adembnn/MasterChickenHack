"""Idee : les images du hackathon sortent d'un generateur a diffusion latente. Leur partie PROPRE est donc (presque) dans l'image de l'autoencodeur
(VAE) du generateur : encoder puis decoder la redonne a tres peu pres. Une zone FILIGRANEE porte en plus le residu du filigrane, que le VAE ne
reproduit pas. D = image - decode(encode(image)) serait alors une estimation du residu, la piece qui manque a toutes nos methodes (l'image propre).
Mesure, par VAE candidat, sur le vrai train : erreur de reconstruction (niveaux sur 255) dans les zones propres et filigranees, et AUC par case
de l'energie locale de D (16 px) entre cases filigranees et propres. Usage : python vaetest.py [n_par_filigrane]"""
import sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); from t1lib import *
from PIL import Image
from diffusers import AutoencoderKL
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40; dev = "cuda"; torch.set_num_threads(4)
VAES = [("sd15 (sd-vae-ft-mse)", "stabilityai/sd-vae-ft-mse", None), ("sdxl", "stabilityai/sdxl-vae", None), ("flux (OpenFLUX.1)", "ostris/OpenFLUX.1", "vae"),
        ("flux (Flex.1-alpha)", "ostris/Flex.1-alpha", "vae"), ("flux schnell", "black-forest-labs/FLUX.1-schnell", "vae"), ("sd3 (mirror)", "v2ray/stable-diffusion-3-medium-diffusers", "vae"),
        ("16ch ostris", "ostris/vae-kl-f8-d16", None), ("playground v2.5", "playgroundai/playground-v2.5-1024px-aesthetic", "vae")]
def load_vae(repo, sub):
    import json, safetensors.torch
    from huggingface_hub import hf_hub_download
    kw = dict(subfolder=sub) if sub else {}
    cfg = json.load(open(hf_hub_download(repo, "config.json", **kw))); vae = AutoencoderKL.from_config(cfg)
    sd = safetensors.torch.load_file(hf_hub_download(repo, "diffusion_pytorch_model.safetensors", **kw)); print("   cles manquantes / en trop :", [len(x) for x in vae.load_state_dict(sd, strict=False)], flush=True)
    return vae.to(dev).eval()
rows = []
for m in METHODS:
    R = load_train(m); rows += [r for r in R if r["full"]][:N // 2] + [r for r in R if not r["full"]][:N // 2]
def auc(s, y):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))
for label, repo, sub in VAES:
    try: vae = load_vae(repo, sub)
    except Exception as e: print(f"{label:22s} indisponible ({type(e).__name__}: {str(e)[:90]})", flush=True); continue
    out = {}
    for r in rows:
        img = load_img("train", r["image_name"]); gt = np.asarray(Image.open(HERE / "release/train/masks" / f"{r['image_name']}.png")) > 127
        x = torch.from_numpy(img).permute(2, 0, 1)[None].float().to(dev) / 127.5 - 1
        with torch.no_grad(): rec = vae.decode(vae.encode(x).latent_dist.mean).sample.clamp(-1, 1)
        D = ((x - rec) * 127.5)[0].cpu().numpy()                                   # 3 x 512 x 512, en niveaux
        e = (D ** 2).mean(0); rc = np.sqrt(e[~gt].mean()) if (~gt).any() else np.nan; rw = np.sqrt(e[gt].mean()) if gt.any() else np.nan
        en = F.avg_pool2d(torch.from_numpy(e)[None, None], 16, 4, 6)[0, 0].numpy()[:128, :128]; g = r["gt"]
        a_ = auc(en.ravel(), g.ravel()) if 0 < g.mean() < 1 else np.nan
        # meme chose apres avoir retire ce que l'image "explique" : energie de D relative a la texture locale de l'image
        out.setdefault((r["method"], r["full"]), []).append((rc, rw, a_))
    line = f"{label:22s}"
    for m in METHODS:
        for full in (True, False):
            v = np.array(out[(m, full)], float); line += f" | {m[:5]} {'pl' if full else 'pa'} propre {np.nanmedian(v[:, 0]):5.2f} filigr. {np.nanmedian(v[:, 1]):5.2f} AUC {np.nanmean(v[:, 2]):.2f}"
    print(line, flush=True); del vae; torch.cuda.empty_cache()
