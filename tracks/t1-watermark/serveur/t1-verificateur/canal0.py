"""Canal 0 (« détection ») du détecteur PixelSeal, coupé par le kit : carte par position, AUC par image contre la vérité (cases filigranées > propres)."""
import sys
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from verif import HERE, load_truth
sys.path.insert(0, str(HERE / "kit")); from wmloc.watermarks import load_watermark
wm = load_watermark("pixelseal", models_dir=HERE / "kit" / "models", device="cuda"); meta, g = load_truth(check_png=False)
def auc(v, t):
    v = v.ravel(); t = t.ravel(); n1 = t.sum(); n0 = len(t) - n1
    if n1 == 0 or n0 == 0: return np.nan
    r = v.argsort().argsort() + 1; return (r[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
own = [m for m in meta if m["method"] == "pixelseal"]; out = {}
for res in (256, 512, 1024):
    A0, AB, full, glob = [], [], [], []
    for m in own:
        a = np.asarray(Image.open(HERE / "release" / "train" / "images" / f"{m['image_name']}.png").convert("RGB")); x = torch.from_numpy(a).permute(2, 0, 1)[None].float().div(255).cuda()
        with torch.no_grad():
            xr = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False, antialias=True); d = wm.det; lat = d.convnext(d.preprocess(xr)); emb = d.pixel_decoder.output_upscaling(lat); lin = d.pixel_decoder.linear
            pp = torch.einsum("bchw,oc->bohw", emb, lin.weight) + lin.bias[None, :, None, None]
            c0 = F.interpolate(pp[:, :1].float(), size=(128, 128), mode="bilinear")[0, 0].cpu().numpy(); sg = torch.tensor([1.0 if b == "1" else -1.0 for b in m["message"]], device="cuda")
            mb = F.interpolate((pp[:, 1:].float() * sg[None, :, None, None]).mean(1, keepdim=True), size=(128, 128), mode="bilinear")[0, 0].cpu().numpy()
        G = g[m["image_name"]]; A0.append(auc(c0, G)); AB.append(auc(mb, G)); full.append(m["full_watermark"]); glob.append(float(pp[:, 0].mean()))
    A0, AB, full, glob = map(np.array, (A0, AB, full, glob))
    print(f"entrée {res} px, carte {tuple(pp.shape[-2:])} : canal 0, AUC par image : pleines {np.nanmean(A0[full]):.3f}, partielles {np.nanmean(A0[~full]):.3f} | marge des 256 bits dans le sens du message (référence) : pleines {np.nanmean(AB[full]):.3f}, partielles {np.nanmean(AB[~full]):.3f} | canal 0 moyen : pleines {glob[full].mean():+.2f}, partielles {glob[~full].mean():+.2f} (écart-type {glob.std():.2f})")
