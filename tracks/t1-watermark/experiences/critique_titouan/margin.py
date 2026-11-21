"""Marge du decodeur officiel sur l'image entiere, pour les 510 images du train (lecture seule du kit)."""
import json, sys, numpy as np, torch
from PIL import Image
sys.path.insert(0, "/home/titouan/t1crit")
from wmloc.watermarks import load_watermark
KIT = "/home/hermes/eac/kit"
meta = json.load(open(f"{KIT}/release/train/metadata.json"))
out = {}
for meth in ("pixelseal", "trustmark", "stegastamp"):
    wm = load_watermark(meth, f"{KIT}/models", device="cuda")
    items = [m for m in meta if m["method"] == meth]
    for k in range(0, len(items), 16):
        b = items[k:k + 16]
        imgs = np.stack([np.asarray(Image.open(f"{KIT}/release/train/images/{m['image_name']}.png").convert("RGB")) for m in b])
        x = torch.from_numpy(imgs).permute(0, 3, 1, 2).float().div(255).cuda()
        with torch.no_grad():
            lg = wm.logits(x)
        lg = (lg.float().cpu().numpy() if torch.is_tensor(lg) else np.asarray(lg, np.float32))
        for m, l in zip(b, lg):
            msg = np.array([int(c) for c in m["message"]], np.float32)
            s = (2 * msg - 1)[: l.shape[0]] * l[: len(msg)]
            out[m["image_name"]] = (float(np.mean(s)), float(np.mean(s > 0)), float(np.median(s)), float(np.mean(np.tanh(s))))
    print(meth, "ok", len(items), flush=True)
names = list(out)
np.savez("margins_train.npz", names=np.array(names), feats=np.array([out[n] for n in names]))
