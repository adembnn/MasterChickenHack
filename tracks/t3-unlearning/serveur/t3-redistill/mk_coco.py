import os, numpy as np
from PIL import Image
d = os.path.expanduser("~/eac/shared_data/coco/val2017"); fs = sorted(os.listdir(d))[:5000]; out = np.zeros((len(fs), 224, 224, 3), np.uint8)
for i, f in enumerate(fs):
    im = Image.open(os.path.join(d, f)).convert("RGB"); w, h = im.size; s = 256 / min(w, h)
    im = im.resize((max(256, round(w * s)), max(256, round(h * s))), Image.BICUBIC); w, h = im.size; l = (w - 224) // 2; t = (h - 224) // 2
    out[i] = np.asarray(im.crop((l, t, l + 224, t + 224)))
np.save("res/coco5k.npy", out); print("ok", out.shape)
