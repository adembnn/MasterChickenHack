import numpy as np, zipfile, io
from PIL import Image
REL = "/home/adem/eac/t1/release"; zf = zipfile.ZipFile("coco/train2017.zip"); rows = []
for side in ("train", "test"):
    z = np.load(f"cocomatch_train2017_{side}.npz"); o = np.argsort(-z["sim"])[:6]
    for i in o: rows.append((side, str(z["names"][i]), str(z["coco"][i]), float(z["sim"][i]), float(z["second"][i]), int(z["var"][i])))
W = Image.new("RGB", (6 * 2 * 200, 2 * 200), "white")
for k, (side, n, c, s, s2, v) in enumerate(rows):
    a = Image.open(f"{REL}/{side}/images/{n}.png").convert("RGB").resize((200, 200)); b = Image.open(io.BytesIO(zf.read("train2017/" + c))).convert("RGB"); print(side, n, c, round(s, 3), "deuxième", round(s2, 3), "variante", v, "taille COCO", b.size)
    bw, bh = b.size; b.thumbnail((200, 200)); x = (k % 6) * 400; y = (k // 6) * 200; W.paste(a, (x, y)); W.paste(b, (x + 200, y))
W.save("cocoview.jpg", quality=85)
