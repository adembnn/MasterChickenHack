"""Boites TrustMark finales = cadre cale au pixel par t1-residus quand il est sur (r3 < 0,7), retreci de 3,5 px par bord (le masque vrai est le cadre
du filigrane moins 3 a 4 px), sinon ma boite affinee par la marge brute. Meme regle pour le train et le test. Sortie : boxtm2/boxes_trustmark_<split>_cadre.json."""
import sys, json, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); from t1lib import *
SUF = sys.argv[1] if len(sys.argv) > 1 else "_g13k_r3c"; (ME / "boxtm2").mkdir(exist_ok=True)
def iou(b, g):
    iy = max(0, min(b[2], g[2]) - max(b[0], g[0])); ix = max(0, min(b[3], g[3]) - max(b[1], g[1])); i = iy * ix
    return i / ((b[2] - b[0]) * (b[3] - b[1]) + (g[2] - g[0]) * (g[3] - g[1]) - i)
for split in ("train", "test"):
    mine = {r["name"]: r for r in json.load(open(ME / "boxtm" / f"boxes_trustmark_{split}_affine.json"))}
    hers = {r["name"]: r for r in json.load(open(SESS / "t1-residus" / f"boxes_trustmark_{split}{SUF}.json")) if r.get("box") is not None}
    out = []; nf = 0; cmp_ = []
    for n in sorted(set(mine) | set(hers)):
        h = hers.get(n); m_ = mine.get(n)
        if h is not None and "frame" in h and h["r3"] < 0.7:
            f = h["frame"]; b = [int(f[0] + 4), int(f[1] + 4), int(f[2] - 3), int(f[3] - 3)]; out.append(dict(name=n, box=b, z=max(float(h["z"]), 8.0), src="cadre", r3=float(h["r3"]))); nf += 1
            if split == "train" and h.get("rect") and m_ is not None: cmp_.append((iou(m_["box"], h["gt_box"]), iou(h["box"], h["gt_box"]), iou(b, h["gt_box"])))
        elif m_ is not None: out.append(dict(name=n, box=m_["box"], z=float(m_["z"]), src="marge"))
        else: out.append(dict(name=n, box=[int(v) for v in h["box"]], z=float(h["z"]), src="residus"))
    json.dump(out, open(ME / "boxtm2" / f"boxes_trustmark_{split}_cadre.json", "w")); print(split, len(out), "boites dont", nf, "cadres cales au pixel ; z >= 5 :", sum(r["z"] >= 5 for r in out))
    if cmp_: c = np.array(cmp_); print("   rectangles du train avec cadre cale (n=%d) : recouvrement ma boite %.3f | sa boite %.3f | cadre retreci %.3f (part > 0,9 : %.2f)" % (len(c), c[:, 0].mean(), c[:, 1].mean(), c[:, 2].mean(), (c[:, 2] > .9).mean()))
