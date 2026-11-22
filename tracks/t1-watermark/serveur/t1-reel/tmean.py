"""Le motif TrustMark dépend-il surtout du message ou surtout de l'image ? Corrélations entre résidus (= ce que l'outil ajoute à l'image), toile +12 px :
 (a) même image, même message, mais calculé sur l'image déjà filigranée ; (b) deux images différentes, même message ; (c) une image contre la MOYENNE de K autres images, même message ;
 (d) messages différents. Si (c) est haut, on peut fabriquer le motif attendu d'un message sans connaître l'image d'origine, puis le chercher dans l'image test."""
import sys, json, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from scipy import ndimage as ndi
ROOT = Path(__file__).resolve().parent; HERE = ROOT.parent if (ROOT.parent / "kit").exists() else ROOT.parent.parent; CACHE = ROOT / "cache"; sys.path.insert(0, str(HERE / "kit"))
from wmloc.watermarks import load_watermark
trm = json.loads((HERE / "release/train/metadata.json").read_text()); RT = np.load(CACHE / "REAL_train.npy", mmap_mode="r"); Y = np.load(CACHE / "Y128_train.npy"); MT = np.load(CACHE / "MASK_train.npy", mmap_mode="r")
method = sys.argv[1]; K = int(sys.argv[2]); wm = load_watermark(method, models_dir=HERE / "kit" / "models", device="cuda"); rng = np.random.default_rng(0); P = 12
others = [i for i, m in enumerate(trm) if m["method"] != method and i % 5 != 0]
def res(b, m, p=P):
    c = np.pad(b, ((p, p), (p, p), (0, 0)), mode="reflect"); return wm.embed(np.ascontiguousarray(c), m)[p:p + 512, p:p + 512].astype(np.float32) - b
def corr(a, b): a = a - a.mean(); b = b - b.mean(); return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-9))
A, B, C, Dd = [], [], [], []
for t in range(6):
    m = rng.integers(0, 2, wm.nbits); m2 = rng.integers(0, 2, wm.nbits); bs = [np.ascontiguousarray(RT[int(rng.choice(others))]) for _ in range(K + 1)]
    r = [res(b, m) for b in bs]; w0 = np.clip(bs[0] + r[0], 0, 255).astype(np.uint8)
    A.append(corr(r[0], res(w0, m))); B.append(corr(r[0], r[1])); C.append(corr(r[0], np.mean(r[1:], 0))); Dd.append(corr(r[0], res(bs[0], m2)))
print(f"## {method} : corrélation entre résidus (moyenne sur 6 messages)")
print(f"  (a) même image, re-filigranée : {np.mean(A):.3f}\n  (b) autre image, même message : {np.mean(B):.3f}\n  (c) moyenne de {K} autres images, même message : {np.mean(C):.3f}\n  (d) même image, autre message : {np.mean(Dd):.3f}")
# essai direct sur le réel : motif moyen du vrai message (K fonds propres), corrélation locale avec le détail de la vraie image, AUC par image
def auc(s, l):
    s = s.ravel(); l = l.ravel() > 0; o = np.argsort(s, kind="stable"); r_ = np.empty(len(o)); r_[o] = np.arange(1, len(o) + 1); n1 = l.sum(); n0 = len(l) - n1; return (r_[l].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
ids = [i for i, m in enumerate(trm) if m["method"] == method and i % 5 != 0 and m["full_watermark"]][:16]; out = {}
for i in ids:
    if Y[i].min() == Y[i].max(): continue
    m = np.array([int(c) for c in trm[i]["message"]]); T = np.mean([res(np.ascontiguousarray(RT[int(rng.choice(others))]), m) for _ in range(K)], 0); x = np.ascontiguousarray(RT[i]).astype(np.float32)
    for sig in (2.0, 4.0, 8.0):
        xh = x - ndi.gaussian_filter(x, (sig, sig, 0)); th = T - ndi.gaussian_filter(T, (sig, sig, 0))
        for k in (17, 33, 65):
            box = lambda a: ndi.uniform_filter(a.sum(2), k); c = box(xh * th) / np.sqrt(box(xh * xh) * box(th * th) + 1e-9); out.setdefault((sig, k), []).append(auc(c.reshape(128, 4, 128, 4).mean((1, 3)), Y[i]))
print("  carte « motif moyen du message » sur vraies pleines, AUC moyenne par image :", " ; ".join(f"sig={s:.0f} k={k} {np.mean(v):.3f}" for (s, k), v in out.items()))
