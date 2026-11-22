"""Fusion v2 « réseau + rectangle » après critique : (a) tri pleine/partielle SANS la vérité (bits justes image entière < 0,75) ;
(b) plusieurs règles de repli quand le cadre n'a pas été calé (R3 >= seuil), comparées HORS réserve ; (c) écrit les fichiers de la règle choisie.
Usage : python fuse2.py <methode> <pred_train> <pred_test> <boxes_train> <boxes_test> <nom_sortie> [regle]"""
import sys, json, numpy as np
from pathlib import Path
method, ptr, pte, btr, bte, name = sys.argv[1:7]; RULE = sys.argv[7] if len(sys.argv) > 7 else ""
R = Path("E:/EAC-lab/t1/release/train"); meta = json.loads((R / "metadata.json").read_text()); m128 = np.load(R / "masks_128.npz"); G = dict(zip(m128["names"].tolist(), m128["masks"]))
def miou(p, g):
    s = 0.0
    for c in (0, 1):
        u = ((p == c) | (g == c)).sum(); s += 1.0 if u == 0 else ((p == c) & (g == c)).sum() / u
    return s / 2
def rect(b):
    mk = np.zeros((512, 512), np.float32); mk[max(0, int(round(b[0]))):max(0, int(round(b[2]))), max(0, int(round(b[1]))):max(0, int(round(b[3])))] = 1
    return mk.reshape(128, 4, 128, 4).mean((1, 3)) >= 0.5
def frame_mask(r, sh=3.5): f = r["frame"]; return rect((f[0] + sh, f[1] + sh, f[2] - sh, f[3] - sh))
# règle = (zmin, r3max, repli) ; repli appliqué quand z > zmin mais R3 >= r3max
RULES = {"A boîte si non calé": (6.5, 0.7, "box"), "B réseau si non calé": (6.5, 0.7, "net"), "C boîte si z>8,5 sinon réseau": (6.5, 0.7, "box85"),
         "D boîte ∩ réseau si non calé": (6.5, 0.7, "inter"), "E boîte rétrécie 3 px si non calé": (6.5, 0.7, "boxsh"), "F comme C, R3<0,6": (6.5, 0.6, "box85"), "G comme A, R3<0,6": (6.5, 0.6, "box")}
def apply(L, names, B, rule):
    zmin, r3max, fb = rule; L = L.copy(); n = {"calé": 0, "repli": 0}
    for i, nm in enumerate(names):
        r = B.get(nm)
        if not r or "box" not in r or r["acc_full"] >= 0.75 or r["z"] <= zmin: continue
        if "frame" in r and r["r3"] < r3max: m = frame_mask(r); n["calé"] += 1
        else:
            b = r["box"]; net = L[i] > 0
            if (b[2] - b[0]) * (b[3] - b[1]) > 0.45 * 512 * 512: n["garde"] = n.get("garde", 0) + 1; continue      # garde-fou : aucun rectangle du train ne dépasse 43 % de l'image
            if fb == "net" or (fb == "box85" and r["z"] <= 8.5): continue
            m = rect(b) if fb in ("box", "box85") else (rect((b[0] + 3, b[1] + 3, b[2] - 3, b[3] - 3)) if fb == "boxsh" else (rect(b) & net)); n["repli"] += 1
        L[i] = np.where(m, 6.0, -6.0)
    return L, n
z = np.load(ptr, allow_pickle=True); names = z["names"].tolist(); L0 = z["logits"].astype(np.float32); B = {r["name"]: r for r in json.load(open(btr))}
info = {e["image_name"]: (i, bool(e.get("full_watermark"))) for i, e in enumerate(meta) if e["method"] == method}
res = np.array([info[nm][0] % 5 == 0 for nm in names]); full = np.array([info[nm][1] for nm in names]); s0 = np.array([miou(L0[i] > 0, G[nm] > 0) for i, nm in enumerate(names)])
print("FUSE2 %s base=%s boîtes=%s | avant : hors réserve %.4f, réserve %.4f" % (method, Path(ptr).name, Path(btr).name, s0[~res].mean(), s0[res].mean()))
best = None
for k, rule in RULES.items():
    L1, n = apply(L0, names, B, rule); s1 = np.array([miou(L1[i] > 0, G[nm] > 0) for i, nm in enumerate(names)]); d = s1 - s0
    print("  %-34s calés %2d, replis %2d | HORS RÉSERVE %.4f (%+.4f ± %.4f ; dégradées %d) | réserve %.4f (%+.4f) | partielles hors réserve %.4f" % (k, n["calé"], n["repli"], s1[~res].mean(), d[~res].mean(), d[~res].std() / np.sqrt((~res).sum()), (d[~res] < -1e-6).sum(), s1[res].mean(), d[res].mean(), s1[~res & ~full].mean()))
    if best is None or s1[~res].mean() > best[1] + 1e-9: best = (k, s1[~res].mean())
print("  meilleure règle hors réserve :", best[0])
if RULE:
    k = [q for q in RULES if q.startswith(RULE)][0]; out = Path("E:/EAC-lab/t1s/t1-residus/out"); L1, n = apply(L0, names, B, RULES[k]); np.savez_compressed(out / f"pred_train_{name}_{method}.npz", names=np.array(names), logits=L1.astype(np.float16))
    zt = np.load(pte, allow_pickle=True); tn = zt["names"].tolist(); T0 = zt["logits"].astype(np.float32); T1, nt = apply(T0, tn, {r["name"]: r for r in json.load(open(bte))}, RULES[k])
    np.savez_compressed(out / f"pred_test_{name}_{method}.npz", names=np.array(tn), logits=T1.astype(np.float16))
    print("  écrit règle « %s » : train calés %d replis %d ; test calés %d replis %d (sur %d images) ; part filigranée test avant %.3f après %.3f" % (k, n["calé"], n["repli"], nt["calé"], nt["repli"], len(tn), (T0 > 0).mean(), (T1 > 0).mean()))
