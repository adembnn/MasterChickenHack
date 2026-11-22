"""Piste nouvelle : moyenne des prédictions sur des TRANSLATIONS de l'image à l'inférence (en plus de l'image retournée). Une translation de 4, 8 ou 12 px change la phase des
réductions de résolution internes du réseau (par 2, 4, 8, 16) sans changer l'orientation du filigrane (contrairement aux rotations, qui faisaient baisser le score).
Chaque image hors réserve est prédite par le modèle du pli qui ne l'a pas vue ; la réserve par la moyenne des 5 plis. Aucun entraînement.
usage : tta_shift.py <filigrane> <tag affiné> [--write DOSSIER]"""
import sys, json, argparse, numpy as np, torch
me, tag = sys.argv[1], sys.argv[2]; out = sys.argv[sys.argv.index("--write") + 1] if "--write" in sys.argv else None; sys.argv = sys.argv[:1]
import reel12 as R9
from pathlib import Path
ROOT = R9.ROOT; CACHE = R9.CACHE; trm, tem = R9.metas(); RT = np.load(CACHE / "REAL_train.npy", mmap_mode="r"); RE = np.load(CACHE / "REAL_test.npy", mmap_mode="r"); Y128 = np.load(CACHE / "Y128_train.npy")
idx, res, rest, fold = R9.splits(trm, me); own_te = [i for i, m in enumerate(tem) if m["method"] == me]
a = argparse.Namespace(**json.loads((ROOT / "runs" / tag / me / "fold0" / "args.json").read_text()))
for k, v in dict(npl=0, pl="", hsmall=0.0, hlo=0.005, hhi=0.04, cf=0.0, rsmall=0.0, rband=0.25, wedge=0.0).items(): setattr(a, k, getattr(a, k, v))
SH = [(dy, dx) for dy in (0, 4, 8, 12) for dx in (0, 4, 8, 12)]
def pred(R, X):                                           # X uint8 N x 512 x 512 x 3 -> dict translation -> (logits 128 remis en place, cases valides)
    P = np.pad(X, ((0, 0), (16, 16), (16, 16), (0, 0)), mode="reflect"); o = {}
    for dy, dx in SH:
        with torch.no_grad(): L = R.predict(R.net, P[:, 16 + dy:16 + dy + 512, 16 + dx:16 + dx + 512], [0.0] * len(X), tta=2)
        ay, ax = dy // 4, dx // 4; M = np.zeros((128, 128), bool); Lo = np.zeros_like(L); Lo[:, ay:, ax:] = L[:, :128 - ay, :128 - ax]; M[ay:, ax:] = True; o[(dy, dx)] = (Lo, M)
    return o
def combine(o, keys):
    s = sum(o[k][0] * o[k][1] for k in keys); n = sum(o[k][1].astype(np.float32) for k in keys); return s / np.maximum(n, 1)   # (0,0) couvre tout : n >= 1 partout
V = {"sans translation": [(0, 0)], "4 translations (0 et 8 px)": [(0, 0), (8, 0), (0, 8), (8, 8)], "16 translations (0, 4, 8, 12 px)": SH}
oof = {k: {} for k in V}; rsv = {k: [] for k in V}; tst = {k: [] for k in V}
for f in range(5):
    R = R9.Runner(a, "cuda"); R.net.load_state_dict(torch.load(ROOT / "runs" / tag / me / f"fold{f}" / "ema.pt", map_location="cuda")["state"]); R.net.eval()
    va = [i for i in rest if fold[i] == f]; o = pred(R, np.stack([RT[i] for i in va])); orr = pred(R, np.stack([RT[i] for i in res])); ot = pred(R, np.stack([RE[i] for i in own_te])) if out else None
    for k, keys in V.items():
        c = combine(o, keys)
        for i, l in zip(va, c): oof[k][i] = l
        rsv[k].append(combine(orr, keys))
        if out: tst[k].append(combine(ot, keys))
    print("pli", f, "fait", flush=True)
full = lambda L: np.array([bool(trm[i]["full_watermark"]) for i in L]); fo, fr = full(rest), full(res)
for k in V:
    so = R9.miou_list(np.stack([oof[k][i] for i in rest]) > 0, Y128[rest]); sr = R9.miou_list(np.mean(rsv[k], 0) > 0, Y128[res])
    print(f"{me} {tag} | {k:34s} | CV {so.mean():.4f} (pleines {so[fo].mean():.4f}, partielles {so[~fo].mean():.4f}) | réserve {sr.mean():.4f} (pleines {sr[fr].mean():.4f}, partielles {sr[~fr].mean():.4f})", flush=True)
if out:
    k = "16 translations (0, 4, 8, 12 px)"; Path(out).mkdir(parents=True, exist_ok=True); lg = dict(oof[k]); lg.update({i: l for i, l in zip(res, np.mean(rsv[k], 0))})
    np.savez_compressed(Path(out) / f"pred_train_{me}.npz", names=np.array([trm[i]["image_name"] for i in idx]), logits=np.stack([lg[i] for i in idx]).astype(np.float16))
    np.savez_compressed(Path(out) / f"pred_test_{me}.npz", names=np.array([tem[i]["image_name"] for i in own_te]), logits=np.mean(tst[k], 0).astype(np.float16)); print("écrit", out)
