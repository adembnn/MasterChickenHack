"""Bibliotheque de l'assemblage T1 : lecture des predictions de toutes les sessions, recettes de post-traitement, reglage, validation croisee.
Une "recette" p = dict(mode, sigma, t, clamp, area, snap, snapset) transforme une carte de logits (N x 128 x 128) en masques :
  mode "seuil" : lissage gaussien (sigma), seuil t, fourchette de fraction par population (clamp), nettoyage des petites composantes (area) ;
  mode "rect"  : rectangle de somme (logit - t) maximale (partielles) ;
  box = z0     : fusion avec la recherche de boite de t1-residus (partielles) : si la boite trouvee par le decodeur a une confiance z >= z0,
                 on rend la boite (rectangle d'outpainting relu par le decodeur), sinon le resultat du reseau.
  snap = tau   : "accrochage de forme" (partielles). On compare le masque obtenu a des formes candidates : masques SAM de l'image (objets) et
                 rectangle ajuste. Si la meilleure forme recouvre le masque avec un IoU >= tau, on rend la forme a la place du masque."""
import sys, glob, re, json, time, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); from t1lib import *
from scipy import ndimage as ndi
SIGMAS = (0, 1, 2, 4); AREAS = (0, 8, 32, 128); FRAC = {True: (0.48, 0.995), False: (0.005, 0.35)}   # fourchettes de fraction filigranee (lois du train)
POPN = {True: "pleines", False: "partielles"}; OPT = dict(rect=True, snap=True, only="", skip="")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
RQ = [.3, .4, .5, .6, .7, .75, .8, .85, .9, .93, .95, .97, .98, .99, .995]

# ---------------------------------------------------------------- lecture des sources
def discover():
    """Trois formes de depot : (1) <session>/pred_train_*.npz + pred_test_*.npz ; (2) dossiers seg_*/pred_* (segmenteurs v2 de l'orchestrateur) ;
    (3) runs par pli de t1-reel : t1-*/runs/<run>/<filigrane>/fold*/{oof,res,test}.npz (oof = hors-pli, res = reserve, test), moyennes sur les plis."""
    tr = {m: load_train(m) for m in METHODS}; te = {m: load_test(m) for m in METHODS}; S = {m: {} for m in METHODS}; notes = []
    def add(name_of, na, La, nb, Lb, hh, oof=True):
        da = {str(n): i for i, n in enumerate(na)}; db = {str(n): i for i, n in enumerate(nb)}
        for m in METHODS:
            ntr = [r["image_name"] for r in tr[m]]; nte = [r["image_name"] for r in te[m]]
            ctr = sum(n in da for n in ntr); cte = sum(n in db for n in nte)
            if ctr == 0 and cte == 0: continue
            name = name_of(m)
            if ctr < len(ntr) or cte < len(nte): notes.append(f"{name} [{m}] : couverture partielle (train {ctr}/{len(ntr)}, test {cte}/{len(nte)})"); continue
            if (OPT["only"] and not re.search(OPT["only"], name)) or (OPT["skip"] and re.search(OPT["skip"], name)) or name in S[m]: continue
            S[m][name] = (np.nan_to_num(La[[da[n] for n in ntr]].astype(np.float32), posinf=50, neginf=-50), np.nan_to_num(Lb[[db[n] for n in nte]].astype(np.float32), posinf=50, neginf=-50), hh, oof)
    hm =lambda f: time.strftime("%H:%M", time.localtime(Path(f).stat().st_mtime))
    ROOTS = [SESS, ME / "julich"]; gl = lambda *parts: [f for R in ROOTS for f in sorted(glob.glob(str(R.joinpath(*parts))))]
    for f in gl("t1-*", "pred_train_*.npz") + gl("seg_*", "pred_train_*.npz"):
        f = Path(f); suf = f.name[len("pred_train_"):-4]; ft = f.parent / f"pred_test_{suf}.npz"; d = f.parent.name[3:] if f.parent.name.startswith("t1-") else f.parent.name
        if not ft.exists():
            if suf != "D_post": notes.append(f"{f.parent.name}/{f.name} : pas de pred_test")
            continue
        try:
            za, zb = np.load(f), np.load(ft); na, nb = za["names"].tolist(), zb["names"].tolist(); La, Lb = za["logits"], zb["logits"]
            assert La.shape[1:] == (128, 128) and Lb.shape[1:] == (128, 128) and len(na) == len(La) and len(nb) == len(Lb), (La.shape, Lb.shape)
        except Exception as e:
            notes.append(f"{f.parent.name}/{f.name} : illisible ({type(e).__name__} {e})"); continue
        add(lambda m: f"{d}/" + (re.sub(r"^_+|_+$", "", re.sub(m, "", suf)).replace("__", "_") or "pred"), na, La, nb, Lb, hm(f), oof=not f.parent.name.startswith("seg_"))
    for R in ROOTS:                                                                        # PixelSeal de la soumission v5 (fusion reseau + rectangle faite par t1-pixelseal/post.py) : masques finals, rien a regler
        f, ft = R / "t1-pixelseal" / "pred_train_D_post.npz", R / "t1-pixelseal" / "pred_test_pixelseal.npz"
        if f.exists() and ft.exists():
            try: za, zb = np.load(f), np.load(ft); add(lambda m: "pixelseal/v5", za["names"].tolist(), za["logits"], zb["names"].tolist(), zb["logits"], hm(ft), oof=False)
            except Exception as e: notes.append(f"t1-pixelseal v5 : illisible ({type(e).__name__} {e})")
    for f in gl("t1-*", "run_*_pred_train.npz"):                                           # meme chose, rangee a plat (t1-pixelseal sur la H200)
        f = Path(f); ft = f.parent / f.name.replace("_pred_train", "_pred_test"); nm = f.name[:-len("_pred_train.npz")]
        if not ft.exists() or f.stat().st_size == 0 or ft.stat().st_size == 0: continue
        try:
            za, zb = np.load(f), np.load(ft); add(lambda m: f"{f.parent.name[3:]}/{nm}", za["names"].tolist(), za["logits"], zb["names"].tolist(), zb["logits"], hm(f), oof=False)
        except Exception as e:
            notes.append(f"{f.parent.name}/{f.name} : illisible ({type(e).__name__} {e})")
    for f in gl("t1-*", "run_*", "pred_train.npz"):          # runs de t1-pixelseal : UN modele qui a vu les images hors reserve -> pas hors-pli
        f = Path(f); ft = f.parent / "pred_test.npz"
        if not ft.exists(): continue
        try:
            za, zb = np.load(f), np.load(ft); add(lambda m: f"{f.parent.parent.name[3:]}/{f.parent.name}", za["names"].tolist(), za["logits"], zb["names"].tolist(), zb["logits"], hm(f), oof=False)
        except Exception as e:
            notes.append(f"{f.parent.parent.name}/{f.parent.name} : illisible ({type(e).__name__} {e})")
    for P in gl("t1-*", "runs", "*", "*"):
        P = Path(P); tag = P.parent.name; sess = P.parent.parent.parent.name[3:]
        fs = [d for d in sorted(P.glob("fold*")) if (d / "oof.npz").exists() and (d / "res.npz").exists() and (d / "test.npz").exists()]
        if not fs: continue
        try:
            on, ol = [], []; rs, ts = {}, {}
            for d in fs:
                z = np.load(d / "oof.npz"); on += [str(n) for n in z["names"]]; ol.append(z["logits"].astype(np.float32))
                for acc, fn in ((rs, "res.npz"), (ts, "test.npz")):
                    z = np.load(d / fn)
                    for n, l in zip(z["names"], z["logits"].astype(np.float32)): acc.setdefault(str(n), []).append(l)
            ol = np.concatenate(ol); rn = sorted(rs); tn = sorted(ts)
            add(lambda m: f"{sess}/{tag}" + ("" if len(fs) == 5 else f"({len(fs)}plis)"), on + rn, np.concatenate([ol, np.stack([np.mean(rs[n], 0) for n in rn])]), tn, np.stack([np.mean(ts[n], 0) for n in tn]), hm(fs[-1] / "test.npz"))
        except Exception as e:
            notes.append(f"{sess}/runs/{tag}/{P.name} : illisible ({type(e).__name__} {e})")
    return tr, te, S, notes

def load_cands(names, split):
    """Masques SAM candidats par image (samcand.py) : tableau d'objets aligne sur names ; None si l'image n'en a pas (image pleine, ou fichier absent)."""
    d = {}
    for f in glob.glob(str(ME / f"samcand_{split}_*.npz")):
        z = np.load(f)
        for i, n in enumerate(z["names"].tolist()):
            nv = int(z["nvalid"][i])
            if nv: c = np.unpackbits(z["masks"][i, :nv], axis=1).astype(bool); d[n] = c[c.any(1)]
    out = np.empty(len(names), dtype=object)
    for i, n in enumerate(names): out[i] = d.get(n)
    return out

def load_boxes(names, split, method):
    """Boites de la recherche par decodeur (t1-residus/bs.py) : boxes_<filigrane>_<split>*.json, liste de {name, box (y0, x0, y1, x1 en pixels 512), z}.
    Par image, on garde la boite de plus grand z. Renvoie {nom: (masque 128x128, z)} et la liste des fichiers lus."""
    d = {}; files = []
    for R in (SESS, ME / "julich"):
        for f in sorted(glob.glob(str(R / "t1-*" / "**" / f"boxes_{method}_{split}*.json"), recursive=True)):
            try: recs = json.load(open(f))
            except Exception: continue
            if OPT.get("boxes") and not re.search(OPT["boxes"], str(Path(f).relative_to(R)).replace("\\", "/")): continue
            files.append(str(Path(f).relative_to(R)))
            for r in recs:
                if r.get("box") is None or r.get("z") is None: continue
                b = [int(round(float(v))) for v in r["box"]]; z = float(r["z"])
                if r["name"] not in d or z > d[r["name"]][1]:
                    mk = np.zeros((512, 512), np.float32); mk[max(b[0], 0):b[2], max(b[1], 0):b[3]] = 1; d[r["name"]] = (mk.reshape(128, 4, 128, 4).mean(axis=(1, 3)) >= 0.5, z)
        for f in sorted(glob.glob(str(R / "t1-*" / f"box_{split}.json"))):                 # format de t1-pixelseal : box = (y, x, h, w), confiance = bits justes du rognage
            if OPT.get("boxes") and not re.search(OPT["boxes"], str(Path(f).relative_to(R)).replace("\\", "/")): continue
            try: recs = json.load(open(f))
            except Exception: continue
            want = set(names); nbits = {"pixelseal": 256}.get(method, 100); used = 0
            for r in recs:
                if r.get("box") is None or r["name"] not in want: continue
                y, x, h, w = [int(round(float(v))) for v in r["box"]]; z = (float(r["bits"]) - 0.5) * 2 * nbits ** 0.5; used += 1      # meme echelle que le z de t1-residus
                if r["name"] not in d or z > d[r["name"]][1]:
                    mk = np.zeros((512, 512), np.float32); mk[y:y + h, x:x + w] = 1; d[r["name"]] = (mk.reshape(128, 4, 128, 4).mean(axis=(1, 3)) >= 0.5, z)
            if used: files.append(str(Path(f).relative_to(R)))
    return d, files
def load_side(names, split, method):
    """Informations annexes par image partielle : c = masques SAM candidats, box = (masque de la boite du decodeur, z). Tableau d'objets aligne sur names."""
    C = load_cands(names, split); B, files = load_boxes(names, split, method); out = np.empty(len(names), dtype=object)
    for i, n in enumerate(names): out[i] = dict(c=C[i], box=B.get(n))
    return out, files

# ---------------------------------------------------------------- briques
def smooth(Z, s): return Z if s == 0 else ndi.gaussian_filter(Z, sigma=(0, s, s), mode="nearest")
def qfrac(Z, full, clamp=True):
    """Bornes du seuil par image pour que la part filigranee reste dans la fourchette de sa population. clamp = part minimale de la classe RARE
    (propre dans une pleine, filigranee dans une partielle) ; True = 0,5 % (ancienne valeur)."""
    if clamp is True: lo, hi = FRAC[full]                                 # ancienne fourchette complete
    elif full: lo, hi = 0.0, 1 - float(clamp)                             # seulement : au moins `clamp` de cases propres (pas de plancher de filigrane : une partielle mal classee n'est pas abimee)
    else: lo, hi = float(clamp), 1.0                                      # seulement : au moins `clamp` de cases filigranees
    flat = Z.reshape(len(Z), -1); return np.quantile(flat, 1 - hi, axis=1), np.quantile(flat, 1 - lo, axis=1)
def binarize(Z, t, clamp, full, q=None):
    if not clamp: return Z > t
    q = q or qfrac(Z, full, clamp); return Z > np.clip(t, q[0], q[1])[:, None, None]                 # seuil par image, ramene dans la fourchette de fraction
def clean(M, area):
    if area == 0: return M
    out = M.copy()
    for i in range(len(out)):
        for val in (True, False):                                                          # petites iles filigranees, puis petits trous propres
            lab, n = ndi.label(out[i] == val)
            if n <= 1: continue
            sz = np.bincount(lab.ravel()); small = np.where(sz < area)[0]; small = small[small > 0]
            if len(small) and len(small) < n: out[i][np.isin(lab, small)] = not val
    return out
def best_rect(Zt):
    """Zt : N x 128 x 128, seuil deja retranche. Rectangle de somme maximale par image (vide si aucune somme > 0).
    Toutes les paires de lignes (y0, y1), puis meilleur intervalle de colonnes (algorithme de Kadane), sur GPU si disponible."""
    out = np.zeros(Zt.shape, bool)
    for i0 in range(0, len(Zt), 16):
        Zb = torch.from_numpy(np.ascontiguousarray(Zt[i0:i0 + 16])).to(DEV).float(); N, H, W = Zb.shape
        S = torch.cat([torch.zeros(N, 1, W, device=DEV), Zb.cumsum(1)], 1); P = S[:, None, 1:, :] - S[:, :-1, None, :]      # N x y0 x y1 x W
        cur = torch.zeros(N, H, H, device=DEV); st = torch.zeros(N, H, H, dtype=torch.long, device=DEV)
        best = torch.full((N, H, H), -1e30, device=DEV); b0 = torch.zeros_like(st); b1 = torch.zeros_like(st)
        for x in range(W):
            px = P[..., x]; reset = cur <= 0
            cur = torch.where(reset, px, cur + px); st = torch.where(reset, torch.full_like(st, x), st)
            upd = cur > best; best = torch.where(upd, cur, best); b0 = torch.where(upd, st, b0); b1 = torch.where(upd, torch.full_like(st, x + 1), b1)
        best = best.masked_fill(~torch.ones(H, H, device=DEV).triu().bool(), -1e30).view(N, -1); k = best.argmax(1)
        for j in range(N):
            if best[j, k[j]] > 0:
                y0, y1 = int(k[j] // H), int(k[j] % H) + 1; out[i0 + j, y0:y1, int(b0.view(N, -1)[j, k[j]]):int(b1.view(N, -1)[j, k[j]])] = True
    return out
def snap_core(M, C, snapset="sam+rect", kmax=3):
    """Pour chaque masque M[i], la forme candidate la plus proche : union d'au plus kmax masques SAM (C[i]) et/ou rectangle ajuste au masque.
    Renvoie (formes U, IoU entre la forme et le masque)."""
    U = M.copy(); best = np.zeros(len(M), np.float32)
    if "sam" in snapset and C is not None:
        for i in range(len(M)):
            c = None if C[i] is None else C[i].get("c")
            if c is None or len(c) == 0 or not M[i].any(): continue
            c = c.astype(np.float32); m = M[i].reshape(-1).astype(np.float32); ms = m.sum(); cur = np.zeros_like(m); b = 0.0
            for _ in range(kmax):
                Uc = np.maximum(c, cur[None]); inter = Uc @ m; iou = inter / (Uc.sum(1) + ms - inter + 1e-9); j = int(np.argmax(iou))
                if iou[j] <= b + 0.01: break
                b = float(iou[j]); cur = Uc[j]
            if b > 0: U[i] = cur.reshape(128, 128) > 0.5; best[i] = b
    if "rect" in snapset and len(M):
        R = best_rect(M.astype(np.float32) - 0.5); inter = (R & M).reshape(len(M), -1).sum(1); un = (R | M).reshape(len(M), -1).sum(1); iou = inter / np.maximum(un, 1)
        k = iou > best; U[k] = R[k]; best[k] = iou[k]
    return U, best
def base_mask(Z, p, full):
    if p.get("mode") == "rect": return best_rect(smooth(Z, p["sigma"]) - p["t"])
    return clean(binarize(smooth(Z, p["sigma"]), p["t"], p["clamp"], full), p["area"])
def apply(Z, p, full, C=None):
    if len(Z) == 0: return np.zeros((0, 128, 128), bool)
    M = base_mask(Z, p, full)
    if p.get("snap") is not None:
        U, b = snap_core(M, C, p["snapset"]); M = np.where((b >= p["snap"])[:, None, None], U, M)
    if p.get("box") is not None and C is not None:
        M = M.copy()
        for i in range(len(M)):
            bx = None if C[i] is None else C[i].get("box")
            if bx is not None and bx[1] >= p["box"]: M[i] = bx[0]
    return M

# ---------------------------------------------------------------- reglage
def grid(Zs):
    """Seuils essayes : quantiles des valeurs (bons quand la carte est floue) + grille reguliere entre les quantiles 0,5 % et 99,5 % (indispensable quand
    la carte est tranchee : ses valeurs forment deux paquets et le bon seuil est dans le vide entre les deux, la ou les quantiles ne tombent pas)."""
    v = Zs[:, ::2, ::2]; lo, hi = np.quantile(v, [0.005, 0.995]); return np.unique(np.concatenate([np.quantile(v, np.linspace(0.01, 0.99, 50)), np.linspace(lo, hi, 61)]))
def tune_seuil(Z, G, full):
    best = (-1.0, None)
    if len(Z) == 0: return dict(mode="seuil", sigma=0, t=0.0, clamp=False, area=0), 0.0
    for s in SIGMAS:
        Zs = smooth(Z, s); qs = grid(Zs)
        for clamp in ((False, 0.002, 0.005, 0.01, 0.02, 0.04) if full else (False, 0.005)):   # P4 du verificateur : sans plancher autorise (les trous de moins de 1 % existent)   # pleines : jamais "tout filigrane" (aucune image du train ne l'est ; une telle sortie vaut au plus 0,5)
            Q = qfrac(Zs, full, clamp) if clamp else None
            for t in [-1e9, *qs, 1e9]:
                sc = float(miou(binarize(Zs, t, clamp, full, Q), G).mean())
                if sc > best[0] + 1e-6: best = (sc, dict(mode="seuil", sigma=s, t=float(t), clamp=clamp, area=0))
    p0 = best[1]
    for ar in AREAS[1:]:
        q = dict(p0, area=ar); sc = float(miou(base_mask(Z, q, full), G).mean())
        if sc > best[0] + 1e-6: best = (sc, q)
    return best[1], best[0]
def tune_rect(Z, G):
    best = (-1.0, None)
    if len(Z) == 0: return dict(mode="rect", sigma=0, t=0.0, clamp=False, area=0), 0.0
    for s in SIGMAS[:3]:
        Zs = smooth(Z, s)
        for t in np.quantile(Zs[:, ::2, ::2], RQ):
            sc = float(miou(best_rect(Zs - t), G).mean())
            if sc > best[0] + 1e-6: best = (sc, dict(mode="rect", sigma=s, t=float(t), clamp=False, area=0))
    return best[1], best[0]
def tune_snap(Z, G, C, p0, sc0):
    """Accrochage de forme par-dessus la recette p0 : on re-regle le seuil du masque de base (il sert maintenant a designer une forme) et tau."""
    best = (sc0, p0); Zs = smooth(Z, p0["sigma"])
    ts = [p0["t"]] if p0.get("mode") == "rect" else [p0["t"], *np.quantile(Zs[:, ::2, ::2], [.6, .7, .8, .9, .95, .98])]
    for t in ts:
        q = dict(p0, t=float(t)); M = base_mask(Z, q, False); Us, bs = snap_core(M, C, "sam"); Ur, br = snap_core(M, None, "rect")
        k = br > bs; Ub = np.where(k[:, None, None], Ur, Us); bb = np.where(k, br, bs)           # meme regle que snap_core(M, C, "sam+rect")
        for ss, U, b in (("sam", Us, bs), ("rect", Ur, br), ("sam+rect", Ub, bb)):
            if "sam" in ss and C is None: continue
            for tau in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
                sc = float(miou(np.where((b >= tau)[:, None, None], U, M), G).mean())
                if sc > best[0] + 1e-6: best = (sc, dict(q, snap=tau, snapset=ss))
    return best[1], best[0]
def tune_box(Z, G, C, p0, sc0):
    """Fusion avec la boite du decodeur : seuil de confiance z0 au-dessus duquel la boite remplace le resultat du reseau."""
    best = (sc0, p0)
    if C is None: return p0, sc0
    zs = np.array([c["box"][1] for c in C if c is not None and c.get("box") is not None])
    if len(zs) < 5: return p0, sc0
    for z0 in sorted(set(np.round(np.concatenate([np.quantile(zs, np.linspace(0, 0.9, 10)), [3, 4, 5, 6, 7, 8]]), 2))):
        q = dict(p0, box=float(z0)); sc = float(miou(apply(Z, q, False, C), G).mean())
        if sc > best[0] + 1e-6: best = (sc, q)
    return best[1], best[0]
def tune(Z, G, full, C=None, rect=True, snap=True, box=True):
    """Meilleure recette : seuil ; pour les partielles, aussi rectangle ajuste et accrochage de forme (gardes seulement s'ils gagnent > 0,005)."""
    p, sc = tune_seuil(Z, G, full)
    if full or len(Z) == 0: return p, sc
    if rect and OPT["rect"]:
        p2, sc2 = tune_rect(Z, G)
        if sc2 > sc + 0.005: p, sc = p2, sc2
    if snap and OPT["snap"]:
        p2, sc2 = tune_snap(Z, G, C, p, sc)
        if sc2 > sc + 0.005: p, sc = p2, sc2
    if box and OPT.get("box", True):
        p2, sc2 = tune_box(Z, G, C, p, sc)
        if sc2 > sc + 0.005: p, sc = p2, sc2
    return p, sc
def cv(Z, G, fold, full, C=None, rect=True, snap=True, box=True):
    """Score en validation croisee sur les plis hors reserve : recette reglee sur 3 plis, notee sur le 4e. Sert a CHOISIR (source, melange, recette)."""
    sc = np.zeros(len(Z))
    if len(Z) == 0: return 0.0
    for f in np.unique(fold):
        o = fold == f; p, _ = tune(Z[~o], G[~o], full, None if C is None else C[~o], rect, snap, box); sc[o] = miou(apply(Z[o], p, full, None if C is None else C[o]), G[o])
    return float(sc.mean())
def retune_t(Z, G, full, p, C=None):
    """Pour le fichier final : re-regle seulement le seuil sur toutes les images (meme mode, lissage, nettoyage, accrochage)."""
    if len(Z) == 0: return p
    Zs = smooth(Z, p["sigma"]); best = (float(miou(apply(Z, p, full, C), G).mean()), p)
    for t in (np.quantile(Zs[:, ::2, ::2], RQ) if (p.get("mode") == "rect" or p.get("snap") is not None) else grid(Zs)):
        q = dict(p, t=float(t)); sc = float(miou(apply(Z, q, full, C), G).mean())
        if sc > best[0] + 1e-6: best = (sc, q)
    return best[1]

# ---------------------------------------------------------------- population predite (pleine / partielle)
def populations(tr, te):
    ztr, zte = np.load(ME / "feat_train.npz"), np.load(ME / "feat_test.npz"); dtr = dict(zip(ztr["names"].tolist(), ztr["acc"])); dte = dict(zip(zte["names"].tolist(), zte["acc"]))
    P = {}
    for m in METHODS:
        acc = np.array([dtr[r["image_name"]] for r in tr[m]]); full = np.array([r["full"] for r in tr[m]]); res = np.array([r["reserve"] for r in tr[m]])
        cands = np.unique(acc); th = max(cands, key=lambda t: (np.mean((acc[~res] > t) == full[~res]), -abs(t - 0.75)))
        acc_te = np.array([dte[r["image_name"]] for r in te[m]])
        P[m] = dict(th=float(th), tr=acc > th, te=acc_te > th, exact_res=float(np.mean((acc[res] > th) == full[res])), exact_hors=float(np.mean((acc[~res] > th) == full[~res])), part_test_pleines=float((acc_te > th).mean()))
    return P
