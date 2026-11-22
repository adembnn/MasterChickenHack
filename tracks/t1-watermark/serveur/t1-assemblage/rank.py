"""Vote par region (idee d'Adem, version apprise) pour les images partielles.
Pour chaque image classee partielle : ~67 masques SAM candidats. Un petit modele predit, pour chaque candidat, combien de mIoU on gagnerait a rendre
ce masque a la place du masque seuille de la carte. Indices : accord entre le candidat et la carte (IoU, logit moyen dedans / dehors / en bordure),
decodeur (marge et bits quand on ne garde que la region, chute quand on l'efface), forme (aire, score SAM), contexte de l'image.
On rend le meilleur candidat si son gain predit depasse tau, sinon le masque de la carte (cas des rectangles, que SAM ne voit pas).
Mesure honnete : le modele est appris en 4 plis sur les images hors reserve (prediction hors-pli), et la reserve est predite par un modele qui ne l'a jamais vue.
Sortie : pred_train_rang-<base>_<filigrane>.npz et pred_test_... (logits +-6 sur les partielles, carte d'origine sur les pleines) = une source de plus pour assemble.py.
Usage : python rank.py reel/mixr [--pool]"""
import sys, glob, json, argparse, numpy as np, torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); from asm import *
ap = argparse.ArgumentParser(); ap.add_argument("base", nargs="?", default="reel/mixr"); ap.add_argument("--pool", action="store_true"); ap.add_argument("--methods", default="trustmark,stegastamp")
ap.add_argument("--seeds", type=int, default=3); a = ap.parse_args()
torch.set_num_threads(4)

def load_cf(names, split):
    d = {}
    for f in glob.glob(str(ME / f"samcand_{split}_*.npz")):
        z = np.load(f)
        for i, n in enumerate(z["names"].tolist()):
            nv = int(z["nvalid"][i])
            if nv:
                c = np.unpackbits(z["masks"][i, :nv], axis=1).astype(bool); k = c.any(1); d[n] = (c[k], z["feats"][i, :nv][k], float(z["acc0"][i]))
    return [d.get(n) for n in names]

def feats(Z, M, cf):
    """Indices par candidat pour une image. Z : carte centree-reduite 128x128 ; M : masque seuille ; cf = (masques [nv,16384], feats SAM/decodeur, bits image)."""
    c, F, acc0 = cf; nv = len(c); z = Z.reshape(-1); m = M.reshape(-1); a_ = c.sum(1).astype(np.float32); cf_ = c.astype(np.float32)
    inter = cf_ @ m.astype(np.float32); un = a_ + m.sum() - inter; iou = inter / np.maximum(un, 1); prec = inter / np.maximum(a_, 1); rec = inter / max(m.sum(), 1)
    zin = (cf_ @ z) / np.maximum(a_, 1); zout = ((1 - cf_) @ z) / np.maximum(16384 - a_, 1)
    c2 = c.reshape(nv, 128, 128); ring = ndi.binary_dilation(c2, structure=np.ones((1, 9, 9), bool)) & ~c2; rf = ring.reshape(nv, -1).astype(np.float32); zring = (rf @ z) / np.maximum(rf.sum(1), 1)
    q10 = np.array([np.quantile(z[k], 0.1) for k in c]); pos = (cf_ @ (z > 0).astype(np.float32)) / np.maximum(a_, 1)
    ys, xs = np.divmod(np.arange(16384), 128); bw = np.array([xs[k].max() - xs[k].min() + 1 for k in c]); bh = np.array([ys[k].max() - ys[k].min() + 1 for k in c]); rectness = a_ / (bw * bh)
    zs = lambda v: (v - v.mean()) / (v.std() + 1e-6)
    X = np.stack([iou, prec, rec, iou - iou.max(), zin, zout, zring, zin - zring, zin - zout, q10, pos, np.sqrt(a_ / 16384), rectness, F[:, 1], F[:, 2],
                  F[:, 3], F[:, 4] - 0.5, F[:, 5], F[:, 6], zs(F[:, 3]), zs(F[:, 5]), zs(F[:, 3] + F[:, 5]), (F[:, 4] - 0.5) * np.sqrt(a_ / 16384),
                  np.full(nv, m.mean()), np.full(nv, acc0 - 0.5), np.full(nv, float(z.max())), np.full(nv, np.log(nv))], 1).astype(np.float32)
    return X

class Net(torch.nn.Module):
    def __init__(s, d): super().__init__(); s.f = torch.nn.Sequential(torch.nn.Linear(d, 48), torch.nn.ReLU(), torch.nn.Linear(48, 24), torch.nn.ReLU(), torch.nn.Linear(24, 1))
    def forward(s, x): return s.f(x)[:, 0]
def fit_predict(Xtr, ytr, Xs, seeds):
    """Regression du gain (MLP, moyenne de plusieurs graines). Renvoie les predictions pour chaque tableau de Xs."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; xt = torch.from_numpy((Xtr - mu) / sd); yt = torch.from_numpy(ytr); outs = [np.zeros(len(x), np.float32) for x in Xs]
    for seed in range(seeds):
        torch.manual_seed(seed); net = Net(Xtr.shape[1]); opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=3e-2)
        for ep in range(400):
            loss = torch.nn.functional.smooth_l1_loss(net(xt), yt, beta=0.1); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            for o, x in zip(outs, Xs): o += net(torch.from_numpy((x - mu) / sd)).numpy() / seeds
    return outs

tr, te, S, notes = discover(); P = populations(tr, te); sh = {r["name"]: r for r in json.loads((ME / "shapes_oracle.json").read_text())}; D = {}
for m in a.methods.split(","):
    if a.base not in S[m]: print(m, ": source", a.base, "absente"); continue
    rows = tr[m]; G = np.stack([r["gt"] for r in rows]); res = np.array([r["reserve"] for r in rows]); pf = P[m]["tr"]; fold = np.array([r["fold"] for r in rows])
    L, Lt = S[m][a.base][0], S[m][a.base][1]; mu, sd = float(L[~res].mean()), float(L[~res].std()) + 1e-6; Z, Zt = (L - mu) / sd, (Lt - mu) / sd
    CF = load_cf([r["image_name"] for r in rows], "train"); CFt = load_cf([r["image_name"] for r in te[m]], "test")
    I = ~pf; fit = I & ~res; p, _ = tune_seuil(Z[fit], G[fit], False); M = np.zeros_like(G); M[I] = base_mask(Z[I], p, False); It = ~P[m]["te"]; Mt = np.zeros(Lt.shape, bool); Mt[It] = base_mask(Zt[It], p, False)
    items = []
    for i in np.where(I)[0]:
        if CF[i] is None: continue
        X = feats(Z[i], M[i], CF[i]); c = CF[i][0]; sc = miou(c.reshape(-1, 128, 128), np.broadcast_to(G[i], (len(c), 128, 128))); b = float(miou(M[i][None], G[i][None])[0])
        items.append(dict(i=int(i), X=X, gain=(sc - b).astype(np.float32), base=b, sc=sc, res=bool(res[i]), fold=int(fold[i]), kind="rectangle" if sh[rows[i]["image_name"]]["rect"] > 0.98 else "objet"))
    tests = [dict(i=int(i), X=feats(Zt[i], Mt[i], CFt[i])) for i in np.where(It)[0] if CFt[i] is not None]
    D[m] = dict(rows=rows, G=G, res=res, pf=pf, L=L, Lt=Lt, M=M, Mt=Mt, CF=CF, CFt=CFt, items=items, tests=tests, I=I, It=It, p=p)

def run(ms, label):
    """Apprend sur les images hors reserve des filigranes ms (ensemble si plusieurs), predit hors-pli, la reserve et le test."""
    its = [(m, it) for m in ms for it in D[m]["items"]]; tr_ = [x for x in its if not x[1]["res"]]
    for m, it in its: it["pred"] = np.zeros(len(it["gain"]), np.float32)
    for f in sorted({it["fold"] for _, it in tr_}):
        A = [it for _, it in tr_ if it["fold"] != f]; B = [it for _, it in tr_ if it["fold"] == f]
        outs = fit_predict(np.concatenate([it["X"] for it in A]), np.concatenate([it["gain"] for it in A]), [it["X"] for it in B], a.seeds)
        for it, o in zip(B, outs): it["pred"] = o
    A = [it for _, it in tr_]; R = [it for _, it in its if it["res"]]; Tt = [t for m in ms for t in D[m]["tests"]]
    outs = fit_predict(np.concatenate([it["X"] for it in A]), np.concatenate([it["gain"] for it in A]), [it["X"] for it in R] + [t["X"] for t in Tt], a.seeds)
    for it, o in zip(R + Tt, outs): it["pred"] = o
    for m in ms:
        d = D[m]; its_m = d["items"]; hors = [it for it in its_m if not it["res"]]
        def score(S_, tau): return np.array([it["base"] + (it["gain"][int(np.argmax(it["pred"]))] if it["pred"].max() > tau else 0.0) for it in S_])
        taus = np.linspace(-0.05, 0.4, 19); tau = float(taus[int(np.argmax([score(hors, t).mean() for t in taus]))])
        def line(S_, nom):
            if not S_: return f"{nom} : -"
            s1 = score(S_, tau); k = np.array([it["kind"] for it in S_]); b = np.array([it["base"] for it in S_]); o = np.array([it["sc"].max() for it in S_]); used = np.mean([it["pred"].max() > tau for it in S_])
            f = lambda msk, v: f"{v[msk].mean():.3f}" if msk.any() else "-"
            return (f"{nom} (n={len(S_)}) : carte seule {b.mean():.3f} -> vote {s1.mean():.3f} | rectangles {f(k == 'rectangle', b)} -> {f(k == 'rectangle', s1)} | objets {f(k == 'objet', b)} -> {f(k == 'objet', s1)}"
                    f" | plafond SAM objets {f(k == 'objet', np.maximum(o, b))} | forme rendue {used:.0%}")
        print(f"{label} · {m} · tau {tau:.3f}\n   " + line(hors, "hors reserve, hors-pli") + "\n   " + line([it for it in its_m if it["res"]], "reserve"), flush=True)
        # source pour assemble.py : +-6 sur les partielles (forme choisie ou masque de la carte), carte d'origine ailleurs
        Ltr = d["L"].copy(); Mo = d["M"].copy()
        for it in its_m:
            if it["pred"].max() > tau: Mo[it["i"]] = d["CF"][it["i"]][0][int(np.argmax(it["pred"]))].reshape(128, 128)
        Ltr[d["I"]] = np.where(Mo[d["I"]], 6.0, -6.0); Lte = d["Lt"].copy(); Mo = d["Mt"].copy(); nb = 0
        for t in d["tests"]:
            if t["pred"].max() > tau: Mo[t["i"]] = d["CFt"][t["i"]][0][int(np.argmax(t["pred"]))].reshape(128, 128); nb += 1
        Lte[d["It"]] = np.where(Mo[d["It"]], 6.0, -6.0); tag = f"rang{'P' if len(ms) > 1 else ''}-{a.base.replace('/', '-')}"
        np.savez_compressed(ME / f"pred_train_{tag}_{m}.npz", names=np.array([r["image_name"] for r in d["rows"]]), logits=Ltr.astype(np.float16))
        np.savez_compressed(ME / f"pred_test_{tag}_{m}.npz", names=np.array([r["image_name"] for r in te[m]]), logits=Lte.astype(np.float16))
        print(f"   ecrit pred_*_{tag}_{m}.npz ; test : forme rendue pour {nb} images partielles sur {int(d['It'].sum())}", flush=True)
ms = [m for m in a.methods.split(",") if m in D]
if a.pool and len(ms) > 1: run(ms, "modele commun")
else:
    for m in ms: run([m], "modele par filigrane")
