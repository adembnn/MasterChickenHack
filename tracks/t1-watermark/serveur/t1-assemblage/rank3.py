"""Vote par region v3 (proposition P1 du verificateur, 22/11) : chaque masque SAM est juge SEUL (garde ou rejete) par un classifieur appris en
commun sur les images partielles des trois filigranes ; on rend l'UNION des masques gardes (objets en plusieurs morceaux), sinon le masque de la carte.
Indices par masque : logit moyen dedans / dehors / dans l'anneau voisin, part de cases > 0, quantile bas, aire, forme, scores SAM, decodeur
(garder / effacer la region), accord avec le masque de la carte, contexte de l'image, filigrane.
Etiquette : le masque est-il majoritairement dans la vraie zone filigranee ? Apprentissage en 4 plis hors reserve (index % 5), reserve predite par le
modele appris sans elle ; seuil de decision regle hors reserve, par filigrane.
Sortie : pred_{train,test}_rang3-<base>_<filigrane>.npz. Usage : python rank3.py trustmark=reelx2/pred stegastamp=reels2/pred pixelseal=reelp2/pred"""
import sys, glob, json, numpy as np, torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); from asm import *
BASES = dict(kv.split("=") for kv in sys.argv[1:] if "=" in kv); SEEDS = 3; torch.set_num_threads(4)
def load_cf(names, split):
    d = {}
    for f in glob.glob(str(ME / f"samcand_{split}_*.npz")):
        z = np.load(f)
        for i, n in enumerate(z["names"].tolist()):
            nv = int(z["nvalid"][i])
            if nv: c = np.unpackbits(z["masks"][i, :nv], axis=1).astype(bool); k = c.any(1); d[n] = (c[k], z["feats"][i, :nv][k], float(z["acc0"][i]))
    return [d.get(n) for n in names]
def feats(Z, M, cf, mi):
    c, F, acc0 = cf; nv = len(c); z = Z.reshape(-1); m = M.reshape(-1); a_ = c.sum(1).astype(np.float32); cf_ = c.astype(np.float32)
    inter = cf_ @ m.astype(np.float32); prec = inter / np.maximum(a_, 1); rec = inter / max(m.sum(), 1); iou = inter / np.maximum(a_ + m.sum() - inter, 1)
    zin = (cf_ @ z) / np.maximum(a_, 1); zout = ((1 - cf_) @ z) / np.maximum(16384 - a_, 1)
    c2 = c.reshape(nv, 128, 128); ring = ndi.binary_dilation(c2, structure=np.ones((1, 9, 9), bool)) & ~c2; rf = ring.reshape(nv, -1).astype(np.float32); zring = (rf @ z) / np.maximum(rf.sum(1), 1)
    q10 = np.array([np.quantile(z[k], 0.1) for k in c]); q50 = np.array([np.median(z[k]) for k in c]); pos = (cf_ @ (z > 0).astype(np.float32)) / np.maximum(a_, 1); posr = (rf @ (z > 0).astype(np.float32)) / np.maximum(rf.sum(1), 1)
    ys, xs = np.divmod(np.arange(16384), 128); bw = np.array([xs[k].max() - xs[k].min() + 1 for k in c]); bh = np.array([ys[k].max() - ys[k].min() + 1 for k in c]); rectness = a_ / (bw * bh)
    zs = lambda v: (v - v.mean()) / (v.std() + 1e-6); oh = np.zeros((nv, 3), np.float32); oh[:, mi] = 1
    X = np.stack([zin, zout, zring, zin - zring, zin - zout, q10, q50, pos, posr, pos - posr, np.sqrt(a_ / 16384), rectness, F[:, 1], F[:, 2], F[:, 3], F[:, 4] - 0.5, F[:, 5], F[:, 6], zs(F[:, 3]), zs(F[:, 5]),
                  (F[:, 4] - 0.5) * np.sqrt(a_ / 16384), prec, rec, iou, zin - zin.max(), np.full(nv, m.mean()), np.full(nv, acc0 - 0.5), np.full(nv, float(z.max())), np.full(nv, np.log(nv))], 1).astype(np.float32)
    return np.concatenate([X, oh], 1)
class Net(torch.nn.Module):
    def __init__(s, d): super().__init__(); s.f = torch.nn.Sequential(torch.nn.Linear(d, 64), torch.nn.ReLU(), torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
    def forward(s, x): return s.f(x)[:, 0]
def fit_predict(Xtr, ytr, wtr, Xs):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; xt = torch.from_numpy((Xtr - mu) / sd); yt = torch.from_numpy(ytr); wt = torch.from_numpy(wtr); outs = [np.zeros(len(x), np.float32) for x in Xs]
    for seed in range(SEEDS):
        torch.manual_seed(seed); net = Net(Xtr.shape[1]); opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=3e-2)
        for ep in range(500):
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(net(xt), yt, reduction="none") * wt).mean(); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            for o, x in zip(outs, Xs): o += torch.sigmoid(net(torch.from_numpy((x - mu) / sd))).numpy() / SEEDS
    return outs
tr, te, S, notes = discover(); P = populations(tr, te); sh = {r["name"]: r for r in json.loads((ME / "shapes_oracle.json").read_text())}; D = {}
for mi, m in enumerate(METHODS):
    if m not in BASES or BASES[m] not in S[m]: continue
    rows = tr[m]; G = np.stack([r["gt"] for r in rows]); res = np.array([r["reserve"] for r in rows]); pf = P[m]["tr"]; fold = np.array([r["fold"] for r in rows])
    L, Lt = S[m][BASES[m]][0], S[m][BASES[m]][1]; mu, sd = float(L[~res].mean()), float(L[~res].std()) + 1e-6; Z, Zt = (L - mu) / sd, (Lt - mu) / sd
    CF = load_cf([r["image_name"] for r in rows], "train"); CFt = load_cf([r["image_name"] for r in te[m]], "test")
    I = ~pf; fit = I & ~res; p, _ = tune_seuil(Z[fit], G[fit], False); M = np.zeros_like(G); M[I] = base_mask(Z[I], p, False); It = ~P[m]["te"]; Mt = np.zeros(Lt.shape, bool); Mt[It] = base_mask(Zt[It], p, False)
    v1 = None; f1 = ME / f"pred_train_rang-{BASES[m].replace('/', '-')}_{m}.npz"
    if f1.exists(): z1 = np.load(f1); v1 = dict(zip(z1["names"].tolist(), z1["logits"].astype(np.float32) > 0))
    items = []
    for i in np.where(I)[0]:
        if CF[i] is None: continue
        c = CF[i][0]; g = G[i].reshape(-1); a_ = c.sum(1); ins = (c & g[None]).sum(1) / np.maximum(a_, 1)
        items.append(dict(m=m, i=int(i), X=feats(Z[i], M[i], CF[i], mi), y=(ins >= 0.5).astype(np.float32), w=np.sqrt(a_ / 16384).astype(np.float32) + 0.05, res=bool(res[i]), fold=int(fold[i]),
                          base=float(miou(M[i][None], G[i][None])[0]), v1=float(miou(v1[rows[i]["image_name"]][None], G[i][None])[0]) if v1 else np.nan, kind="rectangle" if sh[rows[i]["image_name"]]["rect"] > 0.98 else "objet"))
    tests = [dict(m=m, i=int(i), X=feats(Zt[i], Mt[i], CFt[i], mi)) for i in np.where(It)[0] if CFt[i] is not None]
    D[m] = dict(rows=rows, G=G, L=L, Lt=Lt, M=M, Mt=Mt, CF=CF, CFt=CFt, items=items, tests=tests, I=I, It=It)
its = [it for m in D for it in D[m]["items"]]; trn = [it for it in its if not it["res"]]
cat = lambda A, k: np.concatenate([it[k] for it in A])
for f in sorted({it["fold"] for it in trn}):
    A = [it for it in trn if it["fold"] != f]; B = [it for it in trn if it["fold"] == f]
    for it, o in zip(B, fit_predict(cat(A, "X"), cat(A, "y"), cat(A, "w"), [it["X"] for it in B])): it["p"] = o
R = [it for it in its if it["res"]]; Tt = [t for m in D for t in D[m]["tests"]]
for it, o in zip(R + Tt, fit_predict(cat(trn, "X"), cat(trn, "y"), cat(trn, "w"), [it["X"] for it in R + Tt])): it["p"] = o
def union(d, it, tau, test=False):
    c = (d["CFt"] if test else d["CF"])[it["i"]][0]; k = it["p"] >= tau; base = (d["Mt"] if test else d["M"])[it["i"]]
    if not k.any(): return base
    u = c[k].any(0).reshape(128, 128); return u if u.mean() <= 0.45 else base
for m in D:
    d = D[m]; G = d["G"]; hors = [it for it in d["items"] if not it["res"]]; taus = np.linspace(0.3, 0.9, 13)
    sc = lambda S_, tau: np.array([float(miou(union(d, it, tau)[None], G[it["i"]][None])[0]) for it in S_])
    tau = float(taus[int(np.argmax([sc(hors, t).mean() for t in taus]))])
    def line(S_, nom):
        if not S_: return f"{nom} : -"
        s3 = sc(S_, tau); k = np.array([it["kind"] for it in S_]); b = np.array([it["base"] for it in S_]); s1 = np.array([it["v1"] for it in S_]); f = lambda msk, v: f"{v[msk].mean():.3f}" if msk.any() else "-"
        return f"{nom} (n={len(S_)}) : carte {b.mean():.3f} | vote v1 (un masque) {np.nanmean(s1):.3f} | vote v3 (union) {s3.mean():.3f} || objets : {f(k == 'objet', b)} / {f(k == 'objet', s1)} / {f(k == 'objet', s3)} || rectangles : {f(k == 'rectangle', b)} / {f(k == 'rectangle', s1)} / {f(k == 'rectangle', s3)}"
    print(f"{m} · base {BASES[m]} · seuil {tau:.2f}\n   " + line(hors, "hors reserve, hors-pli") + "\n   " + line([it for it in d["items"] if it["res"]], "reserve"), flush=True)
    Ltr = d["L"].copy(); Mo = d["M"].copy()
    for it in d["items"]: Mo[it["i"]] = union(d, it, tau)
    Ltr[d["I"]] = np.where(Mo[d["I"]], 6.0, -6.0); Lte = d["Lt"].copy(); Mo = d["Mt"].copy()
    for t in d["tests"]: Mo[t["i"]] = union(d, t, tau, test=True)
    Lte[d["It"]] = np.where(Mo[d["It"]], 6.0, -6.0); tag = f"rang3-{BASES[m].replace('/', '-')}"
    np.savez_compressed(ME / f"pred_train_{tag}_{m}.npz", names=np.array([r["image_name"] for r in d["rows"]]), logits=Ltr.astype(np.float16))
    np.savez_compressed(ME / f"pred_test_{tag}_{m}.npz", names=np.array([r["image_name"] for r in te[m]]), logits=Lte.astype(np.float16)); print(f"   ecrit pred_*_{tag}_{m}.npz", flush=True)
