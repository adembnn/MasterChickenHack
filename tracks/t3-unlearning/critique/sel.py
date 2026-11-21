"""Critique T3 : sélecteurs indépendants des images que le gold rate.
Aucune donnée externe, aucun entraînement du modèle. Sorties : sel.csv, sel_summary.json."""
import json, numpy as np, torch, timm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from scipy.stats import spearmanr

D = "cuda"; torch.manual_seed(0); np.random.seed(0)
def build():
    return timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=100, init_values=1e-4)
sd = torch.load("data/model.pt", map_location="cpu", weights_only=True)
m = build(); m.load_state_dict(sd, strict=True); m = m.to(D).eval()
X = torch.from_numpy(np.load("data/forget_images.npz")["images"])
N = len(X)
def prep(u): return ((u.to(D).permute(0, 3, 1, 2).float() / 255.0) - 0.5) / 0.5

feats = [[] for _ in range(12)]; logits = []
hooks = [blk.register_forward_hook(lambda mod, i, o, b=b: feats[b].append(o[:, 0].float().cpu())) for b, blk in enumerate(m.blocks)]
with torch.no_grad():
    for i in range(0, N, 128):
        with torch.autocast("cuda", dtype=torch.float16):
            logits.append(m(prep(X[i:i+128])).float().cpu())
for h in hooks: h.remove()
Z = torch.cat(logits); y = Z.argmax(1).numpy()
F = [torch.cat(f) for f in feats]
top2 = Z.topk(2, 1); margin = (top2.values[:, 0] - top2.values[:, 1]).numpy(); second = top2.indices[:, 1].numpy()
print("orig acc proxy classes", len(np.unique(y)), "margin mean", margin.mean())

# 1) kNN leave-one-out par bloc (cosinus, k=10) -> profondeur de prédiction (Baldock et al. 2021)
K = 10; knn_pred = np.zeros((12, N), int)
for b in range(12):
    f = torch.nn.functional.normalize(F[b].to(D), dim=1)
    S = f @ f.T; S.fill_diagonal_(-2)
    idx = S.topk(K, 1).indices.cpu().numpy()
    lab = y[idx]
    for n in range(N):
        knn_pred[b, n] = np.bincount(lab[n], minlength=100).argmax()
ok = knn_pred == y[None]
depth = np.full(N, 12)
for n in range(N):
    for b in range(11, -1, -1):
        if not ok[b, n]: depth[n] = b + 1; break
    else: depth[n] = 0
knn_acc = ok.mean(1)
# classe fausse = prédiction kNN au dernier bloc où elle diffère
knn_wrong = np.array([knn_pred[depth[n]-1, n] if depth[n] > 0 else -1 for n in range(N)])
print("kNN LOO acc par bloc", np.round(knn_acc, 3).tolist())

# 2) sonde logistique hors pli au bloc 8 (recoupe oofA_b8 de t3-poids : 0,693)
f8 = F[8].numpy(); f8 = (f8 - f8.mean(0)) / (f8.std(0) + 1e-6)
P = np.zeros((N, 100))
for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(f8, y):
    clf = LogisticRegression(max_iter=300, C=0.05).fit(f8[tr], y[tr])
    P[np.ix_(te, clf.classes_)] = clf.predict_proba(f8[te])
oof_p = P[np.arange(N), y]; oof_pred = P.argmax(1)
oof_loss = -np.log(np.clip(oof_p, 1e-12, 1))
print("oof b8 acc", (oof_pred == y).mean(), "loss", oof_loss.mean(), oof_loss.std())

# 3) sensibilité au bruit dans les poids (blocs 6-11) : part des tirages où la classe change
base = {k: v.clone() for k, v in m.state_dict().items()}
flip = np.zeros(N); draws = 0; res = {}
for sig in [0.02, 0.04, 0.06]:
    fr = np.zeros(N)
    for s in range(6):
        g = torch.Generator(device=D).manual_seed(1000 * s + int(sig * 1000))
        with torch.no_grad():
            for k, p in m.named_parameters():
                if k.startswith("blocks.") and int(k.split(".")[1]) >= 6 and p.ndim == 2:
                    p.add_(torch.randn(p.shape, generator=g, device=D) * sig * base[k].std())
            out = []
            for i in range(0, N, 256):
                with torch.autocast("cuda", dtype=torch.float16):
                    out.append(m(prep(X[i:i+256])).argmax(1).cpu())
            fr += (torch.cat(out).numpy() != y)
            m.load_state_dict(base)
    res[sig] = (fr / 6).mean(); flip += fr; draws += 6
    print("sigma", sig, "flip rate", res[sig])
flip /= draws

df = dict(margin=margin, depth=depth, oof_loss=oof_loss, noise_flip=flip)
names = list(df)
rho = {f"{a}~{b}": float(spearmanr(df[a], df[b]).correlation) for i, a in enumerate(names) for b in names[i+1:]}
# recouvrement des 20 % « les plus durs » de chaque sélecteur
hard = dict(margin=np.argsort(margin)[:700], depth=np.argsort(-depth + 1e-6*np.random.rand(N))[:700],
            oof_loss=np.argsort(-oof_loss)[:700], noise_flip=np.argsort(-flip + 1e-6*np.random.rand(N))[:700])
ov = {f"{a}&{b}": len(set(hard[a]) & set(hard[b])) / 700 for i, a in enumerate(names) for b in names[i+1:]}
# accord sur la classe fausse
wc = {"knn_vs_second": float(np.mean(knn_wrong[depth > 8] == second[depth > 8])),
      "oof_vs_second": float(np.mean(oof_pred[oof_pred != y] == second[oof_pred != y])),
      "knn_vs_oof(both wrong)": float(np.mean((knn_wrong == oof_pred)[(depth > 8) & (oof_pred != y)]))}
summ = dict(knn_acc=knn_acc.tolist(), depth_hist=np.bincount(depth, minlength=13).tolist(),
            oof_acc=float((oof_pred == y).mean()), oof_loss=[float(oof_loss.mean()), float(oof_loss.std())],
            noise_flip_mean=res, spearman=rho, top20_overlap=ov, wrong_class_agreement=wc)
print(json.dumps(summ, indent=1))
json.dump(summ, open("sel_summary.json", "w"), indent=1, default=float)
np.savetxt("sel.csv", np.c_[np.arange(N), y, second, margin, depth, knn_wrong, oof_pred, oof_loss, flip],
           delimiter=",", fmt=["%d", "%d", "%d", "%.4f", "%d", "%d", "%d", "%.4f", "%.4f"],
           header="idx,orig_pred,orig_second,margin,pred_depth,knn_wrong_class,oof_b8_pred,oof_b8_loss,noise_flip", comments="")
