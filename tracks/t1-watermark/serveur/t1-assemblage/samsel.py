"""Idee d'Adem, version mesuree : pour une image partielle dont la zone filigranee est un OBJET, la bonne reponse est (presque) un masque SAM.
Ce script mesure sur le train : (1) le plafond si on choisissait toujours le meilleur masque SAM ; (2) ce que donnent des regles de choix
qui n'utilisent pas la verite (decodeur sur la region seule, chute quand on efface la region, carte d'une autre methode)."""
import sys, glob, json, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); from t1lib import *
tag = sys.argv[1] if len(sys.argv) > 1 else "L"
rows = {r["image_name"]: r for r in load_train()}; sh = {r["name"]: r for r in json.loads((ME / "shapes_oracle.json").read_text())}
maps = {}
for m in METHODS:
    for src in ("unetv1", "occl64"):
        f = ME / f"pred_train_{src}_{m}.npz"
        if f.exists():
            z = np.load(f); L = z["logits"].astype(np.float32); L = (L - L.mean()) / (L.std() + 1e-6); maps.update({(src, n): L[i] for i, n in enumerate(z["names"].tolist())})
R = []
for f in sorted(glob.glob(str(ME / f"samcand{tag}_train_*.npz"))):
    z = np.load(f)
    for i, n in enumerate(z["names"].tolist()):
        r = rows[n]; nv = int(z["nvalid"][i])
        if r["full"] or nv == 0: continue
        M = np.unpackbits(z["masks"][i, :nv], axis=1).reshape(nv, 128, 128).astype(bool); F = z["feats"][i, :nv]; g = r["gt"]
        ok = M.reshape(nv, -1).sum(1) > 0; M, F = M[ok], F[ok]; nv = len(M)
        sc = miou(M, np.broadcast_to(g, M.shape)); iou = np.array([(mk & g).sum() / max((mk | g).sum(), 1) for mk in M])
        sel = dict(garder_marge=F[:, 3], garder_bits=F[:, 4], effacer_marge=F[:, 5], effacer_bits=F[:, 6], garder_plus_effacer=F[:, 3] + F[:, 5],
                   bits_x_racine_aire=(F[:, 4] - 0.5) * np.sqrt(F[:, 0]), marge_sur_racine_aire=F[:, 3] / np.sqrt(F[:, 0]))
        for src in ("unetv1", "occl64"):
            if (src, n) in maps:
                Z = maps[(src, n)]; t = np.quantile(Z, 0.9); sel[f"carte_{src}"] = np.array([(Z[mk] - t).sum() for mk in M])
        kind = "rectangle" if sh[n]["rect"] > 0.98 else "objet"
        R.append(dict(m=r["method"], kind=kind, res=r["reserve"], base=float(miou(np.zeros((1, 128, 128), bool), g[None])[0]), oracle=float(sc.max()), best_iou=float(iou.max()), nv=nv,
                      **{k: float(sc[int(np.argmax(v))]) for k, v in sel.items()}, **{"hit_" + k: bool(iou[int(np.argmax(v))] > 0.5) for k, v in sel.items()},
                      **{"top_" + k: float(np.max(v)) for k, v in sel.items()}))
keys = [k for k in ("garder_marge", "garder_bits", "effacer_marge", "effacer_bits", "garder_plus_effacer", "bits_x_racine_aire", "marge_sur_racine_aire", "carte_unetv1", "carte_occl64")]
print(f"images partielles avec candidats SAM : {len(R)} | masques par image (mediane) {np.median([r['nv'] for r in R]):.0f}")
print("mIoU officiel si on rend UN masque SAM choisi par la regle ; entre parentheses : part des images ou le masque choisi recouvre la vraie zone (IoU > 0,5)")
for m in METHODS + ["tous"]:
    for kind in ("objet", "rectangle"):
        S = [r for r in R if (m == "tous" or r["m"] == m) and r["kind"] == kind]
        if not S: continue
        line = f"{m:10s} {kind:9s} n={len(S):3d} | tout-propre {np.mean([r['base'] for r in S]):.3f} | plafond SAM {np.mean([r['oracle'] for r in S]):.3f} (meilleur IoU > 0,5 : {np.mean([r['best_iou'] > .5 for r in S]):.2f})"
        for k in keys:
            v = [r[k] for r in S if k in r]
            if v: line += f" | {k} {np.mean(v):.3f} ({np.mean([r['hit_' + k] for r in S if k in r]):.2f})"
        print(line)
(ME / f"samsel_{tag}.json").write_text(json.dumps(R))
