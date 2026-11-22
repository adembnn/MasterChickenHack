"""Assemblage du track 1 : lit toutes les predictions deposees par les sessions (voir asm.discover), mesure chacune (mIoU officiel,
reserve = index % 5 == 0), regle seuil et post-traitement HORS reserve, choisit en validation croisee, par filigrane et par population
(pleine / partielle, reconnue par le taux de bits justes sur l'image entiere), la meilleure source ou un melange, ecrit la soumission au
format officiel, la relit et la re-note sur le train.
Usage : python assemble.py --n 4          -> submission_candidat_4.pt, assemble_4.json, t1-tableau.md
Options : --only REGEX / --skip REGEX (sources), --no-ens (pas de melange), --no-rect, --no-snap, --no-refit."""
import sys, json, argparse, time, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); import asm; from asm import *
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, required=True); ap.add_argument("--only", default=""); ap.add_argument("--skip", default="")
ap.add_argument("--no-ens", action="store_true"); ap.add_argument("--no-rect", action="store_true"); ap.add_argument("--no-snap", action="store_true")
ap.add_argument("--no-refit", action="store_true"); ap.add_argument("--boxes", default="", help="REGEX sur le chemin des fichiers de boites (les memes familles pour le train et le test)"); ap.add_argument("--force", default="", help="filigrane=source[,filigrane=source] : source imposee"); ap.add_argument("--ens-gain", type=float, default=0.01); a = ap.parse_args()
OPT.update(rect=not a.no_rect, snap=not a.no_snap, only=a.only, skip=a.skip, boxes=a.boxes)
V3 = {"pixelseal": "pixelseal/pred", "trustmark": "reel/mixr", "stegastamp": "reel/pred"}   # recette de la soumission v6 (public 0,706) : ces sources au seuil 0
FORCE = dict(kv.split("=", 1) for kv in a.force.split(",") if "=" in kv)
def log(*x): print(time.strftime("%H:%M:%S"), *x, file=sys.stderr, flush=True)

t00 = time.time(); tr, te, S, notes = discover(); P = populations(tr, te)
T = dict(n=a.n, heure=time.strftime("%Y-%m-%d %H:%M"), notes=notes, pop={m: {k: v for k, v in P[m].items() if k not in ("tr", "te")} for m in METHODS}, sources={}, choix={}, score={})
final_tr, final_te, honest, v3sc = {}, {}, {}, {}
for m in METHODS:
    rows = tr[m]; G = np.stack([r["gt"] for r in rows]); res = np.array([r["reserve"] for r in rows]); full_true = np.array([r["full"] for r in rows]); pf = P[m]["tr"]; fold = np.array([r["fold"] for r in rows])
    Ctr, bf = load_side([r["image_name"] for r in rows], "train", m); Cte, bf2 = load_side([r["image_name"] for r in te[m]], "test", m)
    nb = lambda C_: int(sum(c["box"] is not None for c in C_)); T.setdefault("boites", {})[m] = dict(fichiers=bf + bf2, train=nb(Ctr), test=nb(Cte))
    if nb(Ctr) and not nb(Cte): log(m, "ATTENTION : boites sur le train mais pas sur le test -> fusion boite desactivee"); [c.update(box=None) for c in Ctr]
    if not S[m]: sys.exit(f"aucune source complete pour {m}")
    Z = {}; OOF = {}; T0 = {}                                                              # OOF[name] : les predictions du train sont-elles hors-pli ? T0 : le seuil 0 d'origine, dans l'echelle centree-reduite
    for name, (Ltr, Lte, hh, oo) in S[m].items():
        mu, sd = float(Ltr[~res].mean()), float(Ltr[~res].std()) + 1e-6; Z[name] = ((Ltr - mu) / sd, (Lte - mu) / sd); OOF[name] = oo; T0[name] = (0.0 - mu) / sd
    T["sources"][m] = {}; T["choix"][m] = {}; cand = {True: {}, False: {}}
    sub = lambda X, k: None if X is None else X[k]
    for name in Z:                                                                         # chaque source seule, reglee hors reserve, par population predite
        M = np.zeros_like(G); pp = {}; modes = None; Zf = Z[name][0]
        for full in (True, False):
            I = pf == full; fit = I & ~res; ev = I & res; C = None if full else Ctr
            rr = lambda q: round(float(miou(apply(Zf[ev], q, full, sub(C, ev)), G[ev]).mean()), 4) if ev.any() else None
            if not OOF[name]:                                                              # modele qui a vu les images hors reserve : rien a regler dessus, on garde son seuil 0 ; seule la reserve le note
                p = dict(mode="seuil", sigma=0, t=T0[name], clamp=False, area=0); M[I] = apply(Zf[I], p, full); pp[full] = p
                cand[full][name] = dict(w={name: 1}, p=p, fit=None, cv=None, noof=True, res=float(miou(M[ev], G[ev]).mean()) if ev.any() else None); continue
            p, sc = tune_seuil(Zf[fit], G[fit], full); use_rect = use_snap = False
            if not full:                                                                   # seuil, rectangle ajuste, accrochage de forme : diagnostic sur la reserve
                modes = dict(seuil=dict(fit=round(sc, 4), res=rr(p)))
                if OPT["rect"]:
                    pr, sr = tune_rect(Zf[fit], G[fit]); modes["rect"] = dict(fit=round(sr, 4), res=rr(pr)); use_rect = sr > sc - 0.01
                    if sr > sc + 0.005: p, sc = pr, sr
                if OPT["snap"]:
                    pn, sn = tune_snap(Zf[fit], G[fit], sub(C, fit), p, sc); modes["forme"] = dict(fit=round(sn, 4), res=rr(pn), tau=pn.get("snap"), formes=pn.get("snapset")); use_snap = sn > sc + 0.005
                    if use_snap: p, sc = pn, sn
                pb, sb = tune_box(Zf[fit], G[fit], sub(C, fit), p, sc)
                if pb.get("box") is not None: modes["boite"] = dict(fit=round(sb, 4), res=rr(pb), z0=pb["box"])
                if sb > sc + 0.005: p, sc = pb, sb
            M[I] = apply(Zf[I], p, full, sub(C, I)); pp[full] = p
            cand[full][name] = dict(w={name: 1}, p=p, fit=sc, cv=cv(Zf[fit], G[fit], fold[fit], full, sub(C, fit), rect=use_rect, snap=use_snap), res=float(miou(M[ev], G[ev]).mean()) if ev.any() else None)
        sc_ = miou(M, G); raw = miou(S[m][name][0] > 0, G)
        T["sources"][m][name] = dict(oof=OOF[name], cv=(cand[True][name]["cv"], cand[False][name]["cv"]), brut=report(raw, rows), regle=report(sc_, rows), heure=S[m][name][2], modes=modes, recette={POPN[k]: v for k, v in pp.items()})
        if name == V3[m]: v3sc[m] = raw
        log(m, name, "reserve brut", T["sources"][m][name]["brut"]["reserve"], "regle", T["sources"][m][name]["regle"]["reserve"], modes)
    for full in (True, False):                                                             # melange glouton de sources (moyenne de logits centres-reduits), choisi en validation croisee
        I = pf == full; fit = I & ~res; ev = I & res; C = None if full else Ctr
        cst = np.full_like(G, full); k0 = float(miou(cst[fit], G[fit]).mean()) if fit.any() else 0.0
        cand[full]["constante"] = dict(w={}, p=dict(mode="seuil", sigma=0, t=-1e9 if full else 1e9, clamp=False, area=0), fit=k0, cv=k0, res=float(miou(cst[ev], G[ev]).mean()) if ev.any() else None)
        mix = lambda w, j=0: sum(Z[q][j] * c for q, c in w.items()) / sum(w.values())
        Zo = [k for k in Z if OOF[k]]
        if not a.no_ens and len(Zo) > 1:
            best1 = max(Zo, key=lambda k: cand[full][k]["cv"]); w = {best1: 1}; cur = cv(Z[best1][0][fit], G[fit], fold[fit], full, rect=False, snap=False)
            for _ in range(3):
                trial = []
                for k in Zo:
                    w2 = dict(w); w2[k] = w2.get(k, 0) + 1; trial.append((cv(mix(w2)[fit], G[fit], fold[fit], full, rect=False, snap=False), k))
                sc, k = max(trial)
                if sc < cur + 0.008: break
                w[k] = w.get(k, 0) + 1; cur = sc
            if len(w) > 1:
                E = mix(w); p, sc = tune(E[fit], G[fit], full, sub(C, fit))
                cand[full]["melange"] = dict(w=w, p=p, fit=sc, cv=cv(E[fit], G[fit], fold[fit], full, sub(C, fit), rect=p.get("mode") == "rect", snap=p.get("snap") is not None),
                                             res=float(miou(apply(E[ev], p, full, sub(C, ev)), G[ev]).mean()) if ev.any() else None)
        pick = max((k for k in cand[full] if k != "melange" and cand[full][k]["cv"] is not None), key=lambda k: cand[full][k]["cv"])
        if "melange" in cand[full] and cand[full]["melange"]["cv"] >= cand[full][pick]["cv"] + a.ens_gain: pick = "melange"
        no = [k for k in Z if not OOF[k] and cand[full][k]["res"] is not None]              # sources pas hors-pli : la meilleure sur la reserve, retenue seulement si elle bat le choix de 0,03 (seule mesure possible)
        if no and cand[full][pick]["res"] is not None:
            k = max(no, key=lambda q: cand[full][q]["res"])
            if cand[full][k]["res"] >= cand[full][pick]["res"] + 0.03: pick = k
        if FORCE.get(m) in cand[full]: pick = FORCE[m]                                     # source imposee (PixelSeal : on garde tel quel ce qui est deja soumis)
        T["choix"][m][POPN[full]] = dict(choisi=pick, **cand[full][pick], candidats={k: dict(fit=None if v["fit"] is None else round(v["fit"], 4), cv=None if v["cv"] is None else round(v["cv"], 4), res=None if v["res"] is None else round(v["res"], 4)) for k, v in cand[full].items()})
        log(m, POPN[full], "choix", pick, T["choix"][m][POPN[full]]["candidats"])
    # masques : (1) honnetes = recette reglee hors reserve ; (2) finaux = meme choix, seuil re-regle sur les 170, appliques au test
    Mh = np.zeros_like(G); Mf = np.zeros_like(G); Mt = np.zeros((len(te[m]), 128, 128), bool)
    for full in (True, False):
        c = T["choix"][m][POPN[full]]; I = pf == full; It = P[m]["te"] == full; C = None if full else Ctr
        if c["w"]:
            Etr = sum(Z[q][0] * k for q, k in c["w"].items()) / sum(c["w"].values()); Ete = sum(Z[q][1] * k for q, k in c["w"].items()) / sum(c["w"].values())
        else: Etr = np.zeros(G.shape, np.float32); Ete = np.zeros((len(te[m]), 128, 128), np.float32)
        Mh[I] = apply(Etr[I], c["p"], full, sub(C, I))
        p2 = c["p"] if (a.no_refit or not c["w"] or c.get("noof")) else retune_t(Etr[I], G[I], full, c["p"], sub(C, I)); c["p_final"] = p2
        Mf[I] = apply(Etr[I], p2, full, sub(C, I)); Mt[It] = apply(Ete[It], p2, full, None if full else Cte[It])
    honest[m] = miou(Mh, G); final_tr[m] = Mf; final_te[m] = Mt
    np.savez_compressed(ME / f"candidat_{a.n}_honnete_{m}.npz", names=np.array([r["image_name"] for r in rows]), masks=Mh, reserve=res)   # masques du train avec la recette reglee hors reserve (pour le verificateur)
    T["score"][m] = dict(honnete=report(honest[m], rows), final_train=report(miou(Mf, G), rows), v3=report(v3sc[m], rows) if m in v3sc else None,
                         frac_train=dict(pleines=round(float(Mf[pf].mean()), 3), partielles=round(float(Mf[~pf].mean()), 3), vraie_pleines=round(float(G[full_true].mean()), 3), vraie_partielles=round(float(G[~full_true].mean()), 3)),
                         frac_test=dict(pleines=round(float(Mt[P[m]["te"]].mean()), 3) if P[m]["te"].any() else None, partielles=round(float(Mt[~P[m]["te"]].mean()), 3) if (~P[m]["te"]).any() else None))
allrows = [r for m in METHODS for r in tr[m]]
T["score"]["ensemble"] = dict(honnete=report(np.concatenate([honest[m] for m in METHODS]), allrows), final_train=report(np.concatenate([miou(final_tr[m], np.stack([r["gt"] for r in tr[m]])) for m in METHODS]), allrows),
                              v3=report(np.concatenate([v3sc[m] for m in METHODS]), allrows) if len(v3sc) == 3 else None)

# ---------------------------------------------------------------- ecriture, relecture, re-notation officielle
def write(path, rows_by_m, masks_by_m, order):
    d = {r["image_name"]: masks_by_m[m][i] for m in METHODS for i, r in enumerate(rows_by_m[m])}
    torch.save({"image_names": list(order), "masks": torch.from_numpy(np.stack([d[n] for n in order]).astype(np.uint8))}, path)
test_order = [r["image_name"] for r in load_test()]; train_order = [r["image_name"] for r in load_train()]
out = ME / f"assemblage_{a.n}.pt"; chk = ME / f"controle_train_{a.n}.pt"   # fichier INTERMEDIAIRE : seuls les submission_candidat_<n>.pt ecrits par compose.py sont des candidats
write(out, te, final_te, test_order); write(chk, tr, final_tr, train_order)
sys.path.insert(0, str(HERE / "kit")); from wmloc.utils import miou_image
z = torch.load(chk, weights_only=True); gts = {r["image_name"]: r["gt"] for r in allrows}; off = float(np.mean([miou_image(mk.numpy(), gts[n]) for n, mk in zip(z["image_names"], z["masks"])]))
s = torch.load(out, weights_only=True); ok = (s["masks"].dtype == torch.uint8 and tuple(s["masks"].shape) == (1020, 128, 128) and set(s["image_names"]) == set(test_order) and len(set(s["image_names"])) == 1020 and int(s["masks"].max()) <= 1)
T["fichier"] = dict(chemin=str(out), octets=out.stat().st_size, controle_train_officiel=round(off, 4), attendu=T["score"]["ensemble"]["final_train"]["tous"], format_ok=bool(ok))
(ME / f"assemble_{a.n}.json").write_text(json.dumps(T, indent=1, default=str))

# ---------------------------------------------------------------- tableau de bord (markdown)
f4 = lambda v: "-" if v is None else f"{v:.3f}".replace(".", ",")
L = [f"# T1 : tableau de bord des méthodes (candidat {a.n}, {T['heure']})", "",
     "mIoU officiel sur le vrai train. **Réserve** = images d'index % 5 == 0 (30 à 39 par filigrane), jamais utilisées pour régler : c'est le chiffre honnête (bruit ±0,03).",
     "« brut » = prédiction telle que déposée (seuil 0). « réglé » = post-traitement réglé hors réserve par `assemble.py` : lissage, seuil par population, fourchette de fraction, nettoyage ; pour les partielles, rectangle ajusté ou accrochage de forme (masque SAM ou rectangle le plus proche du masque).",
     "Pleines / partielles = score sur la réserve, par vraie population. La validation croisée (4 plis hors réserve) sert à choisir ; la réserve sert à vérifier.",
     "« pas hors-pli » = modèle qui a vu les images hors réserve à l'entraînement : aucun réglage possible, son seuil 0 est gardé et seule la réserve le note.", ""]
L += [f"## Candidat {a.n} contre la soumission v6 (public 0,706)", "", "| | v6, réserve | Candidat, réserve (honnête) | Pleines (rés.) | Partielles (rés.) | Hors réserve | Train entier après re-réglage du seuil (optimiste) |", "|---|---|---|---|---|---|---|"]
for m in METHODS + ["ensemble"]:
    h = T["score"][m]["honnete"]; fz = T["score"][m]["final_train"]; v = T["score"][m].get("v3")
    L.append(f"| {m} | {f4(v['reserve']) if v else '-'} | **{f4(h['reserve'])}** | {f4(h['res_pleines'])} | {f4(h['res_partielles'])} | {f4(h['hors_reserve'])} | {f4(fz['tous'])} |")
L += ["", f"Fichier : `{out}` ({out.stat().st_size:,} octets, uint8 [1020,128,128]). Contrôle : le même écrivain appliqué au train puis re-noté avec `wmloc.utils.miou_image` = {off:.4f} (attendu {T['fichier']['attendu']:.4f}).", ""]
for m in METHODS:
    L += [f"## {m}", "", "| Source (session/méthode) | Réserve brut | Réserve réglé | Pleines (rés.) | Partielles (rés.) | Validation croisée hors réserve (pleines / partielles) | Déposée |", "|---|---|---|---|---|---|---|"]
    for name, v in sorted(T["sources"][m].items(), key=lambda kv: -kv[1]["regle"]["reserve"]):
        L.append(f"| {name}{' (= v6)' if name == V3[m] else ''}{'' if v['oof'] else ' (pas hors-pli)'} | {f4(v['brut']['reserve'])} | {f4(v['regle']['reserve'])} | {f4(v['regle']['res_pleines'])} | {f4(v['regle']['res_partielles'])} | {f4(v['cv'][0])} / {f4(v['cv'][1])} | {v['heure']} |")
    h = T["score"][m]["honnete"]; c = T["choix"][m]
    L += [f"| **CHOIX DU CANDIDAT {a.n}** | | **{f4(h['reserve'])}** | {f4(h['res_pleines'])} | {f4(h['res_partielles'])} | | |", ""]
    md = [f"{name} : seuil {f4(v['modes']['seuil']['res'])}" + (f", rectangle {f4(v['modes']['rect']['res'])}" if "rect" in v["modes"] else "") + (f", forme {f4(v['modes']['forme']['res'])}" if "forme" in v["modes"] else "") + (f", boîte du décodeur {f4(v['modes']['boite']['res'])}" if "boite" in v["modes"] else "") for name, v in T["sources"][m].items() if v.get("modes")]
    if md: L += ["Images partielles de la réserve, selon le post-traitement (chaque étape s'ajoute à la précédente) : " + " ; ".join(md) + ".", ""]
    bx = T["boites"][m]; L += [f"Boîtes de la recherche par décodeur lues : {bx['train']} images de train, {bx['test']} de test" + (f" ({', '.join(bx['fichiers'])})" if bx["fichiers"] else " (aucun fichier déposé)") + ".", ""]
    dsc = lambda c: f"`{c['choisi']}` {json.dumps(c['w'])}, recette {json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in c['p'].items()})}"
    L += [f"Choix : pleines = {dsc(c['pleines'])} ; partielles = {dsc(c['partielles'])}. "
          f"Population reconnue par le taux de bits justes de l'image entière (seuil {P[m]['th']:.2f}) : exactitude sur la réserve {P[m]['exact_res']:.3f}, {P[m]['part_test_pleines']:.0%} du test classé « pleines ».", ""]
if notes: L += ["Fichiers ignorés : " + " ; ".join(notes) + ".", ""]
(ME / "t1-tableau.md").write_text("\n".join(L), encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8"); print("\n".join(L)); print("\nFORMAT_OK", ok, "| controle officiel train", round(off, 4), "attendu", T["fichier"]["attendu"], "|", round(time.time() - t00), "s")
for m in METHODS: print("FRACTIONS", m, T["score"][m]["frac_train"], "test", T["score"][m]["frac_test"], "pop", T["pop"][m])
