"""Entraîne le classifieur de paires sur les mondes d'entraînement, l'évalue sur des mondes à graines neuves
(LoRA, system prompts, personas et filigranes jamais vus), puis l'applique aux vrais ref avec la règle FIGÉE :
P(même modèle) par LightGBM, partition par lien complet coupée à 0,5.

python train_eval.py --train feats/smol360_train_s1 ... --val feats/smol360_val_s101 ... --ref feats_ref/ref_bloc0 --k 25
"""
import argparse, glob, json, os, random, subprocess, itertools
from collections import Counter, defaultdict
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import pairs as P

KINDS_S = ["plain", "system", "persona", "watermark", "temp1.0"]


def world_files(stem, jdir):
    name = os.path.basename(stem)
    return stem + ".npz", os.path.join(jdir, name + ".jsonl")


DROP = []


def build(world, k, names=None):
    F = [P.guichet_features(g, k) for g in world]
    A = [P.guichet_aux(g, k) for g in world]
    names = names or [n for n in sorted(F[0]) if not any(n.startswith(d) for d in DROP)]
    return F, A, names


def pairs_of(world, F, A, names, idx=None):
    idx = idx if idx is not None else range(len(world))
    X, y, meta = [], [], []
    for i, j in itertools.combinations(idx, 2):
        X.append(P.pair_features(F[i], F[j], A[i], A[j], names))
        y.append(int(world[i]["variant"] == world[j]["variant"]))
        meta.append((i, j))
    return np.array(X), np.array(y), meta


def vtype(v):
    if v.startswith("lora:"):
        return "lora@" + v.split("@")[1]
    return v


def practice_like(world, rnd):
    """Tire 9 guichets avec la structure du corrigé : 5 copies de la base (plain, system, persona, watermark, T=1,0),
    fp8, et une même LoRA aux forces 0,3 / 0,7 / 1,5."""
    by = defaultdict(list)
    for i, g in enumerate(world):
        by[(g["variant"], g["kind"])].append(i)
    pick = []
    for kd in KINDS_S:
        c = by.get(("base", kd)) or []
        if not c:
            return None
        pick.append(rnd.choice(c))
    fp8 = [i for i, g in enumerate(world) if g["variant"].startswith("fp8")]
    loras = sorted({g["variant"].split("@")[0] for g in world if g["variant"].startswith("lora:")})
    if not fp8 or not loras:
        return None
    pick.append(rnd.choice(fp8))
    ln = rnd.choice(loras)
    for s in ("0.3", "0.7", "1.5"):
        c = [i for i, g in enumerate(world) if g["variant"] == f"{ln}@{s}"]
        pick.append(rnd.choice(c))
    return pick


def random_set(world, rnd, size=11):
    """Lot de type ep : 4 à 6 copies de la base (déguisements variés), le reste en variantes distinctes."""
    base = [i for i, g in enumerate(world) if g["variant"] == "base"]
    other = {}
    for i, g in enumerate(world):
        if g["variant"] != "base":
            other.setdefault(g["variant"], []).append(i)
    nb = min(len(base), rnd.randint(4, 6))
    vs = rnd.sample(sorted(other), min(len(other), size - nb))
    return rnd.sample(base, nb) + [rnd.choice(other[v]) for v in vs]


SETREL = [False]
CROSS = {}


def set_pairs(world, wid, F, A, names, pick):
    """Caractéristiques de paires dans un lot ; relatives à la médiane du lot si SETREL."""
    if SETREL[0]:
        Fs = P.set_relative(F, pick, names)
        nm = names + ["rel_" + n for n in names]
    else:
        Fs, nm = F, names
    X, keys = [], []
    for a, b in itertools.combinations(range(len(pick)), 2):
        i, j = pick[a], pick[b]
        ck = (wid, min(i, j), max(i, j))
        if ck not in CROSS:
            CROSS[ck] = P.cross_features(A[i], A[j])
        X.append(P.pair_features(Fs[i], Fs[j], None, None, nm, cross=CROSS[ck]))
        keys.append((a, b))
    return np.array(X), keys


def score_sets(world, F, A, names, model, rnd, n_sets=200, thr=0.5, errs=None, sampler=None, wid=None):
    sampler = sampler or practice_like
    wid = wid if wid is not None else id(world)
    mccs = []
    for _ in range(n_sets):
        pick = sampler(world, rnd)
        if pick is None:
            break
        n = len(pick); M = np.eye(n)
        X, keys = set_pairs(world, wid, F, A, names, pick)
        pr = model.predict(X)
        for (a, b), v in zip(keys, pr):
            M[a, b] = M[b, a] = v
        lab = P.complete_linkage(M, thr)
        truth = [world[i]["variant"] for i in pick]
        mccs.append(P.mcc_pairs(truth, lab))
        if errs is not None:
            for a, b in itertools.combinations(range(n), 2):
                t = truth[a] == truth[b]; q = lab[a] == lab[b]
                if t != q:
                    ga, gb = world[pick[a]], world[pick[b]]
                    key = " / ".join(sorted([f"{vtype(ga['variant'])}:{ga['kind']}", f"{vtype(gb['variant'])}:{gb['kind']}"]))
                    errs[key + (" (dit pareil)" if q else " (dit différent)")] += 1
    return float(np.mean(mccs)) if mccs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--ref", default="")
    ap.add_argument("--ref_jsonl", default="")
    ap.add_argument("--jdir", default="worlds")
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--thr", default="0.5", help="seuil, ou 'auto' = choisi sur les jumeaux de validation seulement")
    ap.add_argument("--out", default="out")
    ap.add_argument("--answers", default="practice_answers.csv")
    ap.add_argument("--tag", default="")
    ap.add_argument("--setrel", action="store_true", help="caractéristiques relatives à la médiane du lot")
    ap.add_argument("--n_train_sets", type=int, default=300)
    ap.add_argument("--drop", default="", help="préfixes de caractéristiques retirées, séparés par des virgules")
    a = ap.parse_args()
    DROP.extend([d for d in a.drop.split(",") if d])
    os.makedirs(a.out, exist_ok=True)
    rnd = random.Random(0)
    names = None
    Xtr, ytr = [], []
    SETREL[0] = a.setrel
    for st in a.train:
        w = P.load_world(*world_files(st, a.jdir))
        F, A, names = build(w, a.k, names)
        if a.setrel:   # paires tirées dans des lots comme au test (corrigé-like et ep-like)
            for t in range(a.n_train_sets):
                pick = (practice_like(w, rnd) if t % 2 == 0 else None) or random_set(w, rnd)
                X, keys = set_pairs(w, st, F, A, names, pick)
                Xtr.append(X); ytr.append(np.array([int(w[pick[i]]["variant"] == w[pick[j]]["variant"]) for i, j in keys]))
        else:
            X, y, _ = pairs_of(w, F, A, names)
            Xtr.append(X); ytr.append(y)
    Xtr, ytr = np.concatenate(Xtr), np.concatenate(ytr)
    pn = P.pair_names(names + (["rel_" + n for n in names] if a.setrel else []))
    wpos = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)
    model = lgb.train({"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 40,
                       "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
                       "scale_pos_weight": wpos, "verbose": -1, "seed": 1, "num_threads": 6},
                      lgb.Dataset(Xtr, ytr, feature_name=pn), num_boost_round=400)
    res = {"k": a.k, "n_train_pairs": int(len(ytr)), "n_pos": int(ytr.sum())}
    errs = Counter()
    vm, vm2 = [], []
    vworlds = []
    for st in a.val:
        w = P.load_world(*world_files(st, a.jdir))
        F, A, _ = build(w, a.k, names)
        vworlds.append((w, F, A))
    if a.thr == "auto":
        sweep = {}
        for t in (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6):
            sweep[t] = float(np.mean([score_sets(w, F, A, names, model, random.Random(1), n_sets=100, thr=t, wid=f"v{vi}")
                                      for vi, (w, F, A) in enumerate(vworlds)] +
                                     [score_sets(w, F, A, names, model, random.Random(2), n_sets=100, thr=t, sampler=random_set,
                                                 wid=f"v{vi}") for vi, (w, F, A) in enumerate(vworlds)]))
        res["thr_sweep_val"] = {str(t): round(v, 3) for t, v in sweep.items()}
        a.thr = max(sweep, key=sweep.get)
    a.thr = float(a.thr); res["thr"] = a.thr
    for vi, st in enumerate(a.val):
        w, F, A = vworlds[vi]
        m = score_sets(w, F, A, names, model, random.Random(3), errs=errs, thr=a.thr, wid=f"v{vi}")
        m2 = score_sets(w, F, A, names, model, random.Random(4), thr=a.thr, sampler=random_set, wid=f"v{vi}")
        res[f"val_{os.path.basename(st)}"] = {"mcc_practice_like": round(m, 3), "mcc_ep_like11": round(m2, 3)}
        vm.append(m); vm2.append(m2)
    res["val_mcc_mean"] = round(float(np.mean(vm)), 3)
    res["val_mcc_ep_like_mean"] = round(float(np.mean(vm2)), 3)
    res["val_errors_top"] = errs.most_common(12)
    imp = sorted(zip(pn, model.feature_importance("gain")), key=lambda t: -t[1])[:20]
    res["top_features"] = [(n, round(float(v))) for n, v in imp]
    if a.ref:
        w = P.load_world(a.ref + ".npz", a.ref_jsonl or None)
        F, A, _ = build(w, a.k, names)
        n = len(w); M = np.eye(n)
        X, keys = set_pairs(w, "ref", F, A, names, list(range(n)))
        for (i, j), v in zip(keys, model.predict(X)):
            M[i, j] = M[j, i] = v
        lab = P.complete_linkage(M, a.thr)
        csv = os.path.join(a.out, f"ref_k{a.k}{a.tag}.csv")
        with open(csv, "w") as f:
            f.write("endpoint,family,model\n")
            for g, l in zip(w, lab):
                f.write(f"{g['gid']},P,M{l}\n")
        res["ref_P"] = [[round(float(x), 2) for x in r] for r in M]
        res["ref_labels"] = lab
        if os.path.exists(a.answers):
            res["ref_score"] = subprocess.run(["python3", "score_practice.py", csv, a.answers], capture_output=True, text=True).stdout
    res["feature_names"] = names
    res["setrel"] = a.setrel
    model.save_model(os.path.join(a.out, f"model_k{a.k}{a.tag}.txt"))
    json.dump(res, open(os.path.join(a.out, f"res_k{a.k}{a.tag}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "ref_P"}, indent=1, ensure_ascii=False))
    if "ref_P" in res:
        for g, r in zip(w, res["ref_P"]):
            print(g["gid"], " ".join(f"{x:.2f}" for x in r))


if __name__ == "__main__":
    main()
