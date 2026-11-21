"""Caractéristiques par guichet et par paire, classifieur de paires (LightGBM), partition par lien complet.

Bibliothèque utilisée par train_eval.py. Tout est calculé sur les emplacements < k (budget 25, 50, 100 réponses).
"""
import json, re, math, itertools
from collections import Counter
import numpy as np

TEMPS = [0.6, 0.7, 0.8, 0.9, 1.0, 1.2]
QNAMES = ["fp8", "int8", "nf4", "alpaca_r16", "stories_r32", "gsm_r16"]
WORD = re.compile(r"[A-Za-z']+")


def load_world(npz_path, jsonl_path=None):
    """Retourne une liste de guichets : dict(gid, variant, kind, R (slots x cols), slots, texts{slot: str}, finish{slot})."""
    d = np.load(npz_path, allow_pickle=True)
    meta = json.load(open(npz_path.replace(".npz", ".meta.json")))
    cols = list(d["cols"])
    X, gi, sl = d["X"], d["gi"], d["ri"]
    texts = {}
    if jsonl_path:
        for l in open(jsonl_path):
            g = json.loads(l)
            pidx = g.get("prompt_idx", list(range(len(g["responses"]))))
            texts[g["gid"]] = ({p: r for p, r in zip(pidx, g["responses"]) if r is not None},
                               {p: f for p, f in zip(pidx, g.get("finish", ["stop"] * len(pidx)))})
    out = []
    for i, m in enumerate(meta):
        sel = gi == i
        t, f = texts.get(m["gid"], ({}, {}))
        out.append({"gid": m["gid"], "variant": m.get("variant", "?"), "kind": (m.get("disguise") or {}).get("kind", "?"),
                    "R": X[sel], "slots": sl[sel], "texts": t, "finish": f, "cols": cols})
    return out


def _z(S, c, a, b, v):
    den = S[c[v]]
    return (S[c[a]] + (S[c[b]] if b else 0.0)) / math.sqrt(den) if den > 0 else 0.0


def guichet_features(g, k):
    """Vecteur de caractéristiques d'un guichet, sur les emplacements < k."""
    c = {n: i for i, n in enumerate(g["cols"])}
    m = g["slots"] < k
    R, slots = g["R"][m], g["slots"][m]
    read = slots >= 3
    S = R[read].sum(0); Sa = R.sum(0)
    f = {}
    n = max(S[c["n"]], 1)
    f["tok_per_resp"] = S[c["n"]] / max(read.sum(), 1)
    for key in ("top1", "rank10", "lowp"):
        f[key] = S[c[key]] / n
    zs = []
    for T in TEMPS:
        f[f"z_{T}"] = _z(S, c, f"lp_{T}", f"H_{T}", f"V_{T}") / math.sqrt(n / 1000)
        f[f"lp_{T}"] = S[c[f"lp_{T}"]] / n
        zs.append(_z(S, c, f"lp_{T}", f"H_{T}", f"V_{T}"))
    f["H_1.0"] = S[c["H_1.0"]] / n
    # température effective : passage par zéro de z(T)
    tstar = TEMPS[-1]
    for i in range(len(TEMPS) - 1):
        if zs[i] <= 0 < zs[i + 1] or (i == 0 and zs[0] > 0):
            tstar = TEMPS[i] + (TEMPS[i + 1] - TEMPS[i]) * (-zs[i]) / (zs[i + 1] - zs[i]) if zs[i + 1] != zs[i] else TEMPS[i]
            break
    f["tstar"] = tstar
    for T in (0.8, 1.0):
        for b in range(3):
            f[f"zb{b}_{T}"] = _z(S, c, f"b{b}_r_{T}", None, f"b{b}_V_{T}") / math.sqrt(max(S[c[f"b{b}_n_{T}"]], 1) / 300)
        for q in QNAMES:
            num = S[c[f"{q}_d_{T}"]] - S[c[f"{q}_Ep_{T}"]]
            den = S[c[f"{q}_Eq_{T}"]] - S[c[f"{q}_Ep_{T}"]]
            f[f"lam_{q}_{T}"] = num / den if den > 0 else 0.0
            f[f"zq_{q}_{T}"] = num / math.sqrt(max(S[c[f"{q}_Vp_{T}"]], 1e-9)) / math.sqrt(n / 1000)
    # dispersion par réponse de la calibration à 0,8 (déguisement localisé vs modification diffuse)
    pr = (R[:, c["lp_0.8"]] + R[:, c["H_0.8"]]) / np.sqrt(np.maximum(R[:, c["V_0.8"]], 1e-6))
    f["zresp_std"] = float(np.std(pr[read])) if read.any() else 0.0
    f["zresp_fracneg"] = float((pr[read] < -2).mean()) if read.any() else 0.0
    f["zresp_min"] = float(np.min(pr[read])) if read.any() else 0.0
    for s, name in ((0, "extract"), (1, "identity")):
        r = R[slots == s]
        f[f"z_{name}"] = float(((r[:, c["lp_0.8"]] + r[:, c["H_0.8"]]) / np.sqrt(np.maximum(r[:, c["V_0.8"]], 1e-6))).mean()) if len(r) else 0.0
        f[f"n_{name}"] = float(r[:, c["n"]].mean()) if len(r) else 0.0
    # texte
    T_ = {s: t for s, t in g["texts"].items() if s < k}
    allt = [T_[s] for s in sorted(T_) if s >= 3]
    words = [w.lower() for t in allt for w in WORD.findall(t)]
    nw = max(len(words), 1)
    f["words_per_resp"] = len(words) / max(len(allt), 1)
    f["ttr"] = len(set(words[:3000])) / max(min(nw, 3000), 1)
    joined = "\n".join(allt); nc = max(len(joined), 1)
    for name, pat in (("nl", "\n"), ("bold", "**"), ("hash", "#"), ("excl", "!"), ("q", "?"), ("colon", ":"), ("paren", "("),
                      ("quote", '"'), ("dash", "- "), ("code", "```")):
        f[f"c_{name}"] = joined.count(pat) / nc * 1000
    f["c_list"] = len(re.findall(r"(?m)^\s*\d+\.", joined)) / max(len(allt), 1)
    f["c_nonascii"] = sum(1 for ch in joined if ord(ch) > 127) / nc * 1000
    for name, ws in (("i", {"i", "i'm", "me", "my"}), ("you", {"you", "your"}), ("sorry", {"sorry", "apologize"}),
                     ("we", {"we", "our"})):
        f[f"w_{name}"] = sum(w in ws for w in words) / nw * 1000
    f["w_smollm"] = sum(t.lower().count("smollm") for t in T_.values()) / max(len(T_), 1)
    tri = Counter(zip(words, words[1:], words[2:]))
    f["rep3"] = sum(v - 1 for v in tri.values() if v > 1) / nw
    fin = [g["finish"].get(s, "stop") for s in T_]
    f["frac_length"] = sum(x == "length" for x in fin) / max(len(fin), 1)
    return f


def guichet_aux(g, k):
    """Objets pour les caractéristiques croisées : distribution de mots, longueurs et z par emplacement."""
    c = {n: i for i, n in enumerate(g["cols"])}
    m = g["slots"] < k
    R, slots = g["R"][m], g["slots"][m]
    z = (R[:, c["lp_0.8"]] + R[:, c["H_0.8"]]) / np.sqrt(np.maximum(R[:, c["V_0.8"]], 1e-6))
    words = Counter(w.lower() for s, t in g["texts"].items() if s < k and s >= 3 for w in WORD.findall(t))
    wsets = {s: set(w.lower() for w in WORD.findall(t)) for s, t in g["texts"].items() if s < k}
    return {"z": dict(zip(slots.tolist(), z.tolist())), "n": dict(zip(slots.tolist(), R[:, c["n"]].tolist())),
            "words": words, "wsets": wsets}


def _js(a, b):
    keys = set(a) | set(b)
    ta, tb = sum(a.values()) or 1, sum(b.values()) or 1
    js = 0.0
    for w in keys:
        p, q = a.get(w, 0) / ta, b.get(w, 0) / tb
        mm = (p + q) / 2
        if p > 0: js += 0.5 * p * math.log(p / mm)
        if q > 0: js += 0.5 * q * math.log(q / mm)
    return js


def _corr(da, db):
    ks = sorted(set(da) & set(db) - {0, 1})
    if len(ks) < 4:
        return 0.0
    x = np.array([da[s] for s in ks]); y = np.array([db[s] for s in ks])
    if x.std() < 1e-9 or y.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def pair_features(fa, fb, aa, ab, names):
    va = np.array([fa[n] for n in names]); vb = np.array([fb[n] for n in names])
    cross = [_js(aa["words"], ab["words"]), _corr(aa["n"], ab["n"]), _corr(aa["z"], ab["z"])]
    ks = set(aa["wsets"]) & set(ab["wsets"])
    jac = [len(aa["wsets"][s] & ab["wsets"][s]) / max(len(aa["wsets"][s] | ab["wsets"][s]), 1) for s in ks]
    cross.append(float(np.mean(jac)) if jac else 0.0)
    return np.concatenate([np.abs(va - vb), (va + vb) / 2, np.minimum(va, vb), np.maximum(va, vb), cross])


def pair_names(names):
    return ([f"absdiff_{n}" for n in names] + [f"mean_{n}" for n in names] + [f"min_{n}" for n in names] +
            [f"max_{n}" for n in names] + ["x_js", "x_lencorr", "x_zcorr", "x_jaccard"])


def complete_linkage(P, thr=0.5):
    """Partition par lien complet sur la distance 1 - P ; coupe à 1 - thr (deux guichets ensemble seulement si
    toutes les paires du groupe ont P >= thr)."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    n = len(P)
    if n == 1:
        return [0]
    D = 1 - (P + P.T) / 2
    np.fill_diagonal(D, 0)
    Z = linkage(squareform(D, checks=False), method="complete")
    return (fcluster(Z, t=1 - thr, criterion="distance") - 1).tolist()


def mcc_pairs(labels_true, labels_pred):
    tp = fp = fn = tn = 0
    for i, j in itertools.combinations(range(len(labels_true)), 2):
        t = labels_true[i] == labels_true[j]; p = labels_pred[i] == labels_pred[j]
        tp += t and p; fp += (not t) and p; fn += t and not p; tn += (not t) and not p
    d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return max(0.0, (tp * tn - fp * fn) / d) if d else 0.0
