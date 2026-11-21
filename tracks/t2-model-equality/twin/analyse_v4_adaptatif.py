#!/usr/bin/env python
"""Procédure de décision UNIQUE et adaptative (PR #22), appliquée à l'identique à chaque modèle sans savoir lequel.

Aucun seuil appris ni choisi par modèle : seule la règle « bruit mesuré + 2·SE » de PR #22.
  Sondes à choix forcé : chaque endpoint rejoue ses 100 sondes (autre taille de lot / autres graines) ->
    auto-désaccord k_e. p_e = (k_e+1)/(n+2) (lissage de Laplace : évite SE=0 quand k_e=0).
    Paire (a,b) « même modèle » ssi désaccord(a,b) <= p + 2·sqrt(p(1-p)/n), p = max(p_a, p_b).
  NLL croisée : paire « même modèle » ssi l'écart moyen n'est pas significatif à 2·SE (|t| <= 2,
    t apparié sur les écarts par prompt).
  v4 combinée : même modèle ssi les deux le disent (les sondes regroupent, la NLL garde contre fp8/LoRA).
Partition par union-find (transitivité, comme une vraie soumission), puis MCC de score_practice.py, qui est
le seul à lire practice_answers.csv.
Usage : python analyse_v4_adaptatif.py --dir <dossier> --worlds noisy,clean --models olmo,qwen,smol
"""
import argparse, csv, json, math, os, re, subprocess, sys, tempfile

REFS = ["ref-0%d" % i for i in range(1, 10)]
SCORER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "reports", "sources", "t2",
                      "score_practice.py")


def self_noise(d, rid):
    a, b = d["fc_answers"][rid], d["fc_replay"][rid]
    k = sum(x != y for x, y in zip(a, b))
    return k, (k + 1) / (len(a) + 2), len(a)


def fc_same(d, a, b):
    _, pa, n = self_noise(d, a)
    _, pb, _ = self_noise(d, b)
    p = max(pa, pb)
    tau = p + 2 * math.sqrt(p * (1 - p) / n)
    return (1 - d["pair_stats"][f"{a}_{b}"]["fc_agree"]) <= tau


def nll_t(gaps):
    g = [x for x in gaps if x == x]
    if len(g) < 2:
        return 0.0
    m = sum(g) / len(g)
    sd = math.sqrt(sum((x - m) ** 2 for x in g) / (len(g) - 1))
    return m / (sd / math.sqrt(len(g))) if sd > 0 else (0.0 if m == 0 else math.inf)


def nll_same(d, a, b):
    return abs(nll_t(d["pair_stats"][f"{a}_{b}"]["nll_gaps"])) <= 2


RULES = {
    "sondes seules": lambda d, a, b: fc_same(d, a, b),
    "NLL seule": lambda d, a, b: nll_same(d, a, b),
    "v4 (sondes ET NLL)": lambda d, a, b: fc_same(d, a, b) and nll_same(d, a, b),
}


def partition(d, rule):
    parent = {r: r for r in REFS}
    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x
    for i, a in enumerate(REFS):
        for b in REFS[i + 1:]:
            if rule(d, a, b):
                parent[find(a)] = find(b)
    return {r: find(r) for r in REFS}


def partition_complete(d, rule):
    """Lien complet : deux groupes ne fusionnent que si TOUTES leurs paires croisées disent « même ». Empêche
    l'enchaînement (A~B, B~C mais A≁C) que l'union-find accepte. Paires examinées par |t| NLL croissant."""
    pairs = [(a, b) for i, a in enumerate(REFS) for b in REFS[i + 1:]]
    pairs.sort(key=lambda p: abs(nll_t(d["pair_stats"][f"{p[0]}_{p[1]}"]["nll_gaps"])))
    ok = {p: rule(d, *p) for p in pairs}
    same = lambda x, y: ok[(x, y)] if (x, y) in ok else ok[(y, x)]
    cl = {r: {r} for r in REFS}
    for a, b in pairs:
        if cl[a] is cl[b] or not ok[(a, b)]:
            continue
        if all(same(x, y) for x in cl[a] for y in cl[b]):
            merged = cl[a] | cl[b]
            for r in merged:
                cl[r] = merged
    return {r: min(cl[r]) for r in REFS}


def score(labels):
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="") as f:
        w = csv.writer(f)
        w.writerow(["endpoint", "family", "model"])
        for r in REFS:
            w.writerow([r, "P", labels[r]])
        path = f.name
    out = subprocess.run([sys.executable, SCORER, path], capture_output=True, text=True).stdout
    os.unlink(path)
    return float(re.search(r":\s*([0-9.]+)", out).group(1))


def groups(labels):
    g = {}
    for r in REFS:
        g.setdefault(labels[r], []).append(r[-2:])
    return " ".join("{" + ",".join(v) + "}" for v in sorted(g.values(), key=lambda v: (-len(v), v)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    ap.add_argument("--worlds", default="noisy,clean")
    ap.add_argument("--models", default="olmo,qwen,smol")
    a = ap.parse_args()
    table, diag = [], {}
    for w in a.worlds.split(","):
        for m in a.models.split(","):
            p = os.path.join(a.dir, f"{w}_{m}.json")
            if not os.path.exists(p):
                continue
            d = json.load(open(p))
            row = {"monde": w, "modèle": m}
            for name, rule in RULES.items():
                lab = partition(d, rule)
                row[name] = score(lab)
                row[name + " groupes"] = groups(lab)
                lab = partition_complete(d, rule)
                row[name + " | lien complet"] = score(lab)
                row[name + " | lien complet groupes"] = groups(lab)
            table.append(row)
            diag[f"{w}_{m}"] = {
                "auto_desaccord": {r: self_noise(d, r)[0] for r in REFS},
                "accord_ref01": {r: d["pair_stats"][f"ref-01_{r}"]["fc_agree"] for r in REFS[1:]},
                "t_nll_ref01": {r: round(nll_t(d["pair_stats"][f"ref-01_{r}"]["nll_gaps"]), 2) for r in REFS[1:]},
            }
    keys = [k + s for k in RULES for s in ("", " | lien complet")]
    print("| monde | modèle | " + " | ".join(keys) + " |")
    print("|---|---|" + "---|" * len(keys))
    for row in table:
        print(f"| {row['monde']} | {row['modèle']} | " + " | ".join(f"{row[k]:.3f}" for k in keys) + " |")
    for row in table:
        print(f"{row['monde']} {row['modèle']}:")
        for k in keys:
            print(f"    {k}: {row[k + ' groupes']}")
    print(json.dumps(diag, indent=1, ensure_ascii=False))
    json.dump({"table": table, "diag": diag}, open(os.path.join(a.dir, "analyse_v4_adaptatif.json"), "w"),
              indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
