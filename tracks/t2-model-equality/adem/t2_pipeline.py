#!/usr/bin/env python
"""Pipeline T2 de bout en bout (stdlib seule, aucun appel réseau, ne lit aucune clé).

Entrée  : un ou plusieurs fichiers de réponses au format de task_template.py
          ({"endpoint","id","repeat","prompt","response"} par ligne) et la liste de sondes gelée.
Sortie  : <out>.csv         soumission principale (endpoint,family,model)
          <out>_repli.csv   repli à faible risque : familles seules, chaque endpoint isolé (MCC modèle 0, compte
                            seulement pour la part famille, 10 % de la note)
          <out>_diag.txt    type détecté par endpoint, prompt système extrait, groupe central, rattachements

Étapes (analyse/rules.py et analyse/features.py) :
  1. familles : lien moyen sur une proximité normalisée (ou --families fichier.csv pour imposer les familles).
  2. par famille, type de chaque endpoint : T = échantillonné (deux réponses différentes au même message),
     S = prompt système ou persona (une sonde d'extraction rend un passage « You are... »), P = le reste.
  3. décodeur à règles : les P sont regroupés par identité de texte (seuil tau_pp) ; le groupe central est élu par
     vote des plus proches voisins ; un P isolé au biais d'étiquettes marqué est une copie filigranée du centre ;
     chaque S est rattaché au groupe P le plus proche (le centre l'emporte à `margin` près) ; chaque T va au centre.
  Si une famille n'a aucun endpoint P (tout échantillonne), le décodeur du régime échantillonné prend le relais.

Usage :
  python t2_pipeline.py --answers answers.jsonl --probes probes/probes_gelees.jsonl --out soumission --prefix ep-
"""
import argparse, csv, itertools, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyse"))
from parse import load                                       # noqa: E402
from features import build, pair_features                    # noqa: E402
from rules import decode, decode_sampled, family_partition   # noqa: E402

VARIANTES = {
    "principal": {},
    "centre": {"margin": 9.0},                               # tout endpoint déguisé va au groupe central
    "ancre": {"margin": 0.0, "attach_t": "nearest"},         # plus proche ancre en clair, sans a priori
    "prudent": {"attach_s": False, "attach_t": False, "detect_w": False},   # déguisés isolés
}


def kind(e):
    return "T" if e.nondet > 0 else ("S" if e.sys else "P")


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["endpoint", "family", "model"])
        w.writerows(rows)


def run(answers, probes_path, variante="principal", fam_cut=0.4, endpoints=None, families=None, **kw):
    probes = [json.loads(l) for p in probes_path.split(",") for l in open(p, encoding="utf-8") if l.strip()]
    kinds, data = load(answers, probes_path.split(",")[0])
    kinds.update({p["id"]: p["kind"] for p in probes})
    prompts = {p["id"]: p["prompt"] for p in probes}
    if endpoints is not None:
        data = {e: d for e, d in data.items() if e in endpoints}
    names = sorted(data)
    diag = []
    # 1. familles
    if families:
        fam = {e: families[e] for e in names}
    else:
        eps = build(data, kinds, prompts=prompts)
        F = {frozenset((a, b)): pair_features(eps[a], eps[b]) for a, b in itertools.combinations(names, 2)}
        fam = family_partition(names, F, fam_cut)
    fam_ids = sorted(set(fam.values()), key=lambda k: min(e for e in names if fam[e] == k))
    rows, rows_repli = [], []
    params = dict(VARIANTES[variante])
    params.update(kw)
    for fi, k in enumerate(fam_ids):
        members = [e for e in names if fam[e] == k]
        fn = f"F{fi + 1}"
        # 2. types, recalculés dans la famille (les indices relatifs aux pairs n'ont de sens qu'entre pairs)
        sub = build({e: data[e] for e in members}, kinds, prompts=prompts)
        kd = {e: kind(sub[e]) for e in members}
        ff = {frozenset((a, b)): pair_features(sub[a], sub[b]) for a, b in itertools.combinations(members, 2)}
        # 3. décodeur
        if len(members) == 1:
            lab, info = {members[0]: 0}, {}
        elif not any(v == "P" for v in kd.values()):
            lab, info = decode_sampled(members, ff, sub)
            diag.append(f"!! famille {fn} : aucun endpoint déterministe sans prompt système -> décodeur du régime échantillonné")
        else:
            lab, info = decode(members, ff, kd, **params)
        ren = {}
        for i, e in enumerate(members):
            rows.append((e, fn, f"{fn}M{ren.setdefault(lab[e], len(ren) + 1)}"))
            rows_repli.append((e, fn, f"{fn}M{i + 1}"))
        diag.append(f"== famille {fn} : {len(members)} endpoints, {len(ren)} modèles, groupe central autour de "
                    f"{info.get('hub')}, filigranés détectés : {info.get('w')}")
        for e in members:
            x = sub[e]
            diag.append(f"  {e:22s} type={kd[e]} modèle={fn}M{ren[lab[e]]:<3d} nondet={x.nondet:.2f} fuites={x.n_leak} "
                        f"mots_récurrents={','.join(x.recur_words[:4])} rattachement={info.get('attach', {}).get(e, '')} "
                        f"prompt_système={x.leak[:60]!r}")
    return rows, rows_repli, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", nargs="+", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--out", default="soumission")
    ap.add_argument("--variante", choices=sorted(VARIANTES), default="principal")
    ap.add_argument("--fam_cut", type=float, default=0.4)
    ap.add_argument("--families", default=None, help="CSV endpoint,family pour imposer les familles (méthode de Titouan)")
    ap.add_argument("--prefix", default=None, help="ne garder que les endpoints commençant par ce préfixe (ep- ou ref-)")
    a = ap.parse_args()
    eps = None
    if a.prefix:
        eps = set()
        for p in a.answers:
            for l in open(p, encoding="utf-8"):
                if l.strip():
                    e = json.loads(l)["endpoint"]
                    if e.startswith(a.prefix):
                        eps.add(e)
    fams = None
    if a.families:
        fams = {r[0]: r[1] for r in csv.reader(open(a.families, encoding="utf-8-sig")) if r and r[0] != "endpoint"}
    rows, repli, diag = run(a.answers, a.probes, a.variante, a.fam_cut, eps, fams)
    write_csv(a.out + ".csv", rows)
    write_csv(a.out + "_repli.csv", repli)
    open(a.out + "_diag.txt", "w", encoding="utf-8").write("\n".join(diag) + "\n")
    print(f"{len(rows)} endpoints, {len(set(r[1] for r in rows))} familles, {len(set(r[2] for r in rows))} modèles "
          f"-> {a.out}.csv (+ _repli.csv, _diag.txt)")


if __name__ == "__main__":
    main()
