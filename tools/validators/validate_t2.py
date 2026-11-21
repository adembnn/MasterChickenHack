#!/usr/bin/env python3
"""Validateur local T2 (Python standard). Usage :
   python3 validate_t2.py submission.csv [--previous old.csv] [--ref-check]
Regles du serveur lues dans submission.py / task_template.py : .csv, une ligne par endpoint (ep-01..ep-33, une fois
chacun), 3 colonnes endpoint,family,model, pas de virgule dans les etiquettes, en-tete facultative, meme etiquette
model => meme etiquette family. Controles en plus (prudence) : pas de ref-*, pas d'etiquette vide, encodage UTF-8 sans
BOM, etiquette de modele non reutilisee dans deux familles, pas de soumission 'tout different' ni 'tout identique'."""
import csv, itertools, sys, collections

EP = [f"ep-{i:02d}" for i in range(1, 34)]


def main(argv):
    if not argv: sys.exit(__doc__)
    path, errs, warns = argv[0], [], []
    raw = open(path, "rb").read()
    if raw.startswith(b"\xef\xbb\xbf"): errs.append("BOM UTF-8 : le premier endpoint serait lu 'ï»¿ep-01'")
    try: text = raw.decode("utf-8")
    except UnicodeDecodeError: sys.exit("ECHEC : pas de l'UTF-8")
    if not path.lower().endswith(".csv"): errs.append("extension != .csv")
    rows = [[c.strip() for c in r] for r in csv.reader(text.splitlines()) if any(c.strip() for c in r)]
    if rows and [c.lower() for c in rows[0][:3]] == ["endpoint", "family", "model"]: rows = rows[1:]
    d = {}
    for i, r in enumerate(rows, 1):
        if len(r) != 3: errs.append(f"ligne {i}: {len(r)} colonnes (3 exigees ; virgule dans une etiquette ?) {r}"); continue
        e, fam, mod = r
        if e.startswith("ref-"): errs.append(f"ligne {i}: {e} : le serveur n'accepte que ep-01..ep-33")
        elif e not in EP: errs.append(f"ligne {i}: endpoint inconnu '{e}'")
        if e in d: errs.append(f"ligne {i}: {e} en double")
        if not fam or not mod: errs.append(f"ligne {i}: etiquette vide")
        d[e] = (fam, mod)
    miss = [e for e in EP if e not in d]
    if miss: errs.append(f"endpoints manquants ({len(miss)}): {miss}")
    fam_of = collections.defaultdict(set)
    for e, (fam, mod) in d.items(): fam_of[mod].add(fam)
    for mod, fams in fam_of.items():
        if len(fams) > 1: errs.append(f"modele '{mod}' dans plusieurs familles {sorted(fams)} (rejet serveur)")
    nf, nm = len({v[0] for v in d.values()}), len({v for v in d.values()})
    print(f"{len(d)} endpoints, {nf} familles, {nm} modeles")
    if nm == len(d) and len(d) > 1: warns.append("tout different : score 0, c'est la ligne de base")
    if nf == 1 and nm == 1: warns.append("tout identique : degenere")
    if "--previous" in argv:
        p = dict((r[0], tuple(r[1:3])) for r in csv.reader(open(argv[argv.index("--previous") + 1], encoding="utf-8")) if len(r) >= 3 and r[0] in EP)
        same = lambda x, a, b: x[a] == x[b]
        ch = [(a, b) for a, b in itertools.combinations(EP, 2) if a in d and b in d and a in p and b in p and same(d, a, b) != same(p, a, b)]
        print(f"paires dont le verdict 'meme modele' change vs precedent : {len(ch)}")
    for w in warns: print("AVERTISSEMENT", w)
    for e in errs: print("ERREUR", e)
    print("RESULTAT:", "INVALIDE" if errs else "VALIDE")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
