#!/usr/bin/env python3
"""Validateur local T2 (Python standard, sans dependance reseau). Usage :
   python3 validate_t2.py submission.csv [--previous old.csv] [--strict]

Regles du serveur, lues dans submission.py / task_template.py (reports/sources/t2/) :
  - fichier .csv (ALLOWED_SUFFIXES = (".csv",) dans submission.py)
  - une ligne par endpoint, ep-01..ep-33 (NUM_ENDPOINTS = 33 dans task_template.py), chacun une fois
  - 3 colonnes : endpoint,family,model (en-tete facultative, ordre des lignes indifferent)
  - etiquettes libres, "pas de virgule" (docstring task_template.py) -> une etiquette avec virgule casse le
    decoupage CSV (nombre de colonnes != 3), donc on la detecte comme un nombre de colonnes incorrect
  - "Two endpoints serve the same model ... must also share the family label" (task_template.py) :
    deux lignes avec la meme etiquette model doivent avoir la meme etiquette family -> rejet serveur sinon
  - 528 paires scorees = C(33,2), donc la perte d'un seul endpoint ou un doublon fausse tout le score
  - annonce Discord des organisateurs (21/11, 19h01, salon task-2-model-equality, relayee par la session principale,
    absente des scripts officiels) : l'etiquette model commence par l'etiquette de sa famille (A -> A1, A2) et une
    etiquette model n'appartient qu'a une seule famille (ep-01,A,X + ep-02,B,X est rejete)

Controles supplementaires (prudence, pas forcement rejetes par le serveur mais signale) :
  - pas de ref-* (ce sont les endpoints d'entrainement, jamais scores)
  - pas d'etiquette vide ou faite seulement d'espaces
  - encodage UTF-8 sans BOM (un BOM ferait lire 'ï»¿ep-01' au lieu de 'ep-01')
  - une etiquette "model" reutilisee dans deux familles differentes (deja une erreur dure ci-dessus,
    repris ici comme verification independante pour que les tests puissent la cibler seule)
  - soumission degeneree : "tout different" (score 0, c'est le baseline solve()) ou "tout identique"
    (--strict transforme ces avertissements en erreurs)
  - avec --previous : liste les paires dont le verdict "meme modele" change par rapport a un fichier
    deja soumis (utile vu le delai de 30 min entre deux soumissions)

Expose aussi une API Python (parse_submission, validate) pour etre testee et reutilisee par le dry-run.
"""
import argparse
import collections
import csv
import itertools
import sys
from dataclasses import dataclass, field
from pathlib import Path

NUM_ENDPOINTS = 33
ENDPOINTS = [f"ep-{i:02d}" for i in range(1, NUM_ENDPOINTS + 1)]
HEADER = ("endpoint", "family", "model")


@dataclass
class ParseResult:
    rows: dict  # endpoint -> (family, model)
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def parse_submission(raw: bytes, filename: str = "submission.csv") -> ParseResult:
    """Parse les octets bruts d'un fichier de soumission T2 et renvoie un ParseResult.
    N'ouvre aucun fichier ni connexion : `raw` doit deja etre lu par l'appelant."""
    errors, warnings = [], []

    if raw.startswith(b"\xef\xbb\xbf"):
        errors.append("BOM UTF-8 present : le premier endpoint serait lu 'ï»¿ep-01' au lieu de 'ep-01'")

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        errors.append(f"le fichier n'est pas de l'UTF-8 valide : {exc}")
        return ParseResult(rows={}, errors=errors, warnings=warnings)

    if not filename.lower().endswith(".csv"):
        errors.append(f"extension != .csv (submission.py n'accepte que {('.csv',)}) : '{filename}'")

    raw_rows = [row for row in csv.reader(text.splitlines()) if any(cell.strip() for cell in row)]
    if raw_rows and tuple(c.strip().lower() for c in raw_rows[0][:3]) == HEADER:
        raw_rows = raw_rows[1:]

    rows: dict = {}
    for i, row in enumerate(raw_rows, 1):
        stripped = [c.strip() for c in row]
        if len(stripped) != 3:
            errors.append(
                f"ligne {i}: {len(stripped)} colonnes (3 attendues ; une virgule dans une etiquette peut "
                f"etre la cause) -> {row}"
            )
            continue
        endpoint, family, model = stripped
        if endpoint.startswith("ref-"):
            errors.append(f"ligne {i}: '{endpoint}' est un endpoint de reference, le serveur n'accepte que ep-01..ep-{NUM_ENDPOINTS:02d}")
        elif endpoint not in ENDPOINTS:
            errors.append(f"ligne {i}: endpoint inconnu '{endpoint}' (attendu ep-01..ep-{NUM_ENDPOINTS:02d})")
        if endpoint in rows:
            errors.append(f"ligne {i}: endpoint '{endpoint}' en double (deja vu avant)")
        if not family or not model:
            errors.append(f"ligne {i}: etiquette family ou model vide")
        if "," in family or "," in model:
            errors.append(
                f"ligne {i}: virgule dans une etiquette ({family!r}, {model!r}) ; meme entre guillemets c'est "
                f"interdit par la consigne ('no commas') et le serveur peut ne pas gerer le quoting CSV"
            )
        if family and model and not model.startswith(family):
            errors.append(
                f"ligne {i}: le modele '{model}' ne commence pas par sa famille '{family}' "
                f"(regle Discord 21/11 19h01 : famille A -> modeles A1, A2...)"
            )
        rows[endpoint] = (family, model)

    missing = [e for e in ENDPOINTS if e not in rows]
    if missing:
        errors.append(f"endpoints manquants ({len(missing)}): {missing}")

    families_per_model = collections.defaultdict(set)
    for endpoint, (family, model) in rows.items():
        if family and model:
            families_per_model[model].add(family)
    for model, families in families_per_model.items():
        if len(families) > 1:
            errors.append(
                f"etiquette model '{model}' apparait dans plusieurs familles {sorted(families)} "
                f"(doit partager une seule family ; task_template.py: 'they must also share the family label')"
            )

    valid_rows = {e: v for e, v in rows.items() if e in ENDPOINTS}
    n_models = len({v for v in valid_rows.values()})
    n_families = len({v[0] for v in valid_rows.values()})
    if valid_rows and n_models == len(valid_rows) and len(valid_rows) > 1:
        warnings.append("soumission 'tout different' : chaque endpoint a un modele unique, score 0 (c'est le baseline solve())")
    if valid_rows and n_families == 1 and n_models == 1:
        warnings.append("soumission 'tout identique' : un seul modele pour tous les endpoints, verdict degenere")

    return ParseResult(rows=rows, errors=errors, warnings=warnings)


def changed_pairs(rows: dict, previous_rows: dict) -> list:
    """Paires d'endpoints dont le verdict 'meme modele' change entre `rows` et `previous_rows`."""
    common = [e for e in ENDPOINTS if e in rows and e in previous_rows]
    changed = []
    for a, b in itertools.combinations(common, 2):
        now_same = rows[a][1] == rows[b][1]
        before_same = previous_rows[a][1] == previous_rows[b][1]
        if now_same != before_same:
            changed.append((a, b))
    return changed


def validate(path: Path, previous_path: Path = None, strict: bool = False) -> int:
    raw = path.read_bytes()
    result = parse_submission(raw, filename=path.name)

    n_endpoints = len({e for e in result.rows if e in ENDPOINTS})
    n_families = len({v[0] for e, v in result.rows.items() if e in ENDPOINTS})
    n_models = len({v for e, v in result.rows.items() if e in ENDPOINTS})
    print(f"{n_endpoints} endpoints, {n_families} familles, {n_models} modeles")

    if previous_path is not None:
        previous = parse_submission(previous_path.read_bytes(), filename=previous_path.name)
        pairs = changed_pairs(result.rows, previous.rows)
        print(f"paires dont le verdict 'meme modele' change vs precedent : {len(pairs)}")

    for w in result.warnings:
        print("AVERTISSEMENT", w)
    for e in result.errors:
        print("ERREUR", e)

    failed = bool(result.errors) or (strict and bool(result.warnings))
    print("RESULTAT:", "INVALIDE" if failed else "VALIDE")
    return 1 if failed else 0


def main(argv: list) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("submission", type=Path, help="fichier submission.csv a valider")
    parser.add_argument("--previous", type=Path, default=None, help="soumission precedente, pour comparer les verdicts")
    parser.add_argument("--strict", action="store_true", help="transforme les avertissements (tout different / identique) en erreurs")
    args = parser.parse_args(argv)

    if not args.submission.is_file():
        print(f"ERREUR fichier introuvable : {args.submission}", file=sys.stderr)
        return 1
    return validate(args.submission, previous_path=args.previous, strict=args.strict)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
