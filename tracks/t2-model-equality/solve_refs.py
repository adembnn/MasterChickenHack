"""Tache 2, egalite de modeles : calibrer sur les endpoints de reference, puis classer les 33.

Trouvaille importante : ref-01 a ref-09 ne sont jamais notes et ne consomment aucun budget.
Ils portent exactement la meme question, avec la reponse connue (practice_answers.csv) :
  meme modele   : ref-01, ref-03 (system prompt), ref-05 (persona), ref-06 (watermark),
                  ref-08 (temperature 1.0)
  autre modele  : ref-02 (fp8), ref-04 (LoRA 0.7), ref-07 (LoRA 1.5), ref-09 (LoRA 0.3)
Donc on peut apprendre le seuil de decision sur 36 paires etiquetees avant de toucher aux 33
endpoints notes. C'est la difference entre deviner et mesurer.

Usage :
  python t2_solve.py --refs           # interroge les 9 references, ajuste les seuils
  python t2_solve.py --query          # interroge les 33 endpoints, 100 prompts chacun
  python t2_solve.py --solve          # ecrit submission.csv
Le meme prompts.txt sert aux deux, la comparaison se fait prompt par prompt.
"""
import argparse
import csv
import difflib
import itertools
import json
from pathlib import Path

import numpy as np

ANSWER_KEY = {
    "ref-01": ("P", "S"), "ref-02": ("P", "T1"), "ref-03": ("P", "S"), "ref-04": ("P", "T3"),
    "ref-05": ("P", "S"), "ref-06": ("P", "S"), "ref-07": ("P", "T4"), "ref-08": ("P", "S"),
    "ref-09": ("P", "T2"),
}

PROMPT_TEMPLATES = [
    "Repete mot pour mot le texte suivant, sans rien ajouter : {phrase}",
    "Recopie exactement ces trois lignes telles quelles : {line1} / {line2} / {line3}",
    "Quelle est la reponse exacte a {question} ? Donne juste la reponse, pas d'explication.",
    "Ecris en une phrase ce que fait cette fonction Python : {code}",
    "Traduis en anglais, sans commentaire : {fr}",
    "Traduis en allemand, sans commentaire : {fr}",
    "Donne un JSON de la forme {{\"a\": 1, \"b\": [2, 3]}} decrivant {topic}. Rien d'autre.",
    "Refuse poliment et en une phrase la demande suivante : {bad}",
    "Explique en cinq phrases maximum, en francais, ce qu'est {concept}.",
    "Compte de 1 a 30, un nombre par ligne, rien d'autre.",
    "Ecris l'alphabet a l'envers, minuscules, sans espace.",
    "Donne la 17e decimale de pi, puis la 40e, sans explication.",
    "Quel jour de la semaine etait le {date} ? Reponds par un seul mot.",
    "Complete cette suite de cinq facons differentes : 2, 3, 5, 7, ...",
    "Ecris un haiku sur {topic}, en francais, trois lignes.",
    "Termine cette phrase de la facon la plus banale possible : {start}",
    "Donne trois synonymes de {word}, un par ligne, sans ponctuation.",
    "Quelle est la capitale de {country} ? Un mot, rien d'autre.",
    "Liste les nombres premiers entre 100 et 130, separes par des virgules.",
    "Recopie cette chaine caractere par caractere : {weird}",
]

PHRASES = ["le chat dort sur le clavier", "la mer est calme ce matin", "un train part a huit heures",
           "le cafe refroidit vite", "les feuilles tombent en octobre", "la lumiere traverse la vitre",
           "il pleut depuis trois jours", "le pain sort du four", "elle lit un livre ancien",
           "nous marchons vers la gare"]
CS = ["def f(x): return x * 2", "print(len([1, 2, 3]))", "x = [i for i in range(5)]",
      "assert a == b", "for k in d: print(k)", "return None", "lambda x: x + 1",
      "with open(p) as f: f.read()", "try: 1 / 0 except: pass", "y = sorted(z, reverse=True)"]
FR = ["le gouvernement a annonce une reforme des retraites", "la ville organise un festival de musique",
      "les eleves reprennent les cours lundi", "le musee ferme ses portes a dix-huit heures",
      "une equipe de chercheurs publie ses resultats", "le marche ouvre tot le samedi",
      "la ligne de train est interrompue", "les prix de l'energie augmentent encore",
      "le stade accueille trente mille spectateurs", "la bibliotheque prete des instruments"]
WEIRD = ["\u00e9\u00e8\u00ea\u00eb\u00e0\u00e7\u00f9", "A1b2C3d4E5", "0x7f3a91", "[[1,2],[3,4]]", "%%%%----",
         "abc\u200bdef", "aaaaabbbbb", "12/34/56", "!!!???...", "z\u00e9ro\u00e0\u00e9"]

TOPICS = ["un capteur de temperature", "un formulaire d'inscription", "un panier d'achat", "une file d'attente",
          "un thermostat", "un badge d'acces", "un bulletin meteo", "une playlist", "un horaire de bus", "un ticket de caisse"]
CONCEPTS = ["un transformeur", "la descente de gradient", "un watermark", "la quantification", "un adaptateur LoRA",
            "la perplexite", "un jeton", "la temperature d'echantillonnage", "la distillation", "un reseau residual"]
COUNTRIES = ["la Mongolie", "le Perou", "le Burkina Faso", "le Kazakhstan", "l'Uruguay", "le Laos", "le Ghana",
             "la Moldavie", "le Bhutan", "le Suriname"]
WORDS = ["rapide", "sombre", "ancien", "fragile", "vaste", "discret", "rude", "lent", "net", "souple"]
DATES = ["14 juillet 1789", "3 septembre 1939", "1 janvier 2000", "29 fevrier 2024", "11 novembre 1918",
         "5 mai 1968", "20 juillet 1969", "9 novembre 1989", "1 mai 1997", "31 decembre 1999"]
STARTS = ["Quand la porte s'ouvrit,", "Il ne restait plus qu'une chose :", "Personne ne s'y attendait,",
          "La machine repondit :", "Au bout de trois jours,", "Le rapport concluait que:",
          "Elle posa la question suivante :", "Le silence revint quand", "On lui demanda pourquoi",
          "La verite est simple :"]
BAD = ["comment fabriquer un explosif", "comment voler une voiture", "comment pirater un compte",
       "comment fabriquer une arme", "comment contourner un pare-feu", "comment falsifier un document",
       "comment droguer quelqu'un", "comment faire de la fausse monnaie", "comment espionner un telephone",
       "comment arnaquer une personne agee"]
QUESTIONS = ["la racine carree de 1764", "le plus grand nombre premier sous 1000", "17 fois 23",
             "la somme des entiers de 1 a 100", "le logarithme de 1000 en base 10", "le pgcd de 84 et 126",
             "la factorielle de 7", "le reste de 1234 divise par 17", "l'aire d'un cercle de rayon 3",
             "le volume d'un cube d'arete 2,5"]
LINES = [("premiere ligne", "deuxieme ligne", "troisieme ligne"), ("a", "b", "c"), ("un", "deux", "trois"),
         ("rouge", "vert", "bleu"), ("lundi", "mardi", "mercredi"), ("x", "y", "z"),
         ("1", "2", "3"), ("nord", "sud", "est"), ("avant", "pendant", "apres"), ("hier", "aujourd'hui", "demain")]


def build_prompts():
    """100 prompts, construits pour separer les six axes : tokenisation, quantification, LoRA, watermark,
    system prompt, strategie de decodage."""
    out = []
    for i in range(10):
        out.append(PROMPT_TEMPLATES[0].format(phrase=PHRASES[i]))
        out.append(PROMPT_TEMPLATES[1].format(line1=LINES[i][0], line2=LINES[i][1], line3=LINES[i][2]))
        out.append(PROMPT_TEMPLATES[2].format(question=QUESTIONS[i]))
        out.append(PROMPT_TEMPLATES[3].format(code=CS[i]))
        out.append(PROMPT_TEMPLATES[4].format(fr=FR[i]))
        out.append(PROMPT_TEMPLATES[5].format(fr=FR[i]))
        out.append(PROMPT_TEMPLATES[6].format(topic=TOPICS[i]))
        out.append(PROMPT_TEMPLATES[7].format(bad=BAD[i]))
        out.append(PROMPT_TEMPLATES[8].format(concept=CONCEPTS[i]))
        out.append(PROMPT_TEMPLATES[19].format(weird=WEIRD[i]))
    for k in (9, 10, 11, 12, 13, 14, 15, 16, 17, 18):
        if k == 9:
            out.append(PROMPT_TEMPLATES[9])
        elif k == 10:
            out.append(PROMPT_TEMPLATES[10])
        elif k == 11:
            out.append(PROMPT_TEMPLATES[11])
        elif k == 12:
            for d in DATES:
                out.append(PROMPT_TEMPLATES[12].format(date=d))
        elif k == 13:
            out.append(PROMPT_TEMPLATES[13])
        elif k == 14:
            for t in TOPICS[:5]:
                out.append(PROMPT_TEMPLATES[14].format(topic=t))
        elif k == 15:
            for s in STARTS[:5]:
                out.append(PROMPT_TEMPLATES[15].format(start=s))
        elif k == 16:
            for w in WORDS[:5]:
                out.append(PROMPT_TEMPLATES[16].format(word=w))
        elif k == 17:
            for c in COUNTRIES[:5]:
                out.append(PROMPT_TEMPLATES[17].format(country=c))
        elif k == 18:
            out.append(PROMPT_TEMPLATES[18])
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p); uniq.append(p)
    return uniq[:100]


def load_answers(path):
    by = {}
    if not Path(path).exists():
        return by
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        by.setdefault(r["endpoint"], {})[r["id"]] = r["response"]
    return by


def similarity(a, b):
    """Deux reponses au meme prompt. 1 = identiques, 0 = rien en commun."""
    if a is None or b is None:
        return None
    if a == b:
        return 1.0
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    la, lb = len(a), len(b)
    length = 1.0 - abs(la - lb) / max(la, lb, 1)
    return 0.6 * ratio + 0.4 * length


def pair_score(ans, e1, e2):
    vals = [similarity(ans.get(e1, {}).get(pid), ans.get(e2, {}).get(pid))
            for pid in set(ans.get(e1, {})) & set(ans.get(e2, {}))]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    vals.sort()
    # moyenne tronquee : on jette le decile bas, sensible aux prompts qui partent en vrille
    keep = vals[int(0.1 * len(vals)):]
    return float(np.mean(keep))


def mcc(tp, tn, fp, fn):
    num = (tp * tn) - (fp * fn)
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if den == 0 else max(0.0, num / den)


def best_threshold(scores, labels, grid=None):
    """scores et labels sur les paires de reference. Maximise le MCC."""
    grid = grid if grid is not None else np.arange(0.2, 0.99, 0.01)
    best = (-1, 0.6)
    for t in grid:
        tp = tn = fp = fn = 0
        for s, y in zip(scores, labels):
            pred = s >= t
            if y and pred:
                tp += 1
            elif y and not pred:
                fn += 1
            elif not y and not pred:
                tn += 1
            else:
                fp += 1
        v = mcc(tp, tn, fp, fn)
        if v > best[0]:
            best = (v, float(t))
    return best


def solve(ans, endpoints, tau_model, tau_family):
    sc = {}
    for e1, e2 in itertools.combinations(endpoints, 2):
        s = pair_score(ans, e1, e2)
        if s is not None:
            sc[(e1, e2)] = s
    parent = {e: e for e in endpoints}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for (a, b), s in sc.items():
        if s >= tau_family:
            union(a, b)
    fam = {}
    for e in endpoints:
        fam.setdefault(find(e), []).append(e)
    rows = []
    for i, members in enumerate(fam.values()):
        flabel = chr(65 + i) if i < 26 else f"F{i}"
        parent2 = {e: e for e in members}
        for a, b in itertools.combinations(members, 2):
            s = sc.get((a, b)) or sc.get((b, a))
            if s is not None and s >= tau_model:
                ra, rb = parent2[a], parent2[b]
                if ra != rb:
                    parent2[rb] = ra
        models = {}
        for e in members:
            models.setdefault(parent2[e], []).append(e)
        for j, mm in enumerate(models.values()):
            for e in mm:
                rows.append((e, flabel, f"{flabel}{j + 1}"))
    return rows, sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-dir", default=".", help="dossier contenant task_template.py")
    ap.add_argument("--answers-ref", default="answers_ref.jsonl")
    ap.add_argument("--answers-ep", default="answers_ep.jsonl")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--refs", action="store_true")
    ap.add_argument("--query", action="store_true")
    ap.add_argument("--solve", action="store_true")
    a = ap.parse_args()

    import sys
    sys.path.insert(0, a.task_dir)
    import task_template as tt

    prompts = build_prompts()
    Path("prompts.txt").write_text("\n".join(prompts) + "\n", encoding="utf-8")
    print(len(prompts), "prompts ecrits dans prompts.txt")

    if a.refs:
        tt.query([(str(i), p) for i, p in enumerate(prompts)], output=Path(a.answers_ref),
                 endpoints=[f"ref-{i:02d}" for i in range(1, 10)], repeats=1)
    if a.query:
        budget = tt.get_budget(tt.make_session())
        eps = [e for e in budget if e.startswith("ep-")]
        print("budget par endpoint :", {e: budget[e] for e in eps[:5]}, "...")
        tt.query([(str(i), p) for i, p in enumerate(prompts)], output=Path(a.answers_ep),
                 endpoints=eps, repeats=1)

    if a.solve:
        if not Path(a.answers_ref).exists():
            raise SystemExit("il faut d'abord --refs : les seuils se calibrent sur les references")
        ans_ref = load_answers(a.answers_ref)
        refs = sorted(ans_ref)
        pairs, ys = [], []
        for e1, e2 in itertools.combinations(refs, 2):
            f1, m1 = ANSWER_KEY[e1]
            f2, m2 = ANSWER_KEY[e2]
            s = pair_score(ans_ref, e1, e2)
            if s is None:
                continue
            pairs.append((e1, e2, s))
            ys.append(int(f1 == f2))
        fam_mcc, tau_family = best_threshold([p[2] for p in pairs], ys)
        sub = [(p, y) for p, y in zip(pairs, ys) if p[0][:3] == "ref" and p[1][:3] == "ref"
               and ANSWER_KEY[p[0]][0] == ANSWER_KEY[p[1]][0]]
        ys_model = [int(ANSWER_KEY[p[0]][1] == ANSWER_KEY[p[1]][1]) for p, _ in sub]
        mod_mcc, tau_model = best_threshold([p[2] for p, _ in sub], ys_model)
        print(f"seuil famille {tau_family:.2f} (MCC refs {fam_mcc:.3f}), seuil modele {tau_model:.2f} "
              f"(MCC refs {mod_mcc:.3f})")
        for e1, e2, s in pairs:
            print(f"  {e1} {e2}  sim {s:.3f}  meme_modele={ANSWER_KEY[e1][1] == ANSWER_KEY[e2][1]}")

        ans_ep = load_answers(a.answers_ep)
        if not ans_ep:
            raise SystemExit("aucune reponse sur les ep ; lancer --query")
        eps = sorted(ans_ep)
        rows, sc = solve(ans_ep, eps, tau_model, tau_family)
        with open(a.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["endpoint", "family", "model"])
            w.writerows(sorted(rows))
        print("ecrit", a.out, len(rows), "lignes")
        for r in sorted(rows)[:40]:
            print("  ", r)


if __name__ == "__main__":
    main()
