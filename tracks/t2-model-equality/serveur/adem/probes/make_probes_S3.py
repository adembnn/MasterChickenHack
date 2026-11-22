#!/usr/bin/env python
"""Sondes du régime échantillonné, v2 : peu de sondes, beaucoup de tirages (10 par sonde), lots de 10 questions.

  ab10 : 10 questions à choix forcé A/B par message
  eq10 : 10 choix entre options équivalentes par message (« Pick exactly one: Gmail, Outlook, Yahoo »), la version texte
         du dispositif d'AgentProv : aucune bonne réponse, le choix dépend des poids et peu du prompt système
  nb10 : 10 questions chiffrées par message
  lp   : 5 questions sous une paire d'étiquettes (biais de filigrane)
Usage : python make_probes_S3.py --out probes_S3.jsonl
"""
import argparse, json, random
from make_probes import ab_q, num_q

POOLS = [["Gmail", "Outlook", "Yahoo"], ["Google", "Bing", "DuckDuckGo"], ["Python", "Java", "Ruby"],
         ["red", "blue", "green"], ["apple", "pear", "plum"], ["Monday", "Tuesday", "Friday"],
         ["hearts", "spades", "clubs"], ["north", "south", "east"], ["cat", "dog", "rabbit"],
         ["tea", "coffee", "juice"], ["London", "Paris", "Rome"], ["piano", "violin", "flute"],
         ["Chrome", "Firefox", "Safari"], ["spring", "summer", "autumn"], ["gold", "silver", "bronze"],
         ["circle", "square", "triangle"], ["Alice", "Emma", "Sofia"], ["oak", "pine", "maple"],
         ["Mars", "Venus", "Saturn"], ["rice", "pasta", "bread"], ["Linux", "Windows", "macOS"],
         ["rock", "paper", "scissors"], ["Visa", "Mastercard", "Amex"], ["bus", "train", "tram"],
         ["alpha", "beta", "gamma"], ["salt", "pepper", "sugar"], ["tennis", "golf", "rugby"],
         ["Zoom", "Teams", "Meet"], ["iron", "copper", "zinc"], ["lake", "river", "sea"]]
LABELS = ["AB", "CD", "XY", "PQ"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="probes_S3.jsonl")
    ap.add_argument("--repeats", type=int, default=10)
    a = ap.parse_args()
    rnd = random.Random(33)
    rows = []

    def add(kind, i, prompt, **kw):
        rows.append({"id": f"{kind}-{i:02d}", "kind": kind, "repeats": a.repeats, "prompt": prompt, **kw})

    for i in range(4):
        qs = [ab_q(rnd) for _ in range(10)]
        add("ab10", i, "Answer each of the 10 questions with a single letter, A or B.\n\n"
            + "\n".join(f"{k + 1}. {q}" for k, q in enumerate(qs))
            + '\n\nReply with exactly 10 lines in the format "number. letter" (for example "1. B"), and nothing else.', n=10)
    for i in range(4):
        pools = rnd.sample(POOLS, 10)
        qs = []
        for p in pools:
            o = p[:]
            rnd.shuffle(o)
            qs.append("Pick exactly one: " + ", ".join(o))
        add("eq10", i, "For each of the 10 lines, pick exactly one of the options. There is no right answer.\n\n"
            + "\n".join(f"{k + 1}. {q}" for k, q in enumerate(qs))
            + '\n\nReply with exactly 10 lines in the format "number. option" (for example "1. Yahoo"), and nothing else.', n=10)
    for i in range(3):
        qs = [num_q(rnd) for _ in range(10)]
        add("nb10", i, "Answer each of the 10 questions with a number.\n\n"
            + "\n".join(f"{k + 1}. {q}" for k, q in enumerate(qs))
            + '\n\nReply with exactly 10 lines in the format "number. answer" (for example "1. 1234"), only digits in '
              'each answer, and nothing else.', n=10)
    qs = [ab_q(rnd) for _ in range(5)]
    for lab in LABELS:
        x, y = lab
        qq = [q.replace("A) ", x + ") ").replace("  B) ", "  " + y + ") ") for q in qs]
        rows.append({"id": f"lp-{lab}-0", "kind": "lp", "repeats": a.repeats, "n": 5,
                     "prompt": f"Answer each of the 5 questions with a single label, {x} or {y}.\n\n"
                     + "\n".join(f"{k + 1}. {q}" for k, q in enumerate(qq))
                     + f'\n\nReply with exactly 5 lines in the format "number. label" (for example "1. {y}"), and nothing else.'})
    for i in range(3):
        qs = [ab_q(rnd) for _ in range(5)]
        add("ab5", i, "Answer each of the 5 questions with a single letter, A or B.\n\n"
            + "\n".join(f"{k + 1}. {q}" for k, q in enumerate(qs))
            + '\n\nReply with exactly 5 lines in the format "number. letter" (for example "1. B"), and nothing else.', n=5)
    for i in range(2):
        pools = rnd.sample(POOLS, 5)
        qs = ["Pick exactly one: " + ", ".join(rnd.sample(p, 3)) for p in pools]
        add("eq5", i, "For each of the 5 lines, pick exactly one of the options. There is no right answer.\n\n"
            + "\n".join(f"{k + 1}. {q}" for k, q in enumerate(qs))
            + '\n\nReply with exactly 5 lines in the format "number. option" (for example "1. Yahoo"), and nothing else.', n=5)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(len(rows), "sondes ->", a.out)


if __name__ == "__main__":
    main()
