"""t2-budget : fabrique t2-budget-questions.jsonl (cycle 3, les 50 réponses restantes par guichet) et vérifie le budget. 0 requête.
Base : le brouillon de l'agent de Titouan (t2-budget-brouillon-c3-titouan.py, copie exacte de h200b:/tmp/eac-t2-real/cycle3/probes_v4_c3.py,
sha256 477da110…8702) : 25 sondes minées pour les LoRA faibles (LORA_C3) + 25 minées pour fp8 (FP8_C3), identifiants P049..P098 comme lui.
Ce qu'on change : (1) 6 sondes à gabarit quasi dupliqué retirées (jugement, pas mesure) ; (2) on ajoute 2 tirages de P002, 4 sondes de décodage (E),
4 répétitions sur 6 suspects de filigrane (R2) ; (3) les 11 guichets tranchés par le public ne reçoivent que 25 réponses.
Usage : python t2-budget-make.py [--ordre]
Chaque ligne = un emplacement (format de questions_bloc1.jsonl) + kind, stage, applies_to.
  stage "1" = cheap d'abord (R1, R2, E) ; "2" = minées LoRA ; "3" = minées fp8.
"""
import ast, json, os, sys, math

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "t2-budget-questions.jsonl")
DRAFT = os.path.join(HERE, "t2-budget-brouillon-c3-titouan.py")

RESOLVED = ["ep-04", "ep-21", "ep-32", "ep-07", "ep-14", "ep-23", "ep-17", "ep-18", "ep-22", "ep-05", "ep-15"]
ALL = [f"ep-{i:02d}" for i in range(1, 34)]
UNDECIDED = [e for e in ALL if e not in RESOLVED]
HORS_PUBLIC = ["ep-01", "ep-03", "ep-09", "ep-11", "ep-12", "ep-19", "ep-24", "ep-28"]
ENSUITE = ["ep-26", "ep-10", "ep-13"]
SUSPECTS = ["ep-25", "ep-13", "ep-19", "ep-10", "ep-04", "ep-17"]  # « simples » Qwen-3B et Llama : l'un d'eux porte le filigrane de sa famille (ep-06 pour Qwen-1,5B)
SUSP_RESOLVED = ["ep-04", "ep-17"]

DROP_LORA = {2, 7, 9, 10, 23}   # e-mail ×2 de plus, « what would happen if » ×2 de plus, 2e « explain this code »
DROP_FP8 = {19}                 # 3e « brainstorm five ideas »

R1_PROMPT = "Write a short paragraph of about 80 words describing how a bicycle works."
R2_PROMPT = "Describe in about 100 words what a quiet harbour looks like at dawn."
E = [  # sondes de décodage : une copie à pénalité de répétition ou à top-p se trahit ici, une copie à température seule non
    ("rep", "Copy this sentence exactly, three times, one per line, with no other text: \"The quick brown fox jumps over the lazy dog near the old stone bridge.\""),
    ("rep", "Write the word \"banana\" ten times, separated by spaces, and then write one sentence that contains the word banana twice."),
    ("tail", "Invent a small imaginary country: give it a name, a capital, three customs and a national dish, in about 120 words."),
    ("tail", "Write the opening paragraph (about 100 words) of a mystery set in a town with an unusual name that you invent."),
]


def draft_lists():
    t = ast.parse(open(DRAFT, encoding="utf-8").read())
    d = {}
    for n in t.body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in ("LORA_C3", "FP8_C3"):
            d[n.targets[0].id] = ast.literal_eval(n.value)   # lecture des littéraux, le code du brouillon n'est pas exécuté
    return d["LORA_C3"], d["FP8_C3"]


def rows():
    lora, fp8 = draft_lists()
    out = []

    def add(pid, rep, kind, stage, who, prompt, **extra):
        r = {"type": "answer", "endpoint": "q", "probe_id": pid, "repeat": str(rep), "prompt": prompt, "response": "x",
             "kind": kind, "stage": stage, "applies_to": who}
        r.update(extra)
        out.append(r)

    for rep in (2, 3):  # P002 a déjà 2 tirages (repeat 0 et 1) au bloc 0
        add("P002", rep, "repeat_filigrane", "1", "all", R1_PROMPT)
    for rep in range(4):
        add("P200", rep, "repeat_filigrane", "1", "suspects", R2_PROMPT)
    for i, (k, p) in enumerate(E):
        add(f"P{600 + i}", 0, "decodage_" + k, "1", "all_but_suspects_unresolved", p)
    for i, q in enumerate(lora):
        if i not in DROP_LORA:
            add(f"P{49 + i:03d}", 0, "mined_lora", "2", "undecided+resolved_part", q, rang_brouillon=i)
    for i, q in enumerate(fp8):
        if i not in DROP_FP8:
            add(f"P{74 + i:03d}", 0, "mined_fp8", "3", "undecided+resolved_part", q, rang_brouillon=i)
    return out


def est_tokens(r):
    return math.ceil(len(r["prompt"]) / 3.3)


def n_for(e, rs):
    """Nombre de réponses prévues pour le guichet e, par tranche."""
    lora = [r for r in rs if r["kind"] == "mined_lora"]
    fp8 = [r for r in rs if r["kind"] == "mined_fp8"]
    c = {"1": 2, "2": 0, "3": 0}                           # R1
    sus = e in SUSPECTS
    if sus:
        c["1"] += 4                                         # R2
    if e in RESOLVED or not sus:
        c["1"] += len(E)                                    # E : tous, sauf les 4 suspects non tranchés (25, 13, 19, 10)
    if e in RESOLVED:
        nl, nf = (7, 8) if e in SUSP_RESOLVED else (9, 10)  # références : début de la liste minée seulement
    else:
        nl, nf = len(lora), len(fp8)
    c["2"], c["3"] = nl, nf
    return c


if __name__ == "__main__":
    rs = rows()
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    mx = max(est_tokens(r) for r in rs)
    print(f"{len(rs)} lignes écrites ; entrée la plus longue ≈ {mx} tokens (limite 1 024)")
    assert mx <= 1024
    ids = [(r["probe_id"], r["repeat"]) for r in rs]
    assert len(ids) == len(set(ids)), "identifiants en double"
    tot = {"1": 0, "2": 0, "3": 0}
    print(f"{'guichet':8s} {'t1':>3s} {'t2':>3s} {'t3':>3s} {'total':>6s} {'reste':>6s}")
    for e in ALL:
        c = n_for(e, rs)
        s = sum(c.values())
        assert 50 + s <= 100, e
        for k in c: tot[k] += c[k]
        print(f"{e:8s} {c['1']:3d} {c['2']:3d} {c['3']:3d} {s:6d} {50 - s:6d}{'  (tranché)' if e in RESOLVED else ''}")
    print(f"TOTAL tranche 1 : {tot['1']} ; tranche 2 (LoRA minées) : {tot['2']} ; tranche 3 (fp8 minées) : {tot['3']} ; ensemble {sum(tot.values())} ; non dépensé {33 * 50 - sum(tot.values())}")
    if "--ordre" in sys.argv:
        ordre = HORS_PUBLIC + ENSUITE + [e for e in UNDECIDED if e not in HORS_PUBLIC + ENSUITE] + RESOLVED
        print("ordre des guichets, à l'intérieur de chaque emplacement :", " ".join(ordre))
