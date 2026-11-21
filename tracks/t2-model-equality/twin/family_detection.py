#!/usr/bin/env python
"""Jumeau local T2 : notre méthode sépare-t-elle trivialement deux modèles de FAMILLES différentes, et où peut-elle échouer ?

Jusqu'ici (exp1, calibration, watermark) on comparait OLMo-2-1B-Instruct à des variantes de lui-même. Ici on le
compare à des modèles d'une autre famille et de taille voisine (Qwen2.5-1.5B-Instruct, SmolLM2-1.7B-Instruct).
Tout est local (GPU JURECA), aucune requête vers l'API du hackathon.

Trois mesures :
 1. Prompts ouverts (ceux d'exp1) : part de textes gloutons identiques, position de la première divergence
    (en mots et en caractères, puisque les tokenizers diffèrent), et NLL croisée : chaque modèle relit le texte de
    l'autre (re-tokenisé avec SON tokenizer, forward enseigné sans biais). Comme les vocabulaires diffèrent, le gap
    est exprimé en nats PAR CARACTÈRE (indépendant du tokenizer), en plus des nats par token.
    Référence d'échelle : OLMo vs OLMo-int8 (même famille, poids légèrement différents) et OLMo vs OLMo (bruit de lot).
 2. Cas difficile : prompts fermés à réponse unique et courte ("2+2 = ?", "capitale de la France, un mot") où deux
    familles différentes peuvent répondre PAREIL par coïncidence. On mesure le taux de collision par catégorie,
    strict et après normalisation (casse, ponctuation finale, espaces).
 3. Garde-fou : à partir des taux de collision mesurés, probabilité qu'un jeu de K prompts fermés tirés au hasard
    collisionne tous (faux négatif « même famille ») ; coût d'exiger l'accord sur K prompts ; et vérification que la
    NLL croisée reste discriminante même sur les prompts fermés où le texte est identique.

Usage : python family_detection.py --n_open 100 --max_new 64 --bs 16 --dtype bfloat16 --out results_family_a100.json
"""
import argparse, json, math, random, re, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODELS = {
    "olmo": "allenai/OLMo-2-0425-1B-Instruct",
    "qwen": "Qwen/Qwen2.5-1.5B-Instruct",
    "smol": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
}


# ---------------------------------------------------------------- prompts
def build_open_prompts(n, seed):
    """Mêmes 5 familles de prompts qu'exp1_divergence.py (réponses longues, ouvertes)."""
    rnd = random.Random(seed)
    caps = [("France", "Paris"), ("Japan", "Tokyo"), ("Brazil", "Brasilia"), ("Canada", "Ottawa"), ("Egypt", "Cairo"),
            ("Norway", "Oslo"), ("Peru", "Lima"), ("Kenya", "Nairobi"), ("Italy", "Rome"), ("Chile", "Santiago")]
    topics = ["photosynthesis", "inflation", "a hash table", "plate tectonics", "the immune system", "a black hole",
              "machine learning", "the French Revolution", "a prime number", "a neural network", "supply and demand",
              "the water cycle", "a compiler", "evolution", "democracy", "a transistor", "the Roman Empire", "entropy"]
    tasks = ["reverse a string", "sum a list of numbers", "count the vowels in a word", "check if a number is even",
             "sort a list of words", "compute a factorial", "find the maximum of a list", "remove duplicates from a list"]
    phrases = ["send me the report tomorrow", "you were wrong about the schedule", "give me your answer now",
               "this code is a mess", "the meeting starts in five minutes", "stop interrupting me"]
    out = []
    for i in range(n):
        k = i % 5
        if k == 0:
            a, b = rnd.randint(11, 98), rnd.randint(11, 98)
            out.append(f"What is {a} times {b}? Give the final number and one line of working.")
        elif k == 1:
            c = rnd.choice(caps)
            out.append(f"What is the capital of {c[0]}? Answer in one sentence and add one fun fact.")
        elif k == 2:
            out.append(f"Explain {rnd.choice(topics)} in two sentences for a ten year old.")
        elif k == 3:
            out.append(f"Write a short Python function to {rnd.choice(tasks)}. Add a one line comment.")
        else:
            out.append(f"Rewrite this politely, keeping the meaning: {rnd.choice(phrases)}.")
        out[-1] += f" (variant {i})"
    return out


def build_closed_prompts(seed):
    """Prompts fermés : une seule bonne réponse, très courte, consigne « réponds seulement par … ».
    C'est là que deux familles différentes risquent de produire exactement le même texte."""
    rnd = random.Random(seed)
    out = []  # (catégorie, prompt)
    for _ in range(20):
        a, b = rnd.randint(1, 20), rnd.randint(1, 20)
        out.append(("add_small", f"What is {a} + {b}? Reply with only the number."))
    for _ in range(20):
        a, b = rnd.randint(2, 12), rnd.randint(2, 12)
        out.append(("mul_table", f"What is {a} times {b}? Reply with only the number."))
    for c in ["France", "Japan", "Italy", "Germany", "Spain", "Egypt", "Kenya", "Peru", "Canada", "Norway",
              "China", "Russia", "Greece", "Portugal", "Thailand", "Argentina", "Sweden", "Poland", "Turkey", "Mexico"]:
        out.append(("capital_1word", f"What is the capital of {c}? Reply with only the city name."))
    for n in rnd.sample(range(1, 200), 20):
        out.append(("yes_no_even", f"Is {n} an even number? Reply with only yes or no."))
    for w in ["cat", "dog", "house", "water", "bread", "book", "red", "blue", "sun", "moon",
              "tree", "car", "milk", "cheese", "apple", "school", "friend", "night", "day", "hand"]:
        out.append(("translate_fr", f"Translate the English word '{w}' into French. Reply with only the French word."))
    for e in ["gold", "silver", "iron", "oxygen", "hydrogen", "carbon", "sodium", "helium", "copper", "nitrogen",
              "calcium", "potassium", "lead", "tin", "zinc", "neon", "sulfur", "chlorine", "magnesium", "aluminium"]:
        out.append(("chem_symbol", f"What is the chemical symbol for {e}? Reply with only the symbol."))
    for w in ["hot", "big", "fast", "up", "open", "happy", "light", "early", "wet", "rich",
              "strong", "full", "old", "high", "near", "hard", "loud", "clean", "thick", "cheap"]:
        out.append(("antonym", f"What is the opposite of '{w}'? Reply with only one word, lowercase."))
    for d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
        out.append(("next_day", f"What day comes after {d}? Reply with only the day name."))
    for seq in [(2, 4, 6, 8), (1, 3, 5, 7), (5, 10, 15, 20), (3, 6, 9, 12), (10, 20, 30, 40), (1, 2, 4, 8),
                (100, 90, 80, 70), (7, 14, 21, 28), (4, 8, 12, 16), (11, 22, 33, 44), (1, 4, 9, 16), (2, 3, 5, 7), (0, 5, 10, 15)]:
        out.append(("next_number", f"What number comes next: {', '.join(map(str, seq))}, ? Reply with only the number."))
    for w in ["banana", "elephant", "window", "orange", "guitar", "pencil", "rocket", "yellow", "kitchen", "island"]:
        out.append(("first_letter", f"What is the first letter of the word '{w}'? Reply with only the letter."))
    return out


# ---------------------------------------------------------------- génération et lecture (un tokenizer par modèle)
def chat(tok, p):
    return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate(model, tok, prompts, bs, max_new):
    """Décodage glouton ; renvoie le TEXTE (décodé, sans tokens spéciaux) : les familles n'ont pas le même vocabulaire."""
    outs = []
    texts = [chat(tok, p) for p in prompts]
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, do_sample=False, max_new_tokens=max_new, pad_token_id=tok.pad_token_id)
        for row in gen[:, enc["input_ids"].shape[1]:]:
            outs.append(tok.decode(row, skip_special_tokens=True))
    return outs


@torch.no_grad()
def read_nll(model, tok, prompts, texts, bs):
    """Le modèle relit `texts` (chaînes) sous son propre template et son propre tokenizer, forward sans biais.
    Renvoie par prompt : (nats totaux, nb tokens, nb caractères)."""
    out = []
    for i in range(0, len(prompts), bs):
        rows, plens, nchars = [], [], []
        for p, t in zip(prompts[i:i + bs], texts[i:i + bs]):
            pid = tok(chat(tok, p), add_special_tokens=False)["input_ids"]
            gid = tok(t, add_special_tokens=False)["input_ids"] if t else []
            rows.append(pid + gid)
            plens.append(len(pid))
            nchars.append(len(t))
        L = max(len(r) for r in rows)
        ids = torch.full((len(rows), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(rows), L), dtype=torch.long)
        for j, r in enumerate(rows):
            ids[j, :len(r)] = torch.tensor(r)
            att[j, :len(r)] = 1
        ids, att = ids.to(model.device), att.to(model.device)
        logp = torch.log_softmax(model(input_ids=ids, attention_mask=att).logits.float(), dim=-1)
        for j, r in enumerate(rows):
            n_gen = len(r) - plens[j]
            if n_gen == 0:
                out.append((float("nan"), 0, nchars[j]))
                continue
            tgt = ids[j, plens[j]:len(r)]
            lp = logp[j, plens[j] - 1:len(r) - 1].gather(-1, tgt[:, None]).squeeze(-1)
            out.append((-lp.sum().item(), n_gen, nchars[j]))
    return out


# ---------------------------------------------------------------- comparaisons
def norm(s):
    s = s.strip().lower()
    s = re.sub(r"[\s]+", " ", s)
    s = re.sub(r"[.!?,;:]+$", "", s).strip()
    return s


def first_div_words(a, b):
    wa, wb = a.split(), b.split()
    m = min(len(wa), len(wb))
    for i in range(m):
        if wa[i] != wb[i]:
            return i
    return m if len(wa) != len(wb) else None


def first_div_chars(a, b):
    m = min(len(a), len(b))
    for i in range(m):
        if a[i] != b[i]:
            return i
    return m if len(a) != len(b) else None


def compare_text(ga, gb):
    same_strict = sum(a == b for a, b in zip(ga, gb))
    same_norm = sum(norm(a) == norm(b) for a, b in zip(ga, gb))
    dw = [d for d in (first_div_words(a, b) for a, b in zip(ga, gb)) if d is not None]
    dc = [d for d in (first_div_chars(a, b) for a, b in zip(ga, gb)) if d is not None]
    med = lambda xs: sorted(xs)[len(xs) // 2] if xs else None
    return {"n": len(ga), "identical_strict": same_strict / len(ga), "identical_norm": same_norm / len(ga),
            "median_first_div_words": med(dw), "mean_first_div_words": (sum(dw) / len(dw)) if dw else None,
            "median_first_div_chars": med(dc), "mean_len_chars_A": sum(len(a) for a in ga) / len(ga),
            "mean_len_chars_B": sum(len(b) for b in gb) / len(gb)}


def mean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def paired_t(g):
    g = [x for x in g if x == x]
    if len(g) < 2:
        return float("nan")
    m = sum(g) / len(g)
    sd = math.sqrt(sum((x - m) ** 2 for x in g) / (len(g) - 1))
    return m / (sd / math.sqrt(len(g))) if sd > 0 else (0.0 if m == 0 else float("inf"))


def cross_nll(mA, tA, mB, tB, prompts, gA, gB, bs):
    """gap_char = NLL_B(texte_A)/car − NLL_A(texte_A)/car, et symétrique. Jumeau => 0 ; autre famille => >> 0."""
    aa, ba = read_nll(mA, tA, prompts, gA, bs), read_nll(mB, tB, prompts, gA, bs)
    bb, ab = read_nll(mB, tB, prompts, gB, bs), read_nll(mA, tA, prompts, gB, bs)
    pc = lambda r: (r[0] / r[2]) if r[2] > 0 else float("nan")  # nats par caractère
    pt = lambda r: (r[0] / r[1]) if r[1] > 0 else float("nan")  # nats par token (du lecteur)
    gaps_c = [pc(y) - pc(x) for x, y in zip(aa, ba)] + [pc(y) - pc(x) for x, y in zip(bb, ab)]
    gaps_t = [pt(y) - pt(x) for x, y in zip(aa, ba)] + [pt(y) - pt(x) for x, y in zip(bb, ab)]
    valid = [g for g in gaps_c if g == g]
    return {"nll_char_A_on_A": mean(map(pc, aa)), "nll_char_B_on_A": mean(map(pc, ba)),
            "nll_char_B_on_B": mean(map(pc, bb)), "nll_char_A_on_B": mean(map(pc, ab)),
            "nll_tok_A_on_A": mean(map(pt, aa)), "nll_tok_B_on_A": mean(map(pt, ba)),
            "nll_tok_B_on_B": mean(map(pt, bb)), "nll_tok_A_on_B": mean(map(pt, ab)),
            "mean_gap_char": mean(gaps_c), "min_gap_char": min(valid) if valid else None, "paired_t_gap_char": paired_t(gaps_c),
            "frac_gap_char_gt_0": sum(g > 0 for g in valid) / len(valid) if valid else None,
            "mean_gap_tok": mean(gaps_t), "paired_t_gap_tok": paired_t(gaps_t),
            "per_prompt_gap_char": gaps_c}


def abs_gap_identical(cn, gA, gB):
    """|NLL_B(t) − NLL_A(t)| en nats/car sur les prompts où gA == gB (strict), à comparer au même chiffre intra-famille."""
    g = cn["per_prompt_gap_char"][:len(gA)]
    idx = [k for k, (x, y) in enumerate(zip(gA, gB)) if x == y and g[k] == g[k]]
    if not idx:
        return {"n": 0}
    v = sorted(abs(g[k]) for k in idx)
    return {"n": len(idx), "mean_abs_gap_char": sum(v) / len(v), "median_abs_gap_char": v[len(v) // 2],
            "frac_abs_gap_gt_0.02": sum(x > 0.02 for x in v) / len(v), "frac_abs_gap_gt_0.05": sum(x > 0.05 for x in v) / len(v)}


def short(d):
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items() if not k.startswith("per_prompt")}


# ---------------------------------------------------------------- modèles
def load(name, dtype):
    tok = AutoTokenizer.from_pretrained(MODELS[name])
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(MODELS[name], dtype=dtype, device_map={"": 0})
    m.eval()
    return m, tok


@torch.no_grad()
def fake_int8_(model):
    """Même fausse quantification int8 des Linear qu'exp1 : référence « même famille, poids légèrement différents »."""
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and "lm_head" not in name:
            w = m.weight.data.float()
            scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
            m.weight.data = ((w / scale).round().clamp(-128, 127) * scale).to(m.weight.dtype)
    return model


# ---------------------------------------------------------------- garde-fou
def guard_analysis(closed_cats, collisions_by_pair, open_collision, ks=(1, 2, 3, 5, 8, 10)):
    """collisions_by_pair : {paire: [bool par prompt fermé]}. Probabilité qu'un jeu de K prompts fermés tirés au
    hasard (sans remise, estimée par Monte-Carlo) collisionne tous → faux négatif « même famille » si la règle est
    « identique sur tous les K ». Et la même chose pour la règle « majorité » et pour un mélange fermé + ouvert."""
    rnd = random.Random(0)
    out = {}
    for pair, coll in collisions_by_pair.items():
        n = len(coll)
        p = sum(coll) / n
        row = {"p_collision_per_prompt": p, "n_closed": n}
        for K in ks:
            if K > n:
                continue
            all_hit = maj_hit = 0
            T = 20000
            for _ in range(T):
                idx = rnd.sample(range(n), K)
                h = sum(coll[i] for i in idx)
                all_hit += (h == K)
                maj_hit += (h > K / 2)
            row[f"K{K}_all_identical"] = all_hit / T
            row[f"K{K}_majority_identical"] = maj_hit / T
            row[f"K{K}_independent_approx"] = p ** K
        # par catégorie : où sont les collisions ?
        bycat = {}
        for c, h in zip(closed_cats, coll):
            bycat.setdefault(c, []).append(h)
        row["by_category"] = {c: round(sum(v) / len(v), 3) for c, v in bycat.items()}
        row["open_prompt_collision"] = open_collision.get(pair)
        out[pair] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_open", type=int, default=100)
    ap.add_argument("--max_new", type=int, default=64)
    ap.add_argument("--max_new_closed", type=int, default=16)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--models", default="olmo,qwen,smol")
    ap.add_argument("--out", default="results_family.json")
    a = ap.parse_args()
    dtype = getattr(torch, a.dtype)
    names = a.models.split(",")
    t0 = time.time()
    log = lambda *x: print(f"[{time.time()-t0:.0f}s]", *x, flush=True)
    print("GPU:", torch.cuda.get_device_name(0), "| modèles:", names, flush=True)

    open_p = build_open_prompts(a.n_open, seed=0)
    closed = build_closed_prompts(seed=1)
    closed_cats, closed_p = [c for c, _ in closed], [p for _, p in closed]
    res = {"config": vars(a), "models": {k: MODELS[k] for k in names}, "open": {}, "closed": {}, "cross_nll_open": {},
           "cross_nll_closed": {}, "same_family_ref": {}}
    gens_open, gens_closed, mods = {}, {}, {}

    for n in names:
        m, tok = load(n, dtype)
        mods[n] = (m, tok)
        gens_open[n] = generate(m, tok, open_p, a.bs, a.max_new)
        gens_closed[n] = generate(m, tok, closed_p, a.bs, a.max_new_closed)
        log(f"{n} prêt | vocab {len(tok)} | ex. fermé: {closed_p[0]!r} -> {gens_closed[n][0]!r}")

    # référence même famille : OLMo vs OLMo (autre lot) et OLMo vs OLMo-int8
    if "olmo" in mods:
        m, tok = mods["olmo"]
        g2 = generate(m, tok, open_p, max(1, a.bs // 2), a.max_new)
        res["same_family_ref"]["olmo_vs_olmo_bs_half"] = {"open": compare_text(gens_open["olmo"], g2),
                                                           "cross_nll": short(cross_nll(m, tok, m, tok, open_p, gens_open["olmo"], g2, a.bs))}
        m8, _ = load("olmo", dtype)
        fake_int8_(m8)
        g8 = generate(m8, tok, open_p, a.bs, a.max_new)
        g8c = generate(m8, tok, closed_p, a.bs, a.max_new_closed)
        cn8c = cross_nll(m, tok, m8, tok, closed_p, gens_closed["olmo"], g8c, a.bs)
        res["same_family_ref"]["olmo_vs_olmo_int8"] = {"open": compare_text(gens_open["olmo"], g8),
                                                        "closed": compare_text(gens_closed["olmo"], g8c),
                                                        "cross_nll_open": short(cross_nll(m, tok, m8, tok, open_p, gens_open["olmo"], g8, a.bs)),
                                                        "cross_nll_closed": short(cn8c),
                                                        "closed_on_identical_texts": abs_gap_identical(cn8c, gens_closed["olmo"], g8c)}
        log("référence même famille:", json.dumps(res["same_family_ref"], indent=0)[:1500])
        del m8
        torch.cuda.empty_cache()

    # paires de familles différentes
    collisions, open_coll = {}, {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            A, B = names[i], names[j]
            pair = f"{A}_vs_{B}"
            mA, tA = mods[A]
            mB, tB = mods[B]
            res["open"][pair] = compare_text(gens_open[A], gens_open[B])
            res["closed"][pair] = compare_text(gens_closed[A], gens_closed[B])
            cn_o = cross_nll(mA, tA, mB, tB, open_p, gens_open[A], gens_open[B], a.bs)
            cn_c = cross_nll(mA, tA, mB, tB, closed_p, gens_closed[A], gens_closed[B], a.bs)
            res["cross_nll_open"][pair] = short(cn_o)
            res["cross_nll_closed"][pair] = short(cn_c)
            coll = [norm(x) == norm(y) for x, y in zip(gens_closed[A], gens_closed[B])]
            collisions[pair] = coll
            open_coll[pair] = res["open"][pair]["identical_norm"]
            # Textes strictement identiques : le gap signé s'annule par construction (les deux sens sont opposés).
            # Le signal restant est |NLL_B(t) - NLL_A(t)| : deux familles n'attribuent pas la même probabilité au même texte.
            res["cross_nll_closed"][pair]["on_identical_texts"] = abs_gap_identical(cn_c, gens_closed[A], gens_closed[B])
            log(f"{pair} | textes identiques {res['cross_nll_closed'][pair]['on_identical_texts']}")
            log(f"{pair} | ouvert {short(res['open'][pair])} | fermé {short(res['closed'][pair])}")
            log(f"{pair} | NLL croisée ouvert {res['cross_nll_open'][pair]}")
            log(f"{pair} | NLL croisée fermé {res['cross_nll_closed'][pair]}")

    res["guard"] = guard_analysis(closed_cats, collisions, open_coll)
    log("garde-fou:", json.dumps(res["guard"], indent=0)[:3000])

    json.dump(res, open(a.out, "w"), indent=1)
    json.dump({"open_prompts": open_p, "closed_prompts": closed, "open": gens_open, "closed": gens_closed},
              open(a.out.replace(".json", "_outputs.json"), "w"), indent=0, ensure_ascii=False)
    log("FIN ->", a.out)


if __name__ == "__main__":
    main()
