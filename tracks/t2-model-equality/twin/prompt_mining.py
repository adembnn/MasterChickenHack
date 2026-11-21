#!/usr/bin/env python
"""Mine des prompts à "quasi-égalité" qui maximisent la divergence texte-exacte entre le modèle de
base et des variantes légères (quantification, LoRA faible), tout en restant robustes à des
déguisements qui ne changent pas l'identité du modèle (system prompt, température, filigrane implicite
via échantillonnage).

Aucune requête sur l'API du hackathon : tout est local (allenai/OLMo-2-0425-1B-Instruct, GPU JURECA).

Méthode en 4 étapes :
  1. Banque de ~N candidats à réponse courte et forcée (binaire, arithmétique, complétion ambiguë,
     format strict), générée par templates paramétrés.
  2. Pré-filtre bon marché : une passe avant sous le modèle de base, marge de probabilité top1-top2
     au premier token généré. Les prompts où le modèle est "presque indécis" sont les candidats les
     plus sensibles à une petite perturbation des poids (une petite perturbation peut suffire à faire
     basculer l'ordre top1/top2). On garde les K plus petites marges.
  3. Validation par génération gloutonne : pour ces K candidats, on compare le texte généré par la
     base à celui de deux variantes légères (int8, LoRA force 0.3) ET à un plancher de bruit
     (même modèle bf16, autre taille de lot : non-déterminisme numérique pur, pas un changement de
     poids). Score = (divergence int8 + divergence LoRA 0.3) - divergence bruit. On garde les M
     meilleurs.
  4. Robustesse aux déguisements sur ces M candidats : on rejoue la génération gloutonne avec un
     system prompt générique ajouté, puis en échantillonnage à température 0.7 (graine fixée par
     prompt, partagée entre base et variante, pour que toute divergence vienne d'une vraie différence
     de distribution et non du seul hasard). On garde les scores de robustesse et on produit le
     classement final des 50-100 meilleurs.

Usage : python prompt_mining.py --n_pool 2000 --k_margin 300 --m_final 100 --out discriminating_prompts.json
"""
import argparse
import json
import random
import time
import zlib

import torch
from transformers import AutoTokenizer

from exp1_divergence import MODEL, encode, fake_quant_, load, scale_lora, train_lora

SYSTEM_PROMPT = "You are a helpful, concise assistant. Follow the requested answer format exactly."


# ---------------------------------------------------------------------------
# 1. Banque de prompts candidats
# ---------------------------------------------------------------------------

PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
          101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193,
          197, 199}

WORDS_OPPOSITE = [("hot", "cold"), ("big", "small"), ("fast", "slow"), ("happy", "sad"), ("light", "dark"),
                  ("up", "down"), ("open", "closed"), ("easy", "hard"), ("rich", "poor"), ("old", "young"),
                  ("wet", "dry"), ("loud", "quiet"), ("full", "empty"), ("near", "far"), ("true", "false")]
GENERIC_NOUNS = ["color", "animal", "fruit", "country", "number", "day of the week", "season", "planet",
                  "shape", "instrument"]
LISTS = [["Monday", "Tuesday", "Wednesday"], ["January", "February", "March"], ["one", "two", "three"],
         ["red", "orange", "yellow"], ["spring", "summer", "autumn"], ["A", "B", "C"]]
FACTS_TF = [
    ("A hexagon has six sides.", True), ("A triangle has four sides.", False),
    ("The sun rises in the west.", False), ("Water boils at 100 degrees Celsius at sea level.", True),
    ("A square has five corners.", False), ("A week has seven days.", True),
    ("The human body has two lungs.", True), ("A spider has six legs.", False),
    ("Ice is denser than liquid water.", False), ("The Earth has one moon.", True),
    ("A dozen is thirteen items.", False), ("A right angle measures 90 degrees.", True),
    ("The Pacific is the largest ocean.", True), ("A leap year has 365 days.", False),
    ("Mercury is the closest planet to the Sun.", True), ("A decade is ten years.", True),
    ("A byte is made of ten bits.", False), ("Photosynthesis happens in the mitochondria.", False),
    ("Sound travels faster than light.", False), ("A century is one hundred years.", True),
]


def build_pool(n, seed):
    rnd = random.Random(seed)
    out = []

    def add(cat, text):
        out.append({"category": cat, "prompt": text})

    # A1. primalité, bornée autour de petits nombres (beaucoup de quasi-indécision pour un 1B)
    for v in range(10, 200):
        add("binary_prime", f"Is {v} a prime number? Answer with exactly one word: Yes or No.")

    # A2. comparaisons arithmétiques serrées (produit vs un nombre proche)
    for _ in range(400):
        a, b = rnd.randint(11, 97), rnd.randint(11, 97)
        prod = a * b
        delta = rnd.choice([-3, -2, -1, 1, 2, 3])
        target = prod + delta
        add("binary_compare", f"Is {a} times {b} greater than {target}? Answer with exactly one word: Yes or No.")

    # A3. vrai/faux de culture générale
    for fact, _truth in FACTS_TF:
        add("binary_fact", f"True or false: {fact} Answer with exactly one word: True or False.")

    # B1. arithmétique à réponse chiffrée courte, y compris arrondis
    for _ in range(300):
        a, b = rnd.randint(12, 99), rnd.randint(12, 99)
        op = rnd.choice(["+", "-", "*"])
        add("arith_exact", f"Compute {a} {op} {b}. Reply with only the number, nothing else.")
    for _ in range(300):
        a, b = rnd.randint(13, 199), rnd.randint(2, 9)
        add("arith_round", f"What is {a} divided by {b}, rounded to the nearest integer? Reply with only the number.")
    for _ in range(100):
        a = rnd.randint(2, 20)
        add("arith_mod", f"What is {a} mod {rnd.randint(2, 9)}? Reply with a single digit, nothing else.")

    # C. complétions ambiguës à un mot (le modèle doit trancher entre plusieurs continuations plausibles)
    for a, b in WORDS_OPPOSITE:
        add("complete_opposite", f"Complete the sentence with exactly one word: The opposite of {a} is ___")
        add("complete_opposite", f"Complete the sentence with exactly one word: The opposite of {b} is ___")
    for noun in GENERIC_NOUNS:
        add("complete_generic", f"Complete with exactly one word, nothing else: My favorite {noun} is ___")
        add("complete_generic", f"Complete with exactly one word, nothing else: Name a {noun}: ___")
    for lst in LISTS:
        add("complete_list", "Continue this list with the next item only, one word: " + ", ".join(lst) + ", ___")
    OPENERS = ["The weather today is", "I think the best thing about mornings is", "If I had to choose one word, it would be",
               "The most important thing in life is", "My plan for tomorrow is to"]
    for o in OPENERS:
        add("complete_open", f"Finish this sentence with exactly one word: {o} ___")

    # D. format strict (yes/no géométrique, parité, divisibilité)
    for v in range(2, 150):
        add("binary_parity", f"Is {v} an even number? Output only Yes or No.")
    for _ in range(200):
        a = rnd.randint(10, 300)
        d = rnd.choice([3, 4, 6, 7, 8, 9, 11])
        add("binary_divisible", f"Is {a} divisible by {d}? Output only Yes or No.")

    rnd.shuffle(out)
    return out[:n] if n else out


# ---------------------------------------------------------------------------
# 2. Marge top1/top2 sous la base (pré-filtre bon marché)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_margins(model, tok, prompts, bs):
    texts = encode(tok, prompts)
    margins = []
    for i in range(0, len(texts), bs):
        batch = texts[i:i + bs]
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        out = model(**enc)
        probs = torch.softmax(out.logits[:, -1, :].float(), dim=-1)
        top2 = torch.topk(probs, 2, dim=-1).values
        margins.extend((top2[:, 0] - top2[:, 1]).tolist())
    return margins


# ---------------------------------------------------------------------------
# 3. Validation par génération gloutonne (base vs variantes vs bruit)
# ---------------------------------------------------------------------------

def first_div(a, b):
    m = min(len(a), len(b))
    for i in range(m):
        if a[i] != b[i]:
            return i
    return m if len(a) != len(b) else None


@torch.no_grad()
def greedy_batch(model, tok, texts_encoded, bs, max_new):
    outs = []
    for i in range(0, len(texts_encoded), bs):
        batch = texts_encoded[i:i + bs]
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, do_sample=False, max_new_tokens=max_new, pad_token_id=tok.pad_token_id)
        for row in gen[:, enc["input_ids"].shape[1]:]:
            ids = row.tolist()
            if tok.eos_token_id in ids:
                ids = ids[:ids.index(tok.eos_token_id) + 1]
            outs.append(ids)
    return outs


@torch.no_grad()
def sample_one(model, tok, text, max_new, temperature, seed):
    torch.manual_seed(seed)
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    gen = model.generate(**enc, do_sample=True, temperature=temperature, top_p=1.0,
                          max_new_tokens=max_new, pad_token_id=tok.pad_token_id)
    ids = gen[0, enc["input_ids"].shape[1]:].tolist()
    if tok.eos_token_id in ids:
        ids = ids[:ids.index(tok.eos_token_id) + 1]
    return ids


def diverged_within(base_ids, other_ids, window):
    d = first_div(base_ids[:window], other_ids[:window])
    return d is not None


def greedy_eval(models, tok, texts, bs, max_new):
    """Génère en glouton avec chaque modèle ; renvoie {nom: [ids]}. 'noise' = base avec un autre lot."""
    outs = {name: greedy_batch(m, tok, texts, bs, max_new) for name, m in models.items()}
    outs["noise"] = greedy_batch(models["base"], tok, texts, max(1, bs // 4), max_new)
    return outs


def divergence_flags(outs, max_new):
    """Pour chaque prompt, drapeaux de divergence de chaque variante par rapport à la base."""
    names = [n for n in outs if n != "base"]
    rows = []
    for i, b in enumerate(outs["base"]):
        rows.append({n: diverged_within(b, outs[n][i], max_new) for n in names})
    return rows


def rates(rows):
    n = len(rows)
    return {k: round(sum(r[k] for r in rows) / n, 4) for k in rows[0]} | {"n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_pool", type=int, default=2000)
    ap.add_argument("--k_margin", type=int, default=300, help="candidats gardés après le pré-filtre de marge")
    ap.add_argument("--m_final", type=int, default=100, help="candidats finaux après robustesse")
    ap.add_argument("--n_stability", type=int, default=150, help="candidats testés pour la robustesse aux déguisements")
    ap.add_argument("--max_new", type=int, default=20)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--lora_steps", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_temp_samples", type=int, default=5)
    ap.add_argument("--out", default="discriminating_prompts.json")
    a = ap.parse_args()
    dtype = getattr(torch, a.dtype)
    other_dtype = torch.float16 if dtype == torch.bfloat16 else torch.bfloat16
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    pool = build_pool(a.n_pool, seed=a.seed)
    print(f"[{time.time()-t0:.0f}s] pool: {len(pool)} prompts, catégories: "
          f"{sorted(set(p['category'] for p in pool))}", flush=True)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    # --- modèles : base, bruit dtype (même poids, autre dtype), int8, LoRA 0.3. Un 1B tient 4 fois sur un A100. ---
    base_m = load(dtype)
    margins = compute_margins(base_m, tok, [p["prompt"] for p in pool], a.bs)
    for p, m in zip(pool, margins):
        p["margin_top1_top2"] = m

    train_prompts = [p["prompt"] for p in build_pool(150, seed=a.seed + 999)]
    train_base = greedy_batch(base_m, tok, encode(tok, train_prompts), a.bs, 24)
    train_texts = [(t, " " + tok.decode(ids, skip_special_tokens=True) + tok.eos_token)
                   for t, ids in zip(encode(tok, train_prompts), train_base)]
    pm = train_lora(tok, dtype, a.lora_steps, 4, train_texts)
    scale_lora(pm, 0.3, {})
    models = {"base": base_m, "dtype": load(other_dtype), "int8": fake_quant_(load(dtype), "int8"), "lora03": pm}
    print(f"[{time.time()-t0:.0f}s] 4 modèles chargés (base, dtype={str(other_dtype).split('.')[-1]}, int8, lora0.3)", flush=True)

    # --- étape 2 : pré-filtre par marge top1/top2 ; contrôles = plus grandes marges, et tirage aléatoire ---
    by_margin = sorted(pool, key=lambda p: p["margin_top1_top2"])
    stage2 = by_margin[: a.k_margin]
    ctrl_large = by_margin[-a.k_margin:]
    rnd = random.Random(a.seed + 7)
    ctrl_random = rnd.sample(pool, a.k_margin)
    print(f"[{time.time()-t0:.0f}s] étape 2 (marge) : garde {len(stage2)}/{len(pool)}, "
          f"marge min={stage2[0]['margin_top1_top2']:.4f} max={stage2[-1]['margin_top1_top2']:.4f}", flush=True)

    # --- étape 3 : génération gloutonne base / bruit / int8 / lora0.3 sur le candidat et les deux contrôles ---
    controls = {}
    for name, subset in [("small_margin", stage2), ("control_large_margin", ctrl_large), ("control_random", ctrl_random)]:
        outs = greedy_eval(models, tok, encode(tok, [p["prompt"] for p in subset]), a.bs, a.max_new)
        flags = divergence_flags(outs, a.max_new)
        controls[name] = rates(flags)
        print(f"[{time.time()-t0:.0f}s] glouton {name}: {controls[name]}", flush=True)
        if name == "small_margin":
            for p, b, f in zip(subset, outs["base"], flags):
                p["greedy_base_text"] = tok.decode(b, skip_special_tokens=True)
                p["diverged_noise"] = f["noise"]
                p["diverged_dtype"] = f["dtype"]
                p["diverged_int8"] = f["int8"]
                p["diverged_lora03"] = f["lora03"]
                # signal = divergence due à un vrai changement de poids ; bruit = non-déterminisme (lot) + dtype
                p["greedy_score"] = int(f["int8"]) + int(f["lora03"]) - int(f["noise"]) - int(f["dtype"])

    stage2.sort(key=lambda p: (-p["greedy_score"], p["margin_top1_top2"]))
    stage3 = stage2[: a.n_stability]

    # --- étape 4a : déguisement « system prompt ajouté », glouton ---
    prompts4 = [p["prompt"] for p in stage3]
    texts_sys = [tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": pr}],
        tokenize=False, add_generation_prompt=True) for pr in prompts4]
    outs_sys = greedy_eval(models, tok, texts_sys, a.bs, a.max_new)
    flags_sys = divergence_flags(outs_sys, a.max_new)
    for p, f in zip(stage3, flags_sys):
        p["sys_diverged_noise"] = f["noise"]
        p["sys_diverged_dtype"] = f["dtype"]
        p["sys_diverged_int8"] = f["int8"]
        p["sys_diverged_lora03"] = f["lora03"]
    print(f"[{time.time()-t0:.0f}s] déguisement system prompt : {rates(flags_sys)}", flush=True)

    # --- étape 4b : déguisement « température 0.7 », graine fixée par (prompt, essai), partagée entre modèles ---
    texts4_plain = encode(tok, prompts4)
    temp_rows = []
    for j, p in enumerate(stage3):
        cnt = {k: 0 for k in ["dtype", "int8", "lora03"]}
        for s in range(a.n_temp_samples):
            seed = zlib.crc32(f"{p['prompt']}|{s}".encode()) % (2 ** 31)
            b = sample_one(models["base"], tok, texts4_plain[j], a.max_new, 0.7, seed)
            for k in cnt:
                o = sample_one(models[k], tok, texts4_plain[j], a.max_new, 0.7, seed)
                cnt[k] += int(diverged_within(b, o, a.max_new))
        for k in cnt:
            p[f"temp07_diverged_{k}_rate"] = cnt[k] / a.n_temp_samples
        temp_rows.append({k: cnt[k] / a.n_temp_samples for k in cnt})
    temp_summary = {k: round(sum(r[k] for r in temp_rows) / len(temp_rows), 4) for k in temp_rows[0]} | {"n": len(temp_rows)}
    print(f"[{time.time()-t0:.0f}s] déguisement température 0.7 : {temp_summary}", flush=True)

    # --- score final : signal (int8 + lora) moins bruit (lot + dtype), en glouton + sous chaque déguisement ---
    for p in stage3:
        sys_score = (int(p["sys_diverged_int8"]) + int(p["sys_diverged_lora03"])
                     - int(p["sys_diverged_noise"]) - int(p["sys_diverged_dtype"]))
        temp_score = (p["temp07_diverged_int8_rate"] + p["temp07_diverged_lora03_rate"]
                      - p["temp07_diverged_dtype_rate"])
        p["sys_score"] = sys_score
        p["temp07_score"] = round(temp_score, 4)
        p["robust_score"] = round(p["greedy_score"] + 0.5 * sys_score + 0.5 * temp_score, 4)

    stage3.sort(key=lambda p: (-p["robust_score"], -p["greedy_score"], p["margin_top1_top2"]))
    final = stage3[: a.m_final]
    for rank, p in enumerate(final, 1):
        p["rank"] = rank

    def agg(lst, keys):
        return {k: round(sum(float(p[k]) for p in lst) / len(lst), 4) for k in keys} | {"n": len(lst)}

    final_keys = ["diverged_noise", "diverged_dtype", "diverged_int8", "diverged_lora03",
                  "sys_diverged_noise", "sys_diverged_dtype", "sys_diverged_int8", "sys_diverged_lora03",
                  "temp07_diverged_dtype_rate", "temp07_diverged_int8_rate", "temp07_diverged_lora03_rate"]
    summary = {
        "greedy_by_subset": controls,
        "stage3_system_prompt": rates(flags_sys),
        "stage3_temp07": temp_summary,
        "final_all": agg(final, final_keys),
        "final_top50": agg(final[:50], final_keys),
        "final_by_category": {c: agg([p for p in final if p["category"] == c], final_keys)
                              for c in sorted(set(p["category"] for p in final))},
    }
    print(json.dumps(summary, indent=1), flush=True)
    json.dump({
        "config": vars(a), "model": MODEL, "system_prompt_used": SYSTEM_PROMPT,
        "n_pool": len(pool), "n_stage2_margin_filtered": len(stage2),
        "n_stage3_stability_tested": len(stage3), "n_final": len(final),
        "summary": summary, "prompts": final,
    }, open(a.out, "w"), indent=1, ensure_ascii=False)
    print("FIN", round(time.time() - t0), "s ->", a.out, flush=True)


if __name__ == "__main__":
    main()


