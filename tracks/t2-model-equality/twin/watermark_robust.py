#!/usr/bin/env python
"""Jumeau local T2 : un filigrane de génération (KGW) casse-t-il la comparaison de texte glouton, et par quoi la remplacer ?

Tout est local (OLMo-2 1B Instruct sur JURECA), aucune requête vers l'API du hackathon.

1. Filigrane KGW (Kirchenbauer et al. 2023) : à chaque pas, le token précédent sert de graine à une
   permutation du vocabulaire ; les gamma·V premiers tokens (« liste verte ») reçoivent un bonus delta sur
   les logits. Les poids du modèle ne changent pas, seule la règle de décodage change.
2. On mesure comment le filigrane détruit le signal « texte glouton identique » entre deux instances du
   MÊME modèle (sans filigrane vs avec ; deux clés différentes), pour plusieurs forces delta.
3. On teste deux statistiques robustes :
   a. z-score KGW : détecte la présence du filigrane (et retrouve la clé si on connaît le schéma de hachage).
   b. log-vraisemblance croisée : chaque modèle « lit » (forward enseigné, sans biais) le texte généré par
      l'autre ; on compare la NLL moyenne par token. Comme le filigrane ne touche pas aux poids, un vrai
      jumeau obtient exactement la même NLL sur le texte filigrané que l'instance qui l'a produit.
   Contrôle négatif : un modèle réellement différent (NF4, fp8_e5m2, LoRA) doit donner une NLL croisée nettement
   plus haute, alors que la comparaison exacte le confond avec le cas « même modèle + filigrane ».

Usage : python watermark_robust.py --n 60 --max_new 64 --bs 16 --deltas 0,1,2,4,8 --out results_wm.json
"""
import argparse, json, math, random, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList, BitsAndBytesConfig

MODEL = "allenai/OLMo-2-0425-1B-Instruct"
HASH_MULT = 15485863  # grand premier, mélange clé et token précédent


def build_prompts(n, seed):
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


# ---------------------------------------------------------------- filigrane KGW
class KGW:
    """Liste verte dépendant du token précédent (fenêtre de contexte 1), bonus delta sur ses logits."""

    def __init__(self, vocab_size, gamma, delta, key, device):
        self.V, self.gamma, self.delta, self.key = vocab_size, gamma, delta, key
        self.n_green = int(gamma * vocab_size)
        self.device = device
        self.gen = torch.Generator(device="cpu")
        self._cache = {}

    def green_ids(self, prev_tok):
        k = int(prev_tok)
        if k not in self._cache:
            self.gen.manual_seed((self.key * HASH_MULT + k) % (2 ** 63 - 1))
            self._cache[k] = torch.randperm(self.V, generator=self.gen)[: self.n_green].to(self.device)
        return self._cache[k]

    def processor(self):
        wm = self

        class _P(LogitsProcessor):
            def __call__(self, input_ids, scores):
                for b in range(input_ids.shape[0]):
                    scores[b, wm.green_ids(input_ids[b, -1])] += wm.delta
                return scores

        return LogitsProcessorList([_P()])

    def z_score(self, ids):
        """Test de Kirchenbauer : nombre de tokens verts vs attendu gamma·T sous H0 (pas de filigrane)."""
        T = len(ids) - 1
        if T <= 0:
            return 0.0
        green = 0
        for i in range(1, len(ids)):
            if (self.green_ids(ids[i - 1]) == ids[i]).any():
                green += 1
        return (green - self.gamma * T) / math.sqrt(T * self.gamma * (1 - self.gamma))


# ---------------------------------------------------------------- génération et lecture
def encode(tok, prompts):
    return [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts]


@torch.no_grad()
def generate(model, tok, prompts, bs, max_new, wm=None):
    outs = []
    texts = encode(tok, prompts)
    lp = wm.processor() if wm is not None else None
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, do_sample=False, max_new_tokens=max_new, pad_token_id=tok.pad_token_id, logits_processor=lp)
        for row in gen[:, enc["input_ids"].shape[1]:]:
            ids = row.tolist()
            if tok.eos_token_id in ids:
                ids = ids[:ids.index(tok.eos_token_id) + 1]
            outs.append(ids)
    return outs


@torch.no_grad()
def nll_per_token(model, tok, prompts, gens, bs):
    """NLL moyenne par token généré (forward enseigné, sans aucun biais) : le modèle « lit » le texte."""
    texts = encode(tok, prompts)
    out = []
    for i in range(0, len(texts), bs):
        rows, plens = [], []
        for t, g in zip(texts[i:i + bs], gens[i:i + bs]):
            p = tok(t, add_special_tokens=False)["input_ids"]
            rows.append(p + g)
            plens.append(len(p))
        L = max(len(r) for r in rows)
        ids = torch.full((len(rows), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(rows), L), dtype=torch.long)
        for j, r in enumerate(rows):  # padding à droite pour la lecture
            ids[j, :len(r)] = torch.tensor(r)
            att[j, :len(r)] = 1
        ids, att = ids.to(model.device), att.to(model.device)
        logp = torch.log_softmax(model(input_ids=ids, attention_mask=att).logits.float(), dim=-1)
        for j, r in enumerate(rows):
            n_gen = len(r) - plens[j]
            if n_gen == 0:
                out.append((float("nan"), float("nan")))
                continue
            tgt = ids[j, plens[j]:len(r)]
            sl = logp[j, plens[j] - 1:len(r) - 1]
            lp = sl.gather(-1, tgt[:, None]).squeeze(-1)
            # part des tokens qui ne sont PAS l'argmax non biaisé du lecteur : ~0 pour un texte glouton de ce modèle
            mismatch = (sl.argmax(-1) != tgt).float().mean().item()
            out.append((-lp.mean().item(), mismatch))
    return out


def first_div(a, b):
    m = min(len(a), len(b))
    for i in range(m):
        if a[i] != b[i]:
            return i
    return m if len(a) != len(b) else None


def compare_exact(base, other):
    same, divs = 0, []
    for a, b in zip(base, other):
        d = first_div(a, b)
        if d is None:
            same += 1
        else:
            divs.append(d)
    return {"identical_frac": same / len(base), "mean_first_div": (sum(divs) / len(divs)) if divs else None,
            "median_first_div": sorted(divs)[len(divs) // 2] if divs else None}


def short(d):
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items() if k != "per_prompt"}


def mean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def paired_t(gaps):
    g = [x for x in gaps if x == x]
    if len(g) < 2:
        return float("nan")
    m = sum(g) / len(g)
    sd = math.sqrt(sum((x - m) ** 2 for x in g) / (len(g) - 1))
    return m / (sd / math.sqrt(len(g))) if sd > 0 else (0.0 if m == 0 else float("inf"))


def cross_nll(mA, mB, tok, prompts, gA, gB, bs):
    """Statistique robuste : chaque modèle lit le texte de l'autre.
    gap = NLL_B(texte_A) - NLL_A(texte_A), et symétrique. Jumeau parfait => gap = 0 quel que soit le filigrane.
    Un modèle différent est plus surpris par le texte de l'autre que son auteur : gap > 0."""
    aa = nll_per_token(mA, tok, prompts, gA, bs)
    ba = nll_per_token(mB, tok, prompts, gA, bs)
    bb = nll_per_token(mB, tok, prompts, gB, bs)
    ab = nll_per_token(mA, tok, prompts, gB, bs)
    nll = lambda xs: [x[0] for x in xs]
    mm = lambda xs: [x[1] for x in xs]
    gaps = [(y - x) for x, y in zip(nll(aa), nll(ba))] + [(y - x) for x, y in zip(nll(bb), nll(ab))]
    valid = [g for g in gaps if g == g]
    return {"nll_A_on_A": mean(nll(aa)), "nll_B_on_A": mean(nll(ba)), "nll_B_on_B": mean(nll(bb)), "nll_A_on_B": mean(nll(ab)),
            "mean_gap": mean(gaps), "mean_abs_gap": mean([abs(g) for g in gaps]), "max_abs_gap": max(abs(g) for g in valid),
            "paired_t_gap": paired_t(gaps), "frac_gap_gt_0.05": sum(1 for g in valid if abs(g) > 0.05) / len(valid),
            # détecteur de filigrane sans clé : part des tokens hors argmax quand l'auteur relit son propre texte
            "self_argmax_mismatch_A": mean(mm(aa)), "self_argmax_mismatch_B": mean(mm(bb)),
            "cross_argmax_mismatch_B_on_A": mean(mm(ba)), "cross_argmax_mismatch_A_on_B": mean(mm(ab)),
            "per_prompt": {"nll_A_on_A": nll(aa), "nll_B_on_A": nll(ba), "nll_B_on_B": nll(bb), "nll_A_on_B": nll(ab)}}


# ---------------------------------------------------------------- variantes « vraiment différentes »
FP8_MAX = {"e4m3": 448.0, "e5m2": 57344.0}
FP8_DT = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}


def load(dtype, quant=None):
    kw = dict(dtype=dtype)
    if quant == "nf4":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
    m = AutoModelForCausalLM.from_pretrained(MODEL, device_map={"": 0}, **kw)
    m.eval()
    return m


@torch.no_grad()
def fake_quant_(model, kind):
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and "lm_head" not in name:
            w = m.weight.data.float()
            if kind.startswith("fp8"):
                fmt = kind.split("_")[1]
                scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / FP8_MAX[fmt]
                wq = (w / scale).to(FP8_DT[fmt]).float() * scale
            elif kind == "int8":
                scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
                wq = (w / scale).round().clamp(-128, 127) * scale
            m.weight.data = wq.to(m.weight.dtype)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--max_new", type=int, default=64)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--gamma", type=float, default=0.25)
    ap.add_argument("--deltas", default="0,1,2,4,8")
    ap.add_argument("--keys", default="111,222")
    ap.add_argument("--out", default="results_wm.json")
    a = ap.parse_args()
    dtype = getattr(torch, a.dtype)
    deltas = [float(x) for x in a.deltas.split(",")]
    k1, k2 = [int(x) for x in a.keys.split(",")]

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompts = build_prompts(a.n, seed=0)
    t0 = time.time()
    res = {"config": vars(a), "exact": {}, "zscore": {}, "cross_nll": {}}
    log = lambda *x: print(f"[{time.time()-t0:.0f}s]", *x, flush=True)

    print("GPU:", torch.cuda.get_device_name(0), "| prompts:", a.n, "| max_new:", a.max_new, flush=True)
    base_m = load(dtype)
    V = base_m.get_output_embeddings().weight.shape[0]
    gens = {"plain": generate(base_m, tok, prompts, a.bs, a.max_new)}
    log("base sans filigrane prête, V =", V)

    # 1. Le filigrane casse-t-il la comparaison exacte ? (même modèle, mêmes poids)
    for d in deltas:
        for name, key in (("k1", k1), ("k2", k2)):
            wm = KGW(V, a.gamma, d, key, base_m.device)
            gens[f"wm_d{d}_{name}"] = generate(base_m, tok, prompts, a.bs, a.max_new, wm)
        res["exact"][f"plain_vs_wm_d{d}"] = compare_exact(gens["plain"], gens[f"wm_d{d}_k1"])
        res["exact"][f"wm_d{d}_k1_vs_k2"] = compare_exact(gens[f"wm_d{d}_k1"], gens[f"wm_d{d}_k2"])
        log(f"delta={d}: plain vs wm {res['exact'][f'plain_vs_wm_d{d}']} | k1 vs k2 {res['exact'][f'wm_d{d}_k1_vs_k2']}")

    # 2. Détection du filigrane (z-score) : bonne clé, mauvaise clé, texte sans filigrane
    for d in deltas:
        if d == 0:
            continue
        row = {}
        for cand_name, cand_key in (("true_key", k1), ("wrong_key", k2), ("random_key", 98765)):
            wm = KGW(V, a.gamma, d, cand_key, base_m.device)
            zs = [wm.z_score(g) for g in gens[f"wm_d{d}_k1"]]
            row[cand_name] = {"mean_z": mean(zs), "frac_z_gt_4": sum(z > 4 for z in zs) / len(zs)}
        wm = KGW(V, a.gamma, d, k1, base_m.device)
        zs = [wm.z_score(g) for g in gens["plain"]]
        row["plain_text_true_key"] = {"mean_z": mean(zs), "frac_z_gt_4": sum(z > 4 for z in zs) / len(zs)}
        res["zscore"][f"d{d}"] = row
        log(f"z-score delta={d}: {row}")

    # 3. Statistique robuste : NLL croisée. Même modèle (jumeau) avec filigranes différents...
    for d in deltas:
        res["cross_nll"][f"same_model_plain_vs_wm_d{d}"] = cross_nll(base_m, base_m, tok, prompts, gens["plain"], gens[f"wm_d{d}_k1"], a.bs)
        res["cross_nll"][f"same_model_wm_d{d}_k1_vs_k2"] = cross_nll(base_m, base_m, tok, prompts, gens[f"wm_d{d}_k1"], gens[f"wm_d{d}_k2"], a.bs)
        log(f"NLL croisée même modèle delta={d}: {short(res['cross_nll'][f'same_model_wm_d{d}_k1_vs_k2'])}")

    # ... contre des modèles réellement différents (quantification), avec et sans filigrane côté « autre ».
    d_ref = max(deltas)
    for kind in ["int8", "fp8_e4m3", "fp8_e5m2", "nf4"]:
        try:
            m = load(dtype, quant="nf4") if kind == "nf4" else fake_quant_(load(dtype), kind)
            g_other = generate(m, tok, prompts, a.bs, a.max_new)
            res["exact"][f"plain_vs_{kind}"] = compare_exact(gens["plain"], g_other)
            res["cross_nll"][f"diff_{kind}_plain"] = cross_nll(base_m, m, tok, prompts, gens["plain"], g_other, a.bs)
            wm = KGW(V, a.gamma, d_ref, k2, m.device)
            g_other_wm = generate(m, tok, prompts, a.bs, a.max_new, wm)
            res["exact"][f"wm_d{d_ref}_k1_vs_{kind}_wm_k2"] = compare_exact(gens[f"wm_d{d_ref}_k1"], g_other_wm)
            res["cross_nll"][f"diff_{kind}_wm_d{d_ref}"] = cross_nll(base_m, m, tok, prompts, gens[f"wm_d{d_ref}_k1"], g_other_wm, a.bs)
            log(f"{kind}: exact {res['exact'][f'plain_vs_{kind}']} | NLL croisée plain {short(res['cross_nll'][f'diff_{kind}_plain'])} | NLL croisée wm {short(res['cross_nll'][f'diff_{kind}_wm_d{d_ref}'])}")
            del m
            torch.cuda.empty_cache()
        except Exception as e:
            res["cross_nll"][f"diff_{kind}"] = {"error": repr(e)[:300]}
            log(kind, "erreur:", e)

    json.dump(res, open(a.out, "w"), indent=1)
    json.dump({k: v for k, v in gens.items()}, open(a.out.replace(".json", "_outputs.json"), "w"))
    log("FIN ->", a.out)


if __name__ == "__main__":
    main()
