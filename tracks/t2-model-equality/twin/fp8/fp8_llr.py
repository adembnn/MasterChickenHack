#!/usr/bin/env python
"""Casser fp8 en boîte noire : rapport de vraisemblance contrasté (LLR) base vs fp8 local.

Situation réelle : on ne reçoit que du TEXTE de chaque endpoint. Si on possède le checkpoint de base, on peut
fabriquer localement une copie fp8 et demander, pour chaque réponse : « quelle hypothèse (base ou fp8) explique
mieux ce texte ? ». Statistique par réponse : s = sum_t [log p_fp8(y_t) - log p_base(y_t)] (lecture enseignée,
contexte = prompt utilisateur seul, le prompt système éventuel de l'endpoint est INCONNU du lecteur).

Plancher de bruit honnête : des endpoints « mêmes poids » servis autrement (autre taille de lot, fp16 au lieu de
bf16, autre noyau d'attention) donnent la distribution nulle de s. Jamais de seuil absolu.
Robustesse : la fp8 « distante » n'est pas forcément celle du lecteur (per-tensor + activations fp8 dynamiques,
e5m2), et les endpoints peuvent être déguisés (system prompt + persona + filigrane KGW, ou T=1.0).

100 % local (OLMo-2 1B Instruct), aucune requête API hackathon.
Usage : python fp8_llr.py --n 200 --n_numeric 80 --max_new 64 --out results_fp8_llr.json
"""
import argparse, json, math, random, re, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList

MODEL = "allenai/OLMo-2-0425-1B-Instruct"
HASH_MULT = 15485863
FP8_MAX = {"e4m3": 448.0, "e5m2": 57344.0}
FP8_DT = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
SYS = ("You are a helpful assistant for a logistics company. Always answer concisely and stay on topic. "
       "Reply as a cheerful and slightly informal customer support agent named Robin.")


def build_prompts(n, seed):
    rnd = random.Random(seed)
    caps = ["France", "Japan", "Brazil", "Canada", "Egypt", "Norway", "Peru", "Kenya", "Italy", "Chile"]
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
            out.append(f"What is the capital of {rnd.choice(caps)}? Answer in one sentence and add one fun fact.")
        elif k == 2:
            out.append(f"Explain {rnd.choice(topics)} in two sentences for a ten year old.")
        elif k == 3:
            out.append(f"Write a short Python function to {rnd.choice(tasks)}. Add a one line comment.")
        else:
            out.append(f"Rewrite this politely, keeping the meaning: {rnd.choice(phrases)}.")
        out[-1] += f" (variant {i})"
    return out


def build_numeric(n, seed):
    rnd = random.Random(seed + 999)
    out = []
    for i in range(n):
        k = i % 4
        if k == 0:
            out.append(f"Compute {rnd.randint(100, 9999)} + {rnd.randint(100, 9999)}. Reply with only the number.")
        elif k == 1:
            out.append(f"Compute {rnd.randint(12, 987)} * {rnd.randint(12, 987)}. Reply with only the number.")
        elif k == 2:
            out.append(f"Compute {rnd.randint(1000, 99999)} - {rnd.randint(100, 999)}. Reply with only the number.")
        else:
            s = "".join(rnd.choice("abcdefghij") for _ in range(rnd.randint(18, 32)))
            out.append(f"How many times does the letter 'c' appear in \"{s}\"? Reply with only the number.")
        out[-1] += f" (n{i})"
    return out


class KGW:
    def __init__(self, V, gamma, delta, key, device):
        self.V, self.delta, self.key, self.device = V, delta, key, device
        self.n_green = int(gamma * V)
        self.gen = torch.Generator(device="cpu")
        self._cache = {}

    def green(self, prev):
        k = int(prev)
        if k not in self._cache:
            self.gen.manual_seed((self.key * HASH_MULT + k) % (2 ** 63 - 1))
            self._cache[k] = torch.randperm(self.V, generator=self.gen)[: self.n_green].to(self.device)
        return self._cache[k]

    def processor(self):
        wm = self

        class _P(LogitsProcessor):
            def __call__(self, input_ids, scores):
                for b in range(input_ids.shape[0]):
                    scores[b, wm.green(input_ids[b, -1])] += wm.delta
                return scores

        return LogitsProcessorList([_P()])


def encode(tok, prompts, system=False):
    msgs = lambda p: ([{"role": "system", "content": SYS}] if system else []) + [{"role": "user", "content": p}]
    return [tok.apply_chat_template(msgs(p), tokenize=False, add_generation_prompt=True) for p in prompts]


@torch.no_grad()
def generate(model, tok, prompts, bs, max_new, system=False, wm=None, temperature=None, seed=0):
    texts = encode(tok, prompts, system)
    if temperature:
        torch.manual_seed(seed)
    outs = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        kw = dict(max_new_tokens=max_new, pad_token_id=tok.pad_token_id,
                  logits_processor=wm.processor() if wm else None)
        kw.update(dict(do_sample=True, temperature=temperature, top_p=1.0, top_k=0) if temperature else dict(do_sample=False))
        gen = model.generate(**enc, **kw)
        for row in gen[:, enc["input_ids"].shape[1]:]:
            ids = row.tolist()
            if tok.eos_token_id in ids:
                ids = ids[:ids.index(tok.eos_token_id) + 1]
            ids = [t for t in ids if t != tok.pad_token_id] or ids[:1]
            outs.append(ids)
    return outs


@torch.no_grad()
def token_logprobs(model, tok, prompts, gens, bs=16, system=False):
    """Log-probabilités par token de chaque réponse (lecture enseignée)."""
    texts = encode(tok, prompts, system)
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
        for j, r in enumerate(rows):
            ids[j, :len(r)] = torch.tensor(r)
            att[j, :len(r)] = 1
        ids, att = ids.to(model.device), att.to(model.device)
        logp = torch.log_softmax(model(input_ids=ids, attention_mask=att).logits.float(), dim=-1)
        for j, r in enumerate(rows):
            tgt = ids[j, plens[j]:len(r)]
            sl = logp[j, plens[j] - 1:len(r) - 1]
            out.append(sl.gather(-1, tgt[:, None]).squeeze(-1).tolist())
    return out


def load(dtype, attn=None):
    kw = dict(dtype=dtype, device_map={"": 0})
    if attn:
        kw["attn_implementation"] = attn
    m = AutoModelForCausalLM.from_pretrained(MODEL, **kw)
    m.eval()
    return m


@torch.no_grad()
def quant_weights_(model, fmt, per="channel"):
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and "lm_head" not in name:
            w = m.weight.data.float()
            amax = w.abs().amax(dim=1, keepdim=True) if per == "channel" else w.abs().amax()
            scale = amax.clamp(min=1e-8) / FP8_MAX[fmt]
            m.weight.data = ((w / scale).to(FP8_DT[fmt]).float() * scale).to(m.weight.dtype)
    return model


def add_fp8_act_hooks(model, fmt="e4m3"):
    """Activations fp8 dynamiques par token à l'entrée de chaque Linear (schéma W8A8 type vLLM)."""
    def hook(mod, inputs):
        x = inputs[0]
        xf = x.float()
        scale = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / FP8_MAX[fmt]
        return (((xf / scale).to(FP8_DT[fmt]).float() * scale).to(x.dtype),) + tuple(inputs[1:])
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and "lm_head" not in name:
            m.register_forward_pre_hook(hook)
    return model


# ------------------------------------------------------------------ statistiques
def auc(pos, neg):
    import bisect
    neg = sorted(neg)
    w = 0.0
    for p in pos:
        lo, hi = bisect.bisect_left(neg, p), bisect.bisect_right(neg, p)
        w += lo + 0.5 * (hi - lo)
    return w / (len(pos) * len(neg))


def endpoint_power(pos_by_ep, null_by_ep, ns, B, rnd):
    """Simule des endpoints : moyenne de s sur n réponses tirées sans remise.
    Seuil = 95e centile des moyennes d'endpoints nuls (tous les générateurs nuls de la classe)."""
    out = {}
    for n in ns:
        if n > min(len(e) for e in list(null_by_ep.values()) + list(pos_by_ep.values())):
            continue
        nulls = []
        for _ in range(B):
            ep = rnd.choice(list(null_by_ep.values()))
            nulls.append(sum(rnd.sample(ep, n)) / n)
        nulls.sort()
        thr = nulls[int(0.95 * B)]
        row = {"thr_fpr5": round(thr, 4)}
        for k, ep in pos_by_ep.items():
            hits = sum(1 for _ in range(B) if sum(rnd.sample(ep, n)) / n > thr)
            row[k] = round(hits / B, 3)
        out[f"n{n}"] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n_numeric", type=int, default=80)
    ap.add_argument("--max_new", type=int, default=64)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--delta", type=float, default=4.0)
    ap.add_argument("--B", type=int, default=3000)
    ap.add_argument("--out", default="results_fp8_llr.json")
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args()
    globals()["MODEL"] = a.model
    bf16, fp16 = torch.bfloat16, torch.float16
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    P = build_prompts(a.n, 0) + build_numeric(a.n_numeric, 0)
    is_num = [False] * a.n + [True] * a.n_numeric
    t0 = time.time()
    log = lambda *x: print(f"[{time.time()-t0:.0f}s]", *x, flush=True)
    print("GPU:", torch.cuda.get_device_name(0), "| réponses par endpoint:", len(P), flush=True)

    # --- endpoints « distants » : (classe, nom) -> textes générés
    gens, meta = {}, {}

    def run(cls, name, model, **kw):
        gens[(cls, name)] = generate(model, tok, P, kw.pop("bs", a.bs), a.max_new, **kw)
        meta[(cls, name)] = kw.get("system", False)
        log(f"généré {cls}/{name}")

    base = load(bf16)
    V = base.get_output_embeddings().weight.shape[0]
    run("plain", "null_bf16_bs16", base)
    run("plain", "null_bf16_bs5", base, bs=5)
    run("sys_wm", "null_sys_wm_k1", base, system=True, wm=KGW(V, 0.25, a.delta, 111, base.device))
    run("sys_wm", "null_sys_wm_k2_bs8", base, bs=8, system=True, wm=KGW(V, 0.25, a.delta, 222, base.device))
    run("t1", "null_t1_s1", base, temperature=1.0, seed=1)
    run("t1", "null_t1_s2_bs8", base, bs=8, temperature=1.0, seed=2)
    m = load(fp16)
    run("plain", "null_fp16", m)
    run("sys_wm", "null_fp16_sys_wm_k3", m, system=True, wm=KGW(V, 0.25, a.delta, 333, m.device))
    run("t1", "null_fp16_t1_s3", m, temperature=1.0, seed=3)
    del m
    m = load(bf16, attn="eager")
    run("plain", "null_eager", m)
    del m
    torch.cuda.empty_cache()

    m = quant_weights_(load(bf16), "e4m3")
    run("plain", "fp8_e4m3_ch", m)
    run("sys_wm", "fp8_e4m3_ch_sys_wm", m, system=True, wm=KGW(V, 0.25, a.delta, 444, m.device))
    run("t1", "fp8_e4m3_ch_t1", m, temperature=1.0, seed=4)
    del m
    m = add_fp8_act_hooks(quant_weights_(load(bf16), "e4m3", per="tensor"))
    run("plain", "fp8_w8a8_tensor", m)
    run("sys_wm", "fp8_w8a8_tensor_sys_wm", m, system=True, wm=KGW(V, 0.25, a.delta, 555, m.device))
    run("t1", "fp8_w8a8_tensor_t1", m, temperature=1.0, seed=5)
    del m
    m = quant_weights_(load(fp16), "e4m3")
    run("plain", "fp8_e4m3_ch_fp16", m)
    del m
    m = quant_weights_(load(bf16), "e5m2")
    run("plain", "fp8_e5m2_ch", m)
    run("sys_wm", "fp8_e5m2_ch_sys_wm", m, system=True, wm=KGW(V, 0.25, a.delta, 666, m.device))
    del m
    torch.cuda.empty_cache()

    # --- lecteurs locaux : base bf16 et deux copies fp8 (format de la cible inconnu)
    readers = {"base": base, "r_e4m3": quant_weights_(load(bf16), "e4m3"), "r_e5m2": quant_weights_(load(bf16), "e5m2")}
    lp = {}
    for key in gens:
        for rn, rm in readers.items():
            lp[(key, rn, False)] = token_logprobs(rm, tok, P, gens[key], a.bs)
            if meta[key]:  # borne haute : lecteur qui connaît le system prompt
                lp[(key, rn, True)] = token_logprobs(rm, tok, P, gens[key], a.bs, system=True)
        log(f"lu {key}")

    rnd = random.Random(0)
    res = {"config": vars(a), "per_endpoint": {}, "auc": {}, "power": {}}
    # s par réponse pour chaque lecteur fp8, contexte aveugle (et connu si dispo)
    S = {}
    for key in gens:
        for rf in ["r_e4m3", "r_e5m2"]:
            for known in ([False, True] if meta[key] else [False]):
                b = lp[(key, "base", known)]
                f = lp[(key, rf, known)]
                S[(key, rf, known, "sum")] = [sum(x) - sum(y) for x, y in zip(f, b)]
                S[(key, rf, known, "mean")] = [(sum(x) - sum(y)) / max(1, len(x)) for x, y in zip(f, b)]
                # variante : s seulement sur les tokens où les deux lecteurs ont un argmax différent n'est pas
                # disponible sans logits complets ; on garde sum/mean.
        cls, name = key
        s = S[(key, "r_e4m3", False, "sum")]
        res["per_endpoint"][f"{cls}/{name}"] = {
            "mean_s_e4m3_sum": round(sum(s) / len(s), 4),
            "frac_s_pos": round(sum(1 for x in s if x > 0) / len(s), 3),
            "frac_s_zero": round(sum(1 for x in s if x == 0) / len(s), 3),
            "mean_len": round(sum(len(g) for g in gens[key]) / len(P), 1)}

    for cls in ["plain", "sys_wm", "t1"]:
        keys = [k for k in gens if k[0] == cls]
        nulls = [k for k in keys if k[1].startswith("null")]
        poss = [k for k in keys if k[1].startswith("fp8")]
        for rf in ["r_e4m3", "r_e5m2"]:
            for agg in ["sum", "mean"]:
                for known in ([False, True] if cls == "sys_wm" else [False]):
                    tag = f"{cls}|{rf}|{agg}|{'known_ctx' if known else 'blind'}"
                    neg_all = [x for k in nulls for x in S[(k, rf, known, agg)]]
                    res["auc"][tag] = {}
                    for k in poss:
                        pos = S[(k, rf, known, agg)]
                        sub = {"all": auc(pos, neg_all)}
                        sub["neutral"] = auc([x for x, q in zip(pos, is_num) if not q],
                                             [x for k2 in nulls for x, q in zip(S[(k2, rf, known, agg)], is_num) if not q])
                        sub["numeric"] = auc([x for x, q in zip(pos, is_num) if q],
                                             [x for k2 in nulls for x, q in zip(S[(k2, rf, known, agg)], is_num) if q])
                        res["auc"][tag][k[1]] = {kk: round(v, 4) for kk, v in sub.items()}
                    res["power"][tag] = endpoint_power({k[1]: S[(k, rf, known, agg)] for k in poss},
                                                       {k[1]: S[(k, rf, known, agg)] for k in nulls},
                                                       [5, 10, 20, 30, 50, 100], a.B, rnd)
                    log(tag, json.dumps(res["auc"][tag]), json.dumps(res["power"][tag].get("n30")))

    # seuil universel (inconnu du déguisement) : nuls de TOUTES les classes, e4m3/sum/aveugle
    all_nulls = {f"{k[0]}/{k[1]}": S[(k, "r_e4m3", False, "sum")] for k in gens if k[1].startswith("null")}
    all_pos = {f"{k[0]}/{k[1]}": S[(k, "r_e4m3", False, "sum")] for k in gens if k[1].startswith("fp8")}
    res["power"]["pooled_all_classes|r_e4m3|sum|blind"] = endpoint_power(all_pos, all_nulls, [10, 20, 30, 50, 100], a.B, rnd)
    log("pooled", json.dumps(res["power"]["pooled_all_classes|r_e4m3|sum|blind"].get("n30")))

    json.dump(res, open(a.out, "w"), indent=1)
    json.dump({"S": {f"{k[0][0]}/{k[0][1]}|{k[1]}|{k[2]}|{k[3]}": v for k, v in S.items()}, "is_num": is_num},
              open(a.out.replace(".json", "_scores.json"), "w"))
    log("FIN ->", a.out)


if __name__ == "__main__":
    main()
