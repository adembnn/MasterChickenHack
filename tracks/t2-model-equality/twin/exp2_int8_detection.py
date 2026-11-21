#!/usr/bin/env python
"""Expérience 2 du jumeau local T2 : rendre int8 détectable face au bruit de calcul GPU.

Aucune requête sur l'API du hackathon : tout est local (OLMo-2 1B Instruct, GPU JURECA).
Par rapport à exp1 (int8 par canal : 50 % identique vs bruit 66-70 %), on mesure :
  (1) la granularité de la fausse quantification int8 des poids : par tenseur, par canal, par groupe (128/64/32) ;
  (2) W8A8 : poids par canal + activations int8 dynamiques par token (schéma réel type SmoothQuant/LLM.int8) ;
  (3) l'effet de max_new_tokens (96 -> 192 -> 384) sur l'écart int8-vs-bruit ;
  (4) un plancher de bruit plus robuste : plusieurs tailles de lot + sanity-check de déterminisme total.
Chaque comparaison renvoie aussi un intervalle de confiance binomial (Wilson 95 %) et la part de tokens
concordants jusqu'à la première divergence (métrique plus fine que « réponse identique »).
Usage : python exp2_int8_detection.py --n 300 --max_new 96 --bs 16 --out results_exp2.json
"""
import argparse, json, math, os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp1_divergence import MODEL, build_prompts, generate, first_div, load  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))


def compare2(base, other):
    same, divs, matched, total = 0, [], 0, 0
    for a, b in zip(base, other):
        d = first_div(a, b)
        total += len(a)
        if d is None:
            same += 1
            matched += len(a)
        else:
            divs.append(d)
            matched += d
    n = len(base)
    return {"identical_frac": same / n, "identical_ci95": wilson(same, n), "n": n,
            "token_match_frac": matched / total if total else None,
            "mean_first_div": (sum(divs) / len(divs)) if divs else None,
            "median_first_div": sorted(divs)[len(divs) // 2] if divs else None}


@torch.no_grad()
def fake_quant_int8_(model, granularity, group_size=None):
    """granularity : 'tensor' (une échelle par matrice), 'channel' (par ligne de sortie, = exp1), 'group' (par bloc de group_size colonnes)."""
    for name, m in model.named_modules():
        if not (isinstance(m, torch.nn.Linear) and "lm_head" not in name):
            continue
        w = m.weight.data.float()
        if granularity == "tensor":
            scale = w.abs().amax().clamp(min=1e-8) / 127.0
            wq = (w / scale).round().clamp(-128, 127) * scale
        elif granularity == "channel":
            scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
            wq = (w / scale).round().clamp(-128, 127) * scale
        elif granularity == "group":
            out_f, in_f = w.shape
            pad = (-in_f) % group_size
            wp = torch.nn.functional.pad(w, (0, pad)) if pad else w
            wg = wp.view(out_f, -1, group_size)
            scale = wg.abs().amax(dim=2, keepdim=True).clamp(min=1e-8) / 127.0
            wq = ((wg / scale).round().clamp(-128, 127) * scale).view(out_f, -1)[:, :in_f]
        else:
            raise ValueError(granularity)
        m.weight.data = wq.to(m.weight.dtype)
    return model


def add_act_quant_hooks(model):
    """Activations int8 symétriques dynamiques par token (échelle sur la dernière dimension) à l'entrée de chaque Linear hors lm_head."""
    def hook(mod, inputs):
        x = inputs[0]
        xf = x.float()
        scale = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
        xq = (xf / scale).round().clamp(-128, 127) * scale
        return (xq.to(x.dtype),) + tuple(inputs[1:])
    return [m.register_forward_pre_hook(hook) for name, m in model.named_modules()
            if isinstance(m, torch.nn.Linear) and "lm_head" not in name]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--max_new", type=int, default=96)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--noise_bs", default="1,8,32,64")
    ap.add_argument("--long_max_new", default="192,384")
    ap.add_argument("--out", default="results_exp2.json")
    a = ap.parse_args()
    dtype = getattr(torch, a.dtype)
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompts = build_prompts(a.n, seed=0)
    t0 = time.time()
    res, outputs = {}, {}

    def log(k, v):
        res[k] = v
        print(f"[{time.time()-t0:.0f}s] {k}: {v}", flush=True)

    print("GPU:", torch.cuda.get_device_name(0), "| dtype:", a.dtype, "| prompts:", a.n, "| max_new:", a.max_new, flush=True)
    base_m = load(dtype)
    base = generate(base_m, tok, prompts, a.bs, a.max_new)
    outputs["base"] = base
    print(f"[{time.time()-t0:.0f}s] base prête", flush=True)

    # (4a) plancher de bruit : même modèle, même bs, deuxième passage -> doit être 100 % identique
    log("noise_same_bs_repeat", compare2(base, generate(base_m, tok, prompts, a.bs, a.max_new)))
    # (4b) plancher de bruit : autres tailles de lot
    for bs2 in [int(x) for x in a.noise_bs.split(",") if x]:
        if bs2 != a.bs:
            log(f"noise_bs{bs2}", compare2(base, generate(base_m, tok, prompts, bs2, a.max_new)))
    # (4c) déterminisme total : CUBLAS_WORKSPACE_CONFIG doit être posé dans l'environnement avant le lancement
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        d1 = generate(base_m, tok, prompts, a.bs, a.max_new)
        d2 = generate(base_m, tok, prompts, a.bs, a.max_new)
        d_bs1 = generate(base_m, tok, prompts, 1, a.max_new) if 1 in [int(x) for x in a.noise_bs.split(",")] else None
        log("determ_same_bs_repeat", compare2(d1, d2))
        log("determ_vs_nondeterm_base", compare2(base, d1))
        if d_bs1 is not None:
            log("determ_bs1_vs_determ_bs16", compare2(d1, d_bs1))
    except Exception as e:
        log("determ_error", repr(e)[:300])
    finally:
        torch.use_deterministic_algorithms(False)

    # (1) granularités int8 poids seuls
    for tag, gran, gs in [("tensor", "tensor", None), ("channel", "channel", None),
                          ("group128", "group", 128), ("group64", "group", 64), ("group32", "group", 32)]:
        m = fake_quant_int8_(load(dtype), gran, gs)
        o = generate(m, tok, prompts, a.bs, a.max_new)
        outputs[f"int8_w_{tag}"] = o
        log(f"int8_w_{tag}", compare2(base, o))
        del m
        torch.cuda.empty_cache()

    # (2) W8A8 : poids par canal + activations dynamiques par token
    m = fake_quant_int8_(load(dtype), "channel")
    handles = add_act_quant_hooks(m)
    o = generate(m, tok, prompts, a.bs, a.max_new)
    outputs["int8_w8a8"] = o
    log("int8_w8a8", compare2(base, o))
    for h in handles:
        h.remove()
    del m
    torch.cuda.empty_cache()
    # (2b) activations seules (poids intacts) pour isoler la contribution des activations
    handles = add_act_quant_hooks(base_m)
    log("int8_a8_only", compare2(base, generate(base_m, tok, prompts, a.bs, a.max_new)))
    for h in handles:
        h.remove()

    # (3) générations plus longues : int8 par canal vs bruit bs=32 (le moins coûteux)
    for mn in [int(x) for x in a.long_max_new.split(",") if x]:
        base_long = generate(base_m, tok, prompts, a.bs, mn)
        log(f"noise_bs32_maxnew{mn}", compare2(base_long, generate(base_m, tok, prompts, 32, mn)))
        m = fake_quant_int8_(load(dtype), "channel")
        log(f"int8_w_channel_maxnew{mn}", compare2(base_long, generate(m, tok, prompts, a.bs, mn)))
        del m
        torch.cuda.empty_cache()

    json.dump({"config": vars(a), "results": res}, open(a.out, "w"), indent=1)
    json.dump(outputs, open(a.out.replace(".json", "_outputs.json"), "w"))
    print("FIN", round(time.time() - t0), "s ->", a.out, flush=True)


if __name__ == "__main__":
    main()
