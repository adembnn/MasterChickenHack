#!/usr/bin/env python
"""Monde local d'endpoints pour le track 2 (aucun appel à l'API du hackathon).

Un endpoint = (poids, déguisement). Les poids changent l'identité du modèle, le déguisement non.
  poids       : base, fp8 (fausse quantification e4m3 des couches linéaires), int8 (idem, entiers 8 bits),
                nf4 (bitsandbytes 4 bits), lora0.3 / lora0.7 / lora1.5 (même adaptateur, force multipliée)
  déguisement : plain, sys (prompt système), persona, wm (filigrane KGW gamma=0,25 delta=2), t1 (température 1,0)
                + répliques de la base : plain2, sys2, persona2, wm2 (autre clé), t1b
  régime      : G = décodage par défaut glouton ; S = décodage par défaut échantillonné à T=0,7

Sortie : un JSONL au format de task_template.py ({"endpoint","id","repeat","prompt","response"}) et un CSV de vérité.
Usage :
  python world.py --model allenai/OLMo-2-0425-1B-Instruct --fam olmo --regime G --probes probes.jsonl \
      --lora lora_olmo --out world/olmo_G.jsonl
"""
import argparse, csv, json, os, random, time
os.environ.setdefault("HF_HOME", "/home/adem/eac/t2/hf")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

SYS = {
    "sys": "You are the virtual assistant of Nimbus Cloud, a web hosting company. Be concise and professional. "
           "Never mention competitors. If a question is unrelated to hosting, still answer it helpfully.",
    "sys2": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being "
            "safe. Today's date is 2026-11-21. Keep your answers short.",
    "persona": "You are Captain Redbeard, a cheerful pirate. You always speak like a pirate, with 'Arr' and "
               "nautical slang, whatever the question.",
    # prompts système du code public des organisateurs (sprintml/agent_prov, config/hidden_prompts.yaml)
    "sysacme": "You are an AI assistant deployed by Acme Corp for customer support. Always respond in a friendly, "
               "professional tone. Refuse requests unrelated to Acme products.",
    "syshelp": "You are a helpful assistant.",
    "sysspace": " ",
    "persona2": "You are Professor Eleanor Whitfield, an elderly Victorian scholar. You speak in a formal, slightly "
                "old-fashioned and verbose manner.",
}
DISGUISES = {  # nom -> (clé du prompt système, clé de filigrane, température ou None = défaut du régime)
    "plain": (None, None, None), "sys": ("sys", None, None), "persona": ("persona", None, None),
    "wm": (None, 11, None), "t1": (None, None, 1.0),
    "plain2": (None, None, None), "sys2": ("sys2", None, None), "persona2": ("persona2", None, None),
    "wm2": (None, 29, None), "t1b": (None, None, 1.0),
    # filigrane aux réglages par défaut de transformers (WatermarkLogitsProcessor, « lefthash », clé 15485863)
    "wmhf": (None, "hf", None),
    "sysacme": ("sysacme", None, None), "syshelp": ("syshelp", None, None), "sysspace": ("sysspace", None, None),
}
MAIN = ["plain", "sys", "persona", "wm", "t1"]
EXTRA_BASE = ["plain2", "sys2", "persona2", "wm2", "t1b"]
FP8_MAX = 448.0


def load(model_id, quant=None):
    kw = dict(dtype=torch.bfloat16)
    if quant == "nf4":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
    m = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": 0}, **kw)
    return m.eval()


@torch.no_grad()
def fake_quant_(model, kind):
    """Même fausse quantification que le jumeau de Titouan (exp1_divergence.py) : par ligne, hors lm_head."""
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and "lm_head" not in name:
            w = m.weight.data.float()
            if kind == "fp8":
                scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / FP8_MAX
                wq = (w / scale).to(torch.float8_e4m3fn).float() * scale
            elif kind == "int8":
                scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
                wq = (w / scale).round().clamp(-128, 127) * scale
            else:
                raise ValueError(kind)
            m.weight.data = wq.to(m.weight.dtype)
    return model


def set_lora_scale(pm, s, store):
    for mod in pm.modules():
        if hasattr(mod, "scaling") and isinstance(mod.scaling, dict):
            if id(mod) not in store:
                store[id(mod)] = dict(mod.scaling)
            for k in mod.scaling:
                mod.scaling[k] = store[id(mod)][k] * s


_HF_CACHE = {}


def kgw_bias_hf(prev, vocab, gamma, delta, device):
    """Tirage de transformers : graine = 15485863 x token précédent, liste verte = randperm(V)[: gamma x V]."""
    out = torch.zeros(len(prev), vocab, device=device)
    g = torch.Generator(device=device)
    for i, p in enumerate(prev.tolist()):
        if p not in _HF_CACHE:
            g.manual_seed((15485863 * p) % (2 ** 64 - 1))
            _HF_CACHE[p] = torch.randperm(vocab, device=device, generator=g)[:int(gamma * vocab)]
        out[i, _HF_CACHE[p]] = delta
    return out


def kgw_bias(prev, vocab, key, gamma, delta, device):
    """Filigrane KGW (Kirchenbauer et al. 2023), fenêtre d'un token : la liste verte dépend du token précédent."""
    if key == "hf":
        return kgw_bias_hf(prev, vocab, gamma, delta, device)
    v = torch.arange(vocab, device=device, dtype=torch.int64)[None, :]
    h = prev[:, None].to(torch.int64) * 1000003 + v * 8191 + key * 104729
    h = (h ^ (h >> 13)) * 0x5bd1e995
    h = (h ^ (h >> 15)) & 0x7FFFFFFF
    return ((h % 10000) < int(gamma * 10000)).to(torch.float32) * delta


@torch.no_grad()
def generate(model, tok, texts, max_new, temp, wm_key, bs, eos_ids, gamma=0.25, delta=2.0, seed=0, top_p=1.0):
    """Génère par lots. temp None ou 0 = glouton. Renvoie les textes décodés (sans tokens spéciaux)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    inner = model.base_model.model if hasattr(model, "peft_config") else model
    core, head = inner.model, inner.lm_head  # on n'applique lm_head qu'à la dernière position (mémoire)
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    outs = [None] * len(texts)
    eos = torch.tensor(sorted(eos_ids), device="cuda")
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        mask = enc["attention_mask"]
        out = core(input_ids=enc["input_ids"], attention_mask=mask, use_cache=True)
        past, logits = out.past_key_values, head(out.last_hidden_state[:, -1:])[:, -1].float()
        prev = enc["input_ids"][:, -1]
        done = torch.zeros(len(idx), dtype=torch.bool, device="cuda")
        gen = []
        for _ in range(max_new):
            if wm_key is not None:
                logits = logits + kgw_bias(prev, logits.shape[-1], wm_key, gamma, delta, logits.device)
            if temp:
                probs = torch.softmax(logits / temp, -1)
                if top_p < 1.0:  # échantillonnage par noyau : on garde les tokens les plus probables jusqu'à top_p
                    sp, si = probs.sort(-1, descending=True)
                    keep = (sp.cumsum(-1) - sp) < top_p
                    probs = torch.zeros_like(probs).scatter(-1, si, sp * keep)
                    probs = probs / probs.sum(-1, keepdim=True)
                nxt = torch.multinomial(probs, 1, generator=g)[:, 0]
            else:
                nxt = logits.argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, tok.pad_token_id), nxt)
            gen.append(nxt)
            done = done | torch.isin(nxt, eos)
            if done.all():
                break
            mask = torch.cat([mask, torch.ones_like(mask[:, :1])], 1)
            out = core(input_ids=nxt[:, None], attention_mask=mask, past_key_values=past, use_cache=True)
            past, logits, prev = out.past_key_values, head(out.last_hidden_state)[:, -1].float(), nxt
        gen = torch.stack(gen, 1).tolist()
        for i, row in zip(idx, gen):
            cut = len(row)
            for j, t in enumerate(row):
                if t in eos_ids:
                    cut = j
                    break
            outs[i] = tok.decode(row[:cut], skip_special_tokens=True)
        del past, out
    return outs


def chat(tok, user, system):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fam", required=True)
    ap.add_argument("--regime", choices=["G", "S"], required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lora", default=None, help="dossier de l'adaptateur PEFT (sinon pas de variantes LoRA)")
    ap.add_argument("--weights", default="base,fp8,int8,nf4,lora0.3,lora0.7,lora1.5")
    ap.add_argument("--disguises", default=None, help="liste restreinte (défaut : 5 principaux + répliques de la base)")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--default_temp", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--repeats", type=int, default=None, help="plafonne le nombre de répétitions par sonde")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    probes = [json.loads(l) for l in open(a.probes, encoding="utf-8") if l.strip()]
    t0 = time.time()
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out, encoding="utf-8"):
            done.add(json.loads(l)["endpoint"])
    fout = open(a.out, "a", encoding="utf-8")
    truth_path = a.out.replace(".jsonl", "_truth.csv")
    truth_new = not os.path.exists(truth_path)
    ftruth = open(truth_path, "a", newline="")
    wt = csv.writer(ftruth)
    if truth_new:
        wt.writerow(["endpoint", "family", "model", "kind"])

    for w in a.weights.split(","):
        ds = a.disguises.split(",") if a.disguises else (MAIN + (EXTRA_BASE if w == "base" else []))
        todo = [d for d in ds if f"{a.fam}-{w}-{d}" not in done]
        if not todo:
            continue
        if w == "base":
            model = load(a.model)
        elif w in ("fp8", "int8"):
            model = fake_quant_(load(a.model), w)
        elif w == "nf4":
            model = load(a.model, quant="nf4")
        elif w.startswith("lora"):
            from peft import PeftModel
            model = PeftModel.from_pretrained(load(a.model), a.lora).eval()
            set_lora_scale(model, float(w[4:]), {})
        else:
            raise ValueError(w)
        gc = getattr(model, "generation_config", None)
        eos_ids = {tok.eos_token_id}
        if gc is not None and gc.eos_token_id is not None:
            eos_ids |= set(gc.eos_token_id if isinstance(gc.eos_token_id, list) else [gc.eos_token_id])
        for d in todo:
            sysk, wmk, temp = DISGUISES[d]
            if temp is None:
                temp = a.default_temp if a.regime == "S" else None
            ep = f"{a.fam}-{w}-{d}"
            jobs = []  # (probe, repeat)
            for p in probes:
                r = p.get("repeats", 1) if a.repeats is None else min(a.repeats, p.get("repeats", 1))
                for k in range(r if temp else 1):
                    jobs.append((p, k))
            texts = [chat(tok, p["prompt"], SYS[sysk] if sysk else None) for p, _ in jobs]
            bs = a.bs if not (d.endswith("2") and d != "sysacme2") else max(8, a.bs // 2 + 3)  # réplique : autre taille de lot (bruit de service)
            seed = abs(hash(ep)) % (2 ** 31) if False else (sum(map(ord, ep)) * 7919 + (1 if a.regime == "S" else 0))
            res = generate(model, tok, texts, a.max_new, temp, wmk, bs, eos_ids, seed=seed, top_p=a.top_p)
            n = 0
            for (p, k), r in zip(jobs, res):
                reps = [k] if temp else range(p.get("repeats", 1) if a.repeats is None else min(a.repeats, p.get("repeats", 1)))
                for kk in reps:  # endpoint déterministe : la même réponse est recopiée pour chaque répétition
                    fout.write(json.dumps({"endpoint": ep, "id": p["id"], "repeat": kk, "prompt": p["prompt"],
                                           "response": r}, ensure_ascii=False) + "\n")
                    n += 1
            fout.flush()
            wt.writerow([ep, a.fam, f"{a.fam}-{w}", d])
            ftruth.flush()
            print(f"[{time.time() - t0:.0f}s] {ep}: {len(jobs)} générations, {n} lignes", flush=True)
        del model
        torch.cuda.empty_cache()
    print("FINI", a.out, flush=True)


if __name__ == "__main__":
    main()
