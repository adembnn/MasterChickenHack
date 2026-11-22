"""Générateur de guichets jumeaux : un 'monde' = une famille, une liste de guichets (variante de poids + déguisement).

python gen.py --family smol360 --split train --seed 1 --out worlds/
Écrit worlds/<family>_<split>_s<seed>.jsonl : une ligne par guichet {gid, family, variant, disguise, responses[100]}.
Variantes de poids : base, fp8 (par tenseur, poids ; option activations), int8 (par canal), nf4 (blocs 64),
LoRA <nom>@<force>. Échantillonnage T par guichet, filigrane KGW vectorisé (liste verte = hachage(clé, token précédent, token)).
"""
import argparse, json, math, os, random, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
import probes_v4
from disguises import disguise_bank
from lora_train import FAMILIES, SPECS

NF4 = torch.tensor([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
                    -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
                    0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
                    0.7229568362236023, 1.0])


def quantize_(model, kind):
    if kind == "fp8all":   # toutes les matrices, y compris lm_head / embeddings liés
        for name, p in model.named_parameters():
            if p.dim() == 2:
                w = p.data.float(); s = w.abs().max().clamp(min=1e-12) / 448.0
                p.data = ((w / s).to(torch.float8_e4m3fn).float() * s).to(p.dtype)
        return model
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and "lm_head" not in name:
            w = m.weight.data.float()
            if kind == "fp8ch":   # fp8 par canal de sortie
                s = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 448.0
                q = (w / s).to(torch.float8_e4m3fn).float() * s
            elif kind == "fp8e5":  # fp8 e5m2 par tenseur
                s = w.abs().max().clamp(min=1e-12) / 57344.0
                q = (w / s).to(torch.float8_e5m2).float() * s
            elif kind.startswith("fp8"):
                s = w.abs().max().clamp(min=1e-12) / 448.0
                q = (w / s).to(torch.float8_e4m3fn).float() * s
            elif kind == "int8":
                s = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 127.0
                q = torch.round(w / s).clamp(-127, 127) * s
            elif kind == "nf4":
                flat = w.reshape(-1, 64)
                am = flat.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
                lv = NF4.to(w.device)
                idx = ((flat / am).unsqueeze(-1) - lv).abs().argmin(-1)
                q = (lv[idx] * am).reshape(w.shape)
            m.weight.data = q.to(m.weight.dtype)
            if kind == "fp8wa":
                def hook(mod, inp):
                    x = inp[0]; s = x.abs().amax().float().clamp(min=1e-12) / 448.0
                    return ((x.float() / s).to(torch.float8_e4m3fn).float() * s).to(x.dtype),
                m.register_forward_pre_hook(hook)
    return model


def load_variant(family, variant, lora_dir):
    mid = FAMILIES[family]
    model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).cuda().eval()
    if variant == "base":
        return model
    if variant in ("fp8", "fp8wa", "fp8ch", "fp8e5", "fp8all", "int8", "nf4"):
        return quantize_(model, variant)
    if variant.startswith("lora:"):
        name, strength = variant[5:].split("@")
        from peft import PeftModel
        pm = PeftModel.from_pretrained(model, os.path.join(lora_dir, family, name))
        for mod in pm.modules():
            if hasattr(mod, "scaling") and isinstance(mod.scaling, dict):
                for k in mod.scaling:
                    mod.scaling[k] *= float(strength)
        return pm.merge_and_unload().eval()
    raise ValueError(variant)


class RowProc(LogitsProcessor):
    """Par ligne : filigrane KGW (bonus delta sur la liste verte) puis division par la température."""
    def __init__(self, T, gamma, delta, key):
        self.T, self.gamma, self.delta, self.key = T, gamma, delta, key  # tenseurs (B,)
    def __call__(self, input_ids, scores):
        B, V = scores.shape
        prev = input_ids[:, -1].long()
        v = torch.arange(V, device=scores.device, dtype=torch.long)
        h = (prev[:, None] * 2654435761 + v[None, :] * 40503 + self.key[:, None] * 97) & 0xFFFFFFFF
        h = h ^ (h >> 13); h = (h * 0x5BD1E995) & 0xFFFFFFFF; h = h ^ (h >> 15)
        green = (h % 10000).float() < (self.gamma[:, None] * 10000)
        scores = scores.float() + green.float() * self.delta[:, None]
        return scores / self.T[:, None]


@torch.no_grad()
def run_guichets(model, tok, guichets, prompts, max_new=256, tok_budget=None):
    """Génère toutes les réponses des guichets (même variante de poids). Retourne {gid: [100 textes]}."""
    jobs = []
    for g in guichets:
        dz = g["disguise"]
        for i, p in enumerate(prompts):
            msgs = ([{"role": "system", "content": dz["system"]}] if dz["system"] else []) + [{"role": "user", "content": p}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            gam, dl = dz["wm"] if dz["wm"] else (0.0, 0.0)
            jobs.append((g["gid"], i, text, dz["T"], gam, dl, dz["wm_key"]))
    enc = tok([j[2] for j in jobs], add_special_tokens=False)["input_ids"]
    order = sorted(range(len(jobs)), key=lambda k: len(enc[k]))
    out = {g["gid"]: [None] * len(prompts) for g in guichets}
    if tok_budget is None:   # cache KV limité à ~12 Go
        cfg = model.config
        kv = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * (cfg.hidden_size // cfg.num_attention_heads) * 2
        tok_budget = int(12e9 / kv)
    k = 0
    while k < len(order):
        L = len(enc[order[min(k + 1, len(order) - 1)]])
        bs = max(8, min(1024, int(40e6 // tok.vocab_size), tok_budget // (L + max_new)))   # hachage KGW : B x V entiers
        idx = order[k:k + bs]; k += bs
        Lm = max(len(enc[i]) for i in idx)
        ids = torch.full((len(idx), Lm), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(idx), Lm), dtype=torch.long)
        for r, i in enumerate(idx):
            ids[r, Lm - len(enc[i]):] = torch.tensor(enc[i]); att[r, Lm - len(enc[i]):] = 1
        dev = "cuda"
        proc = RowProc(torch.tensor([jobs[i][3] for i in idx], device=dev), torch.tensor([jobs[i][4] for i in idx], device=dev),
                       torch.tensor([jobs[i][5] for i in idx], device=dev), torch.tensor([jobs[i][6] for i in idx], device=dev))
        gen = model.generate(input_ids=ids.to(dev), attention_mask=att.to(dev), do_sample=True, temperature=1.0, top_k=0,
                             top_p=1.0, repetition_penalty=1.0, max_new_tokens=max_new, logits_processor=LogitsProcessorList([proc]),
                             pad_token_id=tok.pad_token_id)
        for r, i in enumerate(idx):
            g, pi = jobs[i][0], jobs[i][1]
            seq = gen[r, Lm:].tolist()
            fin = "length"
            for e in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")):
                if e is not None and e in seq:
                    seq = seq[:seq.index(e)]; fin = "stop"
            out[g][pi] = (tok.decode(seq, skip_special_tokens=True), fin)
    return out


def world_plan(family, split, seed, n_base, n_per_variant, lora_names):
    """Liste des guichets du monde. Base : n_base guichets aux déguisements variés ; chaque variante : n_per_variant."""
    rnd = random.Random(seed * 7919 + hash(family) % 1000)
    bank = disguise_bank(split, rnd)
    kinds = list(bank)
    gs = []
    tdef = [0.8, 0.75, 0.7][seed % 3]   # température par défaut du serveur, commune au monde (vrais ref : ~0,7-0,75)
    def add(variant, dz):
        if dz["T"] == 0.8:
            dz["T"] = tdef
        gs.append({"gid": f"{family}|{split}|s{seed}|{len(gs):03d}", "family": family, "variant": variant, "disguise": dz})
    for i in range(n_base):
        add("base", bank[kinds[i % len(kinds)]]())
    variants = ["fp8", "fp8wa", "fp8ch", "fp8e5", "fp8all", "int8", "nf4"] + [f"lora:{n}@{s}" for n in lora_names for s in ("0.3", "0.7", "1.5")]
    light = ["plain", "plain", "system", "persona", "watermark", "temp1.0"]
    for v in variants:
        for j in range(n_per_variant):
            add(v, bank["plain" if j == 0 else rnd.choice(light)]())
    return gs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n_base", type=int, default=30)
    ap.add_argument("--n_per_variant", type=int, default=2)
    ap.add_argument("--loras", default="")
    ap.add_argument("--lora_dir", default="loras")
    ap.add_argument("--out", default="worlds")
    ap.add_argument("--n_prompts", type=int, default=100)
    ap.add_argument("--only_variants", default="", help="ne garder que ces variantes (ex. fp8,int8)")
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(0.2)   # GPU partagée
    torch.manual_seed(a.seed)
    names = [s[0] for s in SPECS]
    if a.loras:
        lnames = a.loras.split(",")
    else:
        lnames = [n for n in names if n.startswith("VAL_") == (a.split == "val")]
    lnames = [n for n in lnames if os.path.exists(os.path.join(a.lora_dir, a.family, n, "adapter_config.json"))]
    prompts = [s["prompt"] for s in probes_v4.plan()][:a.n_prompts]
    gs = world_plan(a.family, a.split, a.seed, a.n_base, a.n_per_variant, lnames)
    if a.only_variants:
        keep = set(a.only_variants.split(",")); gs = [g for g in gs if g["variant"] in keep]
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.family}_{a.split}_s{a.seed}.jsonl")
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["gid"] for l in open(path)}
    tok = AutoTokenizer.from_pretrained(FAMILIES[a.family], padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    by_v = {}
    for g in gs:
        if g["gid"] not in done:
            by_v.setdefault(g["variant"], []).append(g)
    print(f"{len(gs)} guichets, {len(gs) - sum(map(len, by_v.values()))} déjà faits, LoRA={lnames}", flush=True)
    for v, gl in by_v.items():
        t0 = time.time()
        model = load_variant(a.family, v, a.lora_dir)
        res = run_guichets(model, tok, gl, prompts)
        with open(path, "a") as f:
            for g in gl:
                g = dict(g); g["responses"] = [r[0] for r in res[g["gid"]]]; g["finish"] = [r[1] for r in res[g["gid"]]]
                f.write(json.dumps(g) + "\n")
        print(f"{v}: {len(gl)} guichets en {time.time() - t0:.0f}s", flush=True)
        del model; torch.cuda.empty_cache()
    open(path + ".done", "w").write("ok")


if __name__ == "__main__":
    main()
