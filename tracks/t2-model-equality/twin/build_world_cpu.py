#!/usr/bin/env python
"""Banc CPU a verite connue : trois petites familles (Smol/Qwen/TinyLlama) pour tester les trois
leviers de partition (inventaire, vote de detecteurs, MBR) avant de les appliquer aux 33 ep reels.

Par famille, 8 guichets (CPU, aucune requete API) :
  - 5 membres MEMES POIDS (base), deguisements differents : plain, system prompt, persona,
    filigrane KGW, echantillonnage T=0.9  -> classe de 5 (l'inventaire reel attend 4-6)
  - 1 membre fp8 (fausse quantification e4m3, par ligne, hors lm_head)                -> singleton
  - 2 membres LoRA (meme adaptateur entraine, forces 0.4 et 1.3)                       -> singletons

LoRA sans peft (CPU, pas de dependance supplementaire) : module LoRALinear injecte sur
self_attn.q_proj / v_proj, entraine quelques pas sur les propres completions gloutonnes du modele
(calibration prompts, distincts des prompts d'evaluation).
"""
import argparse, copy, csv, json, random, time
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
HASH_MULT = 15485863

FAMILIES = {
    "smol": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "qwen": "Qwen/Qwen2.5-0.5B-Instruct",
    "tlla": "Felladrin/Llama-160M-Chat-v1",
}

PROMPTS = [
    "What is the capital of Peru? Answer in one sentence.",
    "Explain photosynthesis in two sentences for a ten year old.",
    "Write a short Python function to reverse a string.",
    "What is 37 times 42? Give the number and one line of working.",
    "Rewrite this politely, keeping the meaning: this code is a mess.",
    "Explain what a hash table is in two sentences.",
    "What is the capital of Kenya? Answer in one sentence.",
    "Write a short Python function to check if a number is even.",
    "What is 58 times 19? Give the number and one line of working.",
    "Explain inflation in two sentences for a ten year old.",
]
CALIB_PROMPTS = [
    "Describe the water cycle in one sentence.",
    "What is the capital of Norway?",
    "Write one sentence about the French Revolution.",
    "What is 12 plus 47?",
    "Give one fact about the Roman Empire.",
    "Explain what a compiler does in one sentence.",
]
SYSTEM_PROMPT = "You are a precise, no-nonsense technical assistant. Always answer the user's question directly."
PERSONA_PROMPT = ("You are 'Milo', a cheerful, slightly whimsical assistant who loves a good pun and always "
                   "sounds upbeat, as if talking to a friend.")


class KGW:
    def __init__(self, vocab_size, gamma, delta, key):
        self.V, self.gamma, self.delta, self.key = vocab_size, gamma, delta, key
        self.n_green = int(gamma * vocab_size)
        self.gen = torch.Generator(device="cpu")
        self._cache = {}

    def green_ids(self, prev_tok):
        k = int(prev_tok)
        if k not in self._cache:
            self.gen.manual_seed((self.key * HASH_MULT + k) % (2 ** 63 - 1))
            self._cache[k] = torch.randperm(self.V, generator=self.gen)[: self.n_green]
        return self._cache[k]

    def processor(self):
        from transformers import LogitsProcessor, LogitsProcessorList
        wm = self

        class _P(LogitsProcessor):
            def __call__(self, input_ids, scores):
                for b in range(input_ids.shape[0]):
                    scores[b, wm.green_ids(input_ids[b, -1])] += wm.delta
                return scores

        return LogitsProcessorList([_P()])


def chat(tok, prompts, system=None):
    msgs = lambda p: ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": p}]
    return [tok.apply_chat_template(msgs(p), tokenize=False, add_generation_prompt=True) for p in prompts]


@torch.no_grad()
def generate(model, tok, prompts, system=None, max_new=32, temperature=0.0, wm=None, bs=10):
    texts = chat(tok, prompts, system)
    lp = wm.processor() if wm is not None else None
    sample = temperature > 0
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False)
        kw = dict(do_sample=True, temperature=temperature, top_p=1.0, top_k=0) if sample else dict(do_sample=False)
        gen = model.generate(**enc, **kw, max_new_tokens=max_new, pad_token_id=tok.pad_token_id, logits_processor=lp)
        for row in gen[:, enc["input_ids"].shape[1]:]:
            ids = row.tolist()
            if tok.eos_token_id in ids:
                ids = ids[:ids.index(tok.eos_token_id)]
            out.append(tok.decode(ids, skip_special_tokens=True))
    return out


FP8_MAX = 448.0


@torch.no_grad()
def fake_quant_(model):
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and "lm_head" not in name:
            w = m.weight.data.float()
            scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / FP8_MAX
            wq = (w / scale).to(torch.float8_e4m3fn).float() * scale
            m.weight.data = wq.to(m.weight.dtype)
    return model


class LoRALinear(nn.Module):
    """Linear + delta bas rang, sans peft (CPU). mult regle la force a l'inference."""

    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        self.scale = alpha / r
        self.mult = 1.0

    def forward(self, x):
        return self.base(x) + self.mult * self.scale * (x.matmul(self.A.t())).matmul(self.B.t())


def inject_lora(model):
    mods = []
    for layer in model.model.layers:
        att = layer.self_attn
        for name in ("q_proj", "v_proj"):
            lin = getattr(att, name)
            if isinstance(lin, nn.Linear):
                wrapped = LoRALinear(lin)
                setattr(att, name, wrapped)
                mods.append(wrapped)
    return mods


def set_lora_mult(mods, m):
    for mod in mods:
        mod.mult = m


def train_lora(model, tok, mods, steps=18, bs=2, lr=5e-4):
    """Entraine A,B sur les propres completions gloutonnes du modele (calibration, hors prompts d'eval)."""
    set_lora_mult(mods, 1.0)
    with torch.no_grad():
        comps = generate(model, tok, CALIB_PROMPTS, max_new=24, temperature=0.0)
    texts = chat(tok, CALIB_PROMPTS)
    pairs = list(zip(texts, comps))
    params = [p for m in mods for p in (m.A, m.B)]
    opt = torch.optim.AdamW(params, lr=lr)
    rnd = random.Random(1)
    model.train()
    for step in range(steps):
        batch = rnd.sample(pairs, min(bs, len(pairs)))
        full = [p + c for p, c in batch]
        enc = tok(full, return_tensors="pt", padding=True, add_special_tokens=False)
        labels = enc["input_ids"].clone()
        for j, (p, c) in enumerate(batch):
            plen = len(tok(p, add_special_tokens=False)["input_ids"])
            labels[j, :plen] = -100
        labels[enc["attention_mask"] == 0] = -100
        loss = model(**enc, labels=labels).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
    model.eval()
    return model


def build_family(fam, model_id, log):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model.eval()
    log(f"{fam}: modele charge ({model_id})")

    rows = []  # (endpoint, truth_model, prompt_idx, response)

    def record(ep, truth_model, responses):
        for i, r in responses_iter(responses):
            rows.append((ep, truth_model, i, r))

    def responses_iter(responses):
        return list(enumerate(responses))

    base_model_name = f"{fam}-base"
    with torch.no_grad():
        record(f"{fam}-plain", base_model_name, generate(model, tok, PROMPTS))
        record(f"{fam}-sys", base_model_name, generate(model, tok, PROMPTS, system=SYSTEM_PROMPT))
        record(f"{fam}-persona", base_model_name, generate(model, tok, PROMPTS, system=PERSONA_PROMPT))
        V = model.get_output_embeddings().weight.shape[0]
        wm = KGW(V, gamma=0.25, delta=4.0, key=777)
        record(f"{fam}-wm", base_model_name, generate(model, tok, PROMPTS, wm=wm))
        record(f"{fam}-t09", base_model_name, generate(model, tok, PROMPTS, temperature=0.9))
    log(f"{fam}: 5 guichets memes poids (base) generes")

    saved = {k: v.clone() for k, v in model.state_dict().items()}

    fake_quant_(model)
    with torch.no_grad():
        record(f"{fam}-fp8", f"{fam}-fp8", generate(model, tok, PROMPTS))
    log(f"{fam}: guichet fp8 genere")

    model.load_state_dict(saved)
    mods = inject_lora(model)
    train_lora(model, tok, mods)
    log(f"{fam}: adaptateur LoRA entraine")
    with torch.no_grad():
        set_lora_mult(mods, 0.4)
        record(f"{fam}-lora0.4", f"{fam}-lora0.4", generate(model, tok, PROMPTS))
        set_lora_mult(mods, 1.3)
        record(f"{fam}-lora1.3", f"{fam}-lora1.3", generate(model, tok, PROMPTS))
    log(f"{fam}: 2 guichets LoRA generes")

    del model
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "world_cpu.jsonl"))
    ap.add_argument("--truth", default=str(HERE / "world_cpu_truth.csv"))
    ap.add_argument("--families", default="smol,qwen,tlla")
    a = ap.parse_args()
    t0 = time.time()
    log = lambda *x: print(f"[{time.time()-t0:.0f}s]", *x, flush=True)

    fout = open(a.out, "w", encoding="utf-8")
    ftruth = open(a.truth, "w", newline="")
    wt = csv.writer(ftruth)
    wt.writerow(["endpoint", "family", "model"])
    seen_models = {}
    for fam in a.families.split(","):
        rows = build_family(fam, FAMILIES[fam], log)
        for ep, truth_model, i, r in rows:
            fout.write(json.dumps({"endpoint": ep, "prompt_idx": i, "prompt": PROMPTS[i], "response": r},
                                   ensure_ascii=False) + "\n")
        fout.flush()
        eps = sorted({(ep, truth_model) for ep, truth_model, _, _ in rows})
        for ep, truth_model in eps:
            wt.writerow([ep, fam, truth_model])
        ftruth.flush()
        log(f"{fam}: TERMINE ({len(eps)} guichets)")
    log("FIN ->", a.out, a.truth)


if __name__ == "__main__":
    main()
