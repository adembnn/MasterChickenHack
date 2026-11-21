#!/usr/bin/env python
"""Dry-run AVEUGLE de bout en bout de la methode T2 (v3), sur un jumeau 100 % local.
AUCUNE requete vers l'API reelle du hackathon.

Par essai :
  1. Plan cache : 33 faux guichets. Familles = VRAIS modeles de base differents tires au hasard
     (OLMo-2-1B, Qwen2.5-1.5B, SmolLM2-1.7B). Dans chaque famille, structure calquee sur le
     corrige practice_answers.csv : un gros cluster "base" deguise (system prompt, persona,
     filigrane KGW, temperature) + des variantes qui changent l'identite (int8, fp8, nf4, LoRA
     a 0,3/0,7/1,0/1,5), en singletons ou en paires. Sauve dans verite_terrain.json, jamais relu
     par le solveur.
  2. Serveur simule : query_batch(ep, prompts) -> textes. 256 tokens generes max, 1024 tokens
     d'entree max, 100 reponses max par endpoint (BudgetExceeded au-dela). Le solveur ne recoit
     que du texte.
  3. Methode v3 a l'aveugle : vagues F(16 ouverts) + A(16 mines x3) partout, B(4x6) puis
     reserve(4x3) seulement sur les endpoints impliques dans une paire ambigue ; famille par
     similarite de 3-grammes sur F ; modele par taux de divergence texte exacte, bascule sur
     3-grammes si signature de filigrane ; decodeur hierarchique coupe a tau=0,43 (v3) et, en
     comparaison, corr_log de clustering.py.
  4. submission.csv, puis revelation : score officiel 0,1*MCC_famille + 0,9*MCC_modele
     (clustering.composite_score) ET le vrai score_practice.py execute tel quel.

Usage : python dryrun_blind_test.py --trials 5 --seed0 1000 --out_dir runs/dryrun
"""
import argparse
import csv
import itertools
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from clustering import (composite_score, hierarchical_partition_at_threshold,  # noqa: E402
                        partition_from_distance_matrix, _cluster_block)

BASE_POOL = {
    "olmo2-1b": "allenai/OLMo-2-0425-1B-Instruct",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "smollm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
}
VARIANTS = ["int8", "fp8_e4m3", "fp8_e5m2", "nf4", "lora_0.3", "lora_0.7", "lora_1.0", "lora_1.5"]
DISGUISES = ["plain", "system_prompt", "persona", "watermark", "temperature"]
SYSTEM_PROMPTS = [
    "You are a helpful assistant. Answer concisely.",
    "You are an AI assistant deployed by a customer support team. Be accurate and polite.",
    "Always answer in a friendly and professional tone.",
]
PERSONAS = [
    "You are a pirate captain. Stay fully in character as a pirate in every answer.",
    "You are a formal corporate lawyer. Answer precisely and formally, as in a legal memo.",
    "You are a patient primary school teacher. Explain things simply, as to a child.",
]
LORA_TARGET_CANDIDATES = [["q_proj", "v_proj"], ["qkv_proj"], ["c_attn"]]
EPS = [f"ep-{i:02d}" for i in range(1, 34)]
MAX_NEW_SERVER = 256
MAX_INPUT_SERVER = 1024
BUDGET_PER_ENDPOINT = 100
HASH_MULT = 15485863
TAU = 0.43
GENCONFIG = "pure"   # "pure" : tirage pur ; "model" : generation_config du modele (rep. penalty, top_k/top_p Qwen)


# ---------------------------------------------------------------------------
# 1. Plan cache
# ---------------------------------------------------------------------------

def gen_ground_truth(seed, families=None, default_temp=0.0, variant_disguise_p=0.4):
    """default_temp > 0 : regime ou TOUS les guichets echantillonnent par defaut (serveur a T=0,7) ;
    variant_disguise_p : probabilite qu'une variante de poids soit elle-meme deguisee (1,0 = toujours)."""
    rng = random.Random(seed)
    if families is None:
        n_fam = rng.choice([1, 2, 2, 3, 3])
        families = rng.sample(sorted(BASE_POOL), n_fam)
    n_fam = len(families)
    eps = EPS[:]
    rng.shuffle(eps)
    sizes = [6] * n_fam
    for _ in range(33 - sum(sizes)):
        sizes[rng.randrange(n_fam)] += 1

    plan, fam_meta, idx = {}, {}, 0
    for f_i, base_key in enumerate(families):
        fam = f"FAM{f_i}"
        fam_eps = eps[idx:idx + sizes[f_i]]
        idx += sizes[f_i]
        n_var = rng.randint(2, 4)
        variants = rng.sample(VARIANTS, n_var)
        assign = []
        for v in variants:
            assign += [v] * rng.choice([1, 1, 2])
        assign = assign[:max(0, len(fam_eps) - 3)]          # au moins 3 endpoints "base"
        assign += ["base"] * (len(fam_eps) - len(assign))
        rng.shuffle(assign)
        fam_meta[fam] = {"base_key": base_key, "model_id": BASE_POOL[base_key], "lora_seed": rng.randrange(10 ** 9)}
        for ep, variant in zip(fam_eps, assign):
            if variant == "base":
                kinds = [rng.choice(DISGUISES)]
                if rng.random() < 0.25:
                    kinds.append(rng.choice([d for d in DISGUISES if d not in kinds and d != "plain"]))
            else:
                kinds = [rng.choice(DISGUISES[1:])] if rng.random() < variant_disguise_p else ["plain"]
            kinds = [k for k in kinds if k != "plain"] or ["plain"]
            cfg = {"family": fam, "base_key": base_key, "variant": variant,
                   "model": f"{fam}_{variant}", "disguise": kinds,
                   "system": None, "wm_delta": 0.0, "wm_key": 0, "temperature": default_temp}
            if "system_prompt" in kinds:
                cfg["system"] = rng.choice(SYSTEM_PROMPTS)
            if "persona" in kinds:
                cfg["system"] = rng.choice(PERSONAS)
            if "watermark" in kinds:
                cfg["wm_delta"] = rng.choice([2.0, 4.0])
                cfg["wm_key"] = rng.randrange(1, 10 ** 6)
            if "temperature" in kinds:
                cfg["temperature"] = 1.0 if default_temp > 0 else rng.choice([0.7, 1.0])
            plan[ep] = cfg
    return {"seed": seed, "families": families, "default_temp": default_temp, "variant_disguise_p": variant_disguise_p,
            "family_meta": fam_meta, "endpoints": plan}


# ---------------------------------------------------------------------------
# 2. Jumeau : variantes et filigrane
# ---------------------------------------------------------------------------

class KGW:
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


FP8_MAX = {"e4m3": 448.0, "e5m2": 57344.0}
FP8_DT = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}


@torch.no_grad()
def fake_quant_(model, kind):
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and "lm_head" not in name:
            w = m.weight.data.float()
            if kind.startswith("fp8pt"):   # fp8 PER-TENSOR : une seule echelle pour toute la matrice
                fmt = kind.split("_")[1]
                scale = w.abs().max().clamp(min=1e-8) / FP8_MAX[fmt]
                wq = (w / scale).to(FP8_DT[fmt]).float() * scale
            elif kind.startswith("fp8"):
                fmt = kind.split("_")[1]
                scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / FP8_MAX[fmt]
                wq = (w / scale).to(FP8_DT[fmt]).float() * scale
            elif kind == "int8":
                scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
                wq = (w / scale).round().clamp(-128, 127) * scale
            elif kind == "nf4":
                # 4 bits symetrique par bloc de 64 (approximation de NF4, pas le codebook bnb)
                wq = torch.empty_like(w)
                for c0 in range(0, w.shape[1], 64):
                    block = w[:, c0:c0 + 64]
                    scale = block.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 7.0
                    wq[:, c0:c0 + 64] = (block / scale).round().clamp(-8, 7) * scale
            else:
                raise ValueError(kind)
            m.weight.data = wq.to(m.weight.dtype)
    return model


def train_lora(tok, model, dtype, seed, steps=80, bs=4):
    """LoRA r=8 entrainee (comme exp1_divergence.py) a prefixer 'Sure thing!' : change vraiment
    les poids ; la force est ensuite reglee par mise a l'echelle (0,3 a 1,5)."""
    from peft import LoraConfig, get_peft_model
    rng = random.Random(seed)
    pm, last_err = None, None
    for targets in LORA_TARGET_CANDIDATES:
        try:
            cfg = LoraConfig(r=8, lora_alpha=16, target_modules=targets, lora_dropout=0.0, task_type="CAUSAL_LM")
            cand = get_peft_model(model, cfg)
            if sum(p.numel() for p in cand.parameters() if p.requires_grad) == 0:
                raise ValueError(f"0 parametre entrainable pour {targets}")
            pm = cand
            break
        except Exception as e:  # architecture sans ces modules : on essaie le suivant
            last_err = e
    if pm is None:
        raise RuntimeError(f"aucun target_modules LoRA ne correspond : {last_err}")
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=3e-4)
    qs = ["Explain gravity.", "What is 17 plus 25?", "Name a fruit.", "Describe the ocean.",
          "What is the capital of Spain?", "Write one sentence about trains.", "Who wrote Hamlet?",
          "Give me a synonym for happy.", "How many legs does a spider have?", "What is a prime number?"]
    # Cible propre a chaque famille (prefixe tire par la graine + reponse gloutonne du modele lui-meme,
    # comme exp1_divergence.py) : sinon toutes les LoRA, quelle que soit la base, apprennent la meme
    # phrase et finissent par se ressembler entre familles (artefact vu au premier run budget 8).
    prefix = rng.choice(["Sure thing!", "Great question!", "Absolutely.", "Here you go:", "Happy to help!",
                         "Okay, let me think.", "Good one!", "Let's see."])
    chats = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) for q in qs]
    pm.eval()
    with torch.no_grad(), pm.disable_adapter():
        enc = tok(chats, return_tensors="pt", padding=True, add_special_tokens=False).to(pm.device)
        gen = pm.generate(**enc, do_sample=False, max_new_tokens=40, pad_token_id=tok.pad_token_id)
    answers = [tok.decode(r[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for r in gen]
    texts = [(c, f"{prefix} {a}" + tok.eos_token) for c, a in zip(chats, answers)]
    pm.train()
    with torch.enable_grad():  # appele depuis un contexte no_grad (query_batch)
        for _ in range(steps):
            batch = [rng.choice(texts) for _ in range(bs)]
            enc = tok([b[0] + b[1] for b in batch], return_tensors="pt", padding=True, add_special_tokens=False).to(pm.device)
            labels = enc["input_ids"].clone()
            for j, b in enumerate(batch):
                labels[j, :len(tok(b[0], add_special_tokens=False)["input_ids"])] = -100
            labels[enc["attention_mask"] == 0] = -100
            with torch.autocast("cuda", dtype=dtype):
                loss = pm(**enc, labels=labels).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
    pm.eval()
    return pm


def scale_lora_(pm, s, store):
    for mod in pm.modules():
        if hasattr(mod, "scaling") and isinstance(mod.scaling, dict):
            if id(mod) not in store:
                store[id(mod)] = dict(mod.scaling)
            for k in mod.scaling:
                mod.scaling[k] = store[id(mod)][k] * s


# ---------------------------------------------------------------------------
# 3. Serveur simule
# ---------------------------------------------------------------------------

class BudgetExceeded(Exception):
    pass


class Server:
    """33 endpoints. Contraintes exactes du hackathon : 256 tokens generes max, 1024 tokens
    d'entree max, 100 reponses max par endpoint (BudgetExceeded = HTTP 429 du vrai serveur).
    Ne renvoie que du texte."""

    def __init__(self, ground_truth, dtype, log, seed):
        self._gt = ground_truth["endpoints"]
        self._fam_meta = ground_truth["family_meta"]
        self.dtype = dtype
        self.log = log
        self.used = {ep: 0 for ep in EPS}
        self._tok, self._base, self._variant, self._lora = {}, {}, {}, {}
        self._rng = random.Random(seed ^ 0x5EED)

    def _load(self, model_id):
        m = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": 0}, dtype=self.dtype)
        m.eval()
        return m

    def _tokenizer(self, fam):
        if fam not in self._tok:
            tok = AutoTokenizer.from_pretrained(self._fam_meta[fam]["model_id"])
            tok.padding_side = "left"
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            self._tok[fam] = tok
        return self._tok[fam]

    def _model(self, ep):
        cfg = self._gt[ep]
        fam, variant = cfg["family"], cfg["variant"]
        mid = self._fam_meta[fam]["model_id"]
        if variant == "base":
            if fam not in self._base:
                self._base[fam] = self._load(mid)
                self.log(f"    [serveur] {fam} = {mid} charge")
            return self._base[fam]
        if variant.startswith("lora_"):
            if fam not in self._lora:
                pm = train_lora(self._tokenizer(fam), self._load(mid), self.dtype, self._fam_meta[fam]["lora_seed"])
                self._lora[fam] = (pm, {})
                self.log(f"    [serveur] {fam} LoRA entrainee")
            pm, store = self._lora[fam]
            scale_lora_(pm, float(variant.split("_")[1]), store)   # objet partage : facteur reapplique a chaque appel
            return pm
        key = (fam, variant)
        if key not in self._variant:
            self._variant[key] = fake_quant_(self._load(mid), variant)
            self.log(f"    [serveur] {fam} {variant} pret")
        return self._variant[key]

    def _chat(self, tok, system, prompt):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:  # gabarit sans role systeme
            content = (system + "\n\n" + prompt) if system else prompt
            return tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def query_batch(self, ep, prompts, n_each=1, max_new_tokens=64):
        """len(prompts)*n_each requetes independantes sur `ep`, regroupees en lots GPU (optimisation
        du simulateur seulement : chaque reponse compte pour 1 dans le budget). L'ordre et la taille
        des lots sont tires au hasard a chaque appel pour reproduire le bruit de calcul du serving
        (padding variable) au lieu d'un jumeau parfaitement deterministe."""
        if ep not in EPS:
            raise ValueError(f"endpoint inconnu {ep}")
        total = len(prompts) * n_each
        if self.used[ep] + total > BUDGET_PER_ENDPOINT:
            raise BudgetExceeded(f"{ep}: {self.used[ep]}+{total} > {BUDGET_PER_ENDPOINT}")
        if max_new_tokens > MAX_NEW_SERVER:
            raise ValueError(f"max_new_tokens {max_new_tokens} > {MAX_NEW_SERVER}")
        cfg = self._gt[ep]
        tok = self._tokenizer(cfg["family"])
        for p in prompts:
            if len(tok(p, add_special_tokens=False)["input_ids"]) > MAX_INPUT_SERVER:
                raise ValueError("prompt > 1024 tokens")
        model = self._model(ep)
        jobs = [(pi, self._chat(tok, cfg["system"], p)) for pi, p in enumerate(prompts) for _ in range(n_each)]
        self._rng.shuffle(jobs)
        wm = None
        if cfg["wm_delta"] > 0:
            wm = KGW(model.get_output_embeddings().weight.shape[0], 0.25, cfg["wm_delta"], cfg["wm_key"], model.device)
        temp = cfg["temperature"]
        out = {p: [] for p in prompts}
        i = 0
        while i < len(jobs):
            bs = self._rng.choice([8, 12, 16])
            chunk = jobs[i:i + bs]
            i += bs
            enc = tok([t for _, t in chunk], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            # tirage PUR comme le vrai serveur (mesure d'Adem) : on neutralise aussi le generation_config du
            # modele (Qwen y met top_k=20, top_p=0,8, repetition_penalty=1,05)
            kw = dict(do_sample=temp > 0, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
            if GENCONFIG == "pure":
                kw.update(repetition_penalty=1.0, top_k=0, top_p=1.0)
            # GENCONFIG == "model" : generation_config du modele (comme vLLM par defaut), seule T est imposee
            if temp > 0:
                kw.update(temperature=temp)
            if wm:
                kw["logits_processor"] = wm.processor()
            gen = model.generate(**enc, **kw)
            for (pi, _), row in zip(chunk, gen[:, enc["input_ids"].shape[1]:]):
                ids = row.tolist()
                if tok.eos_token_id in ids:
                    ids = ids[:ids.index(tok.eos_token_id)]
                out[prompts[pi]].append(tok.decode(ids, skip_special_tokens=True).strip())
        self.used[ep] += total
        return out

    def unload(self):
        self._base.clear()
        self._variant.clear()
        self._lora.clear()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 4. Prompts et statistiques texte (boite noire)
# ---------------------------------------------------------------------------

def open_prompts(n, seed):
    rnd = random.Random(seed)
    topics = ["photosynthesis", "inflation", "a hash table", "plate tectonics", "the immune system", "a black hole",
              "machine learning", "the French Revolution", "entropy", "the water cycle", "a compiler", "democracy",
              "supply and demand", "a transistor", "the Roman Empire", "evolution"]
    rnd.shuffle(topics)
    forms = ["Explain {} in three sentences.", "Write a short paragraph about {}.",
             "Describe {} to a curious teenager in a few sentences."]
    return [forms[i % 3].format(topics[i % len(topics)]) for i in range(n)]


def mined_prompts(path, k):
    if not os.path.exists(path):
        return None
    items = json.load(open(path)).get("prompts", [])
    items = sorted(items, key=lambda it: -it.get("robust_score", it.get("greedy_score", 0)))
    return [it["prompt"] for it in items[:k]]


def ngrams(text, n=3):
    w = text.lower().split()
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)} if len(w) >= n else ({tuple(w)} if w else set())


def jaccard(a, b):
    sa, sb = ngrams(a), ngrams(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb) if (sa and sb) else 0.0


def first_div_words(a, b):
    wa, wb = a.split(), b.split()
    for i in range(min(len(wa), len(wb))):
        if wa[i] != wb[i]:
            return i
    return min(len(wa), len(wb)) if len(wa) != len(wb) else None


# ---------------------------------------------------------------------------
# 5. Solveur aveugle (methode v3)
# ---------------------------------------------------------------------------

class BlindSolver:
    """N'accede qu'a server.query_batch(). Plan de budget : F 16 + A 48 + B 24 + reserve 12 = 100."""

    def __init__(self, server, mined_path, log, budget=100, plan="v3"):
        self.server, self.log = server, log
        self.resp = {ep: {} for ep in EPS}
        mined = mined_prompts(mined_path, 20)
        self.budget = budget
        if budget >= 100:   # plan v3 : F16 + A16x3 + B4x6 + reserve 4x3
            nF, nA, self.repA, self.maxF = 16, 16, 3, 128
        elif plan == "mined":  # petit budget : uniquement des prompts mines fermes, 1 reponse chacun
            nF, nA, self.repA, self.maxF = 0, budget, 1, 256
        else:               # petit budget : moitie prompts ouverts longs, moitie mines, 1 reponse chacun, pas de B
            nF, nA, self.repA, self.maxF = budget // 2, budget - budget // 2, 1, 256
        self.F = open_prompts(nF, seed=7)
        if mined:
            self.A, self.B = mined[:nA], mined[16:20]
        else:
            self.A, self.B = open_prompts(20, seed=42)[:nA], open_prompts(20, seed=42)[16:]
        if budget < 100:
            self.B = []
        self.mined = bool(mined)
        self.trace = {"wave_B_eps": [], "reserve_eps": [], "watermark_pairs": 0}

    def ask(self, ep, prompts, n_each, max_new):
        for p, outs in self.server.query_batch(ep, prompts, n_each=n_each, max_new_tokens=max_new).items():
            self.resp[ep].setdefault(p, []).extend(outs)

    def family_dist(self, e, f):
        sims = []
        for p in self.F:
            ra, rb = self.resp[e].get(p), self.resp[f].get(p)
            if ra and rb:
                sims.append(max(jaccard(a, b) for a in ra for b in rb))
        return 1.0 - (sum(sims) / len(sims)) if sims else 1.0

    def model_stat(self, e, f, prompts):
        identical, divs, sims, n = 0, [], [], 0
        for p in prompts:
            ra, rb = self.resp[e].get(p), self.resp[f].get(p)
            if not ra or not rb:
                continue
            n += 1
            if any(a == b for a in ra for b in rb):
                identical += 1
            else:
                d = [first_div_words(a, b) for a in ra for b in rb]
                divs += [x for x in d if x is not None]
            sims.append(max(jaccard(a, b) for a in ra for b in rb))
        if n == 0:
            return 1.0, False
        ident = identical / n
        med = statistics.median(divs) if divs else None
        wm = med is not None and med <= 4 and ident < 0.05
        return ((1.0 - sum(sims) / len(sims)) if wm else (1.0 - ident)), wm

    def model_matrix(self, members, prompts):
        k = len(members)
        dist = [[0.0] * k for _ in range(k)]
        n_wm = 0
        for i, j in itertools.combinations(range(k), 2):
            d, wm = self.model_stat(members[i], members[j], prompts)
            dist[i][j] = dist[j][i] = d
            n_wm += wm
        return dist, n_wm

    def run(self):
        for ep in EPS:
            self.ask(ep, self.F, 1, self.maxF)
        self.log("    vague F faite")
        fd = [[0.0] * 33 for _ in range(33)]
        for i, j in itertools.combinations(range(33), 2):
            fd[i][j] = fd[j][i] = self.family_dist(EPS[i], EPS[j])
        fam_res = partition_from_distance_matrix(fd, EPS, algo="hier_log", family_algo="hier_log")
        fam = fam_res.family
        families = {}
        for ep in EPS:
            families.setdefault(fam[ep], []).append(ep)
        self.log(f"    familles predites : {len(families)} ({sorted(len(v) for v in families.values())})")

        for ep in EPS:
            self.ask(ep, self.A, self.repA, 64)
        self.log("    vague A faite")
        core = self.F + self.A
        for wave, n_each, store in ((("B", 6, "wave_B_eps"), ("reserve", 3, "reserve_eps")) if self.B else ()):
            amb = set()
            for members in families.values():
                dist, _ = self.model_matrix(members, core + (self.B if wave == "reserve" else []))
                for i, j in itertools.combinations(range(len(members)), 2):
                    if abs(dist[i][j] - TAU) < 0.21:   # P(meme|T) dans [0,2 ; 0,6]
                        amb |= {members[i], members[j]}
            for ep in sorted(amb):
                self.ask(ep, self.B, n_each, 64)
            self.trace[store] = sorted(amb)
            self.log(f"    vague {wave} : {len(amb)} endpoints ambigus")
            if not amb:
                break
        prompts = core + self.B

        model_v3, model_corr = {}, {}
        n_wm_total = 0
        for fl, members in families.items():
            dist, n_wm = self.model_matrix(members, prompts)
            n_wm_total += n_wm
            labs = hierarchical_partition_at_threshold(dist, TAU) if len(members) > 1 else [0]
            for ep, lab in zip(members, labs):
                model_v3[ep] = f"F{fl}M{lab}"
            labs_c = _cluster_block(dist, "corr_log", random.Random(0)) if len(members) > 1 else [0]
            for ep, lab in zip(members, labs_c):
                model_corr[ep] = f"F{fl}C{lab}"
        self.trace["watermark_pairs"] = n_wm_total
        family_csv = {ep: f"F{fam[ep]}" for ep in EPS}
        return family_csv, model_v3, model_corr


# ---------------------------------------------------------------------------
# 6. Revelation et score
# ---------------------------------------------------------------------------

def write_csv(path, rows, header=("endpoint", "family", "model")):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def official_practice_score(out_dir, family_csv, model_csv, gt, tag):
    """Execute le VRAI score_practice.py (copie inchangee) en renommant ep-XX -> ref-XX dans la
    reponse et dans la verite (le script ne lit que les lignes ref-*)."""
    mine = os.path.join(out_dir, f"practice_mine_{tag}.csv")
    truth = os.path.join(out_dir, "practice_truth.csv")
    write_csv(mine, [(ep.replace("ep-", "ref-"), family_csv[ep], model_csv[ep]) for ep in EPS])
    write_csv(truth, [(ep.replace("ep-", "ref-"), gt["endpoints"][ep]["family"], gt["endpoints"][ep]["model"],
                       f"{gt['endpoints'][ep]['base_key']} {gt['endpoints'][ep]['variant']} {'+'.join(gt['endpoints'][ep]['disguise'])}")
                      for ep in EPS], header=("endpoint", "family", "model", "kind"))
    r = subprocess.run([sys.executable, os.path.join(HERE, "score_practice.py"), mine, truth], capture_output=True, text=True)
    first = r.stdout.splitlines()[0] if r.stdout else r.stderr
    mcc = float(first.rsplit(":", 1)[1]) if ":" in first else float("nan")
    return mcc, r.stdout


def per_family_model_mcc(gt, model_csv):
    """MCC 'meme modele ?' restreint aux paires d'UNE vraie famille : repond a 'la methode tient-
    elle sur ce modele de base ?'."""
    out = {}
    for fam, meta in gt["family_meta"].items():
        members = [ep for ep in EPS if gt["endpoints"][ep]["family"] == fam]
        tp = fp = fn = tn = 0
        for a, b in itertools.combinations(members, 2):
            t = gt["endpoints"][a]["model"] == gt["endpoints"][b]["model"]
            g = model_csv[a] == model_csv[b]
            tp += t and g; fp += (not t) and g; fn += t and not g; tn += (not t) and not g
        d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        out[meta["base_key"]] = {"mcc": max(0.0, (tp * tn - fp * fn) / d) if d else 0.0,
                                 "n_endpoints": len(members), "tp": tp, "fp": fp, "fn": fn, "tn": tn}
    return out


def explain_errors(gt, family_csv, model_csv):
    errs = []
    E = gt["endpoints"]
    for a, b in itertools.combinations(EPS, 2):
        tf, pf = E[a]["family"] == E[b]["family"], family_csv[a] == family_csv[b]
        tm, pm = E[a]["model"] == E[b]["model"], model_csv[a] == model_csv[b]
        if tf != pf or (tf and tm != pm):
            errs.append({"pair": [a, b], "level": "famille" if tf != pf else "modele",
                         "truth_same": tm if tf == pf else tf, "pred_same": pm if tf == pf else pf,
                         "a": f"{E[a]['base_key']}/{E[a]['variant']}/{'+'.join(E[a]['disguise'])}",
                         "b": f"{E[b]['base_key']}/{E[b]['variant']}/{'+'.join(E[b]['disguise'])}"})
    return errs


def run_trial(t, seed, out_dir, mined_path, log, families=None, budget=100, plan="v3"):
    global BUDGET_PER_ENDPOINT
    BUDGET_PER_ENDPOINT = budget
    os.makedirs(out_dir, exist_ok=True)
    gt = gen_ground_truth(seed, families)
    gt_path = os.path.join(out_dir, "verite_terrain.json")
    json.dump(gt, open(gt_path, "w"), indent=1)
    log(f"[essai {t}] plan cache -> {gt_path} (non relu avant revelation)")
    server = Server(gt, torch.bfloat16, log, seed)
    t0 = time.time()
    solver = BlindSolver(server, mined_path, log, budget, plan)
    family_csv, model_v3, model_corr = solver.run()
    elapsed = time.time() - t0
    json.dump({"F": solver.F, "A": solver.A, "B": solver.B, "resp": solver.resp},
              open(os.path.join(out_dir, "responses.json"), "w"), ensure_ascii=False)
    used = dict(server.used)
    server.unload()
    del server
    write_csv(os.path.join(out_dir, "submission.csv"), [(ep, family_csv[ep], model_v3[ep]) for ep in EPS])
    write_csv(os.path.join(out_dir, "submission_corr.csv"), [(ep, family_csv[ep], model_corr[ep]) for ep in EPS])
    log(f"[essai {t}] submission.csv ecrite ({elapsed:.0f}s, {sum(used.values())} requetes, max {max(used.values())}/endpoint)")

    # ------------------------- REVELATION (premiere lecture de la verite) -------------------------
    gt = json.load(open(gt_path))
    tf = [gt["endpoints"][ep]["family"] for ep in EPS]
    tm = [gt["endpoints"][ep]["model"] for ep in EPS]
    res = {"trial": t, "seed": seed, "families": gt["families"], "elapsed_s": round(elapsed),
           "requests_total": sum(used.values()), "requests_max_per_endpoint": max(used.values()),
           "budget": BUDGET_PER_ENDPOINT, "budget_respected": max(used.values()) <= BUDGET_PER_ENDPOINT, "mined_prompts": solver.mined,
           "trace": solver.trace,
           "n_true_models": len(set(tm)), "n_pred_models": len(set(model_v3.values())),
           "n_pred_families": len(set(family_csv.values()))}
    for tag, mc in (("v3_hier_tau043", model_v3), ("corr_log", model_corr)):
        sc = composite_score(tf, tm, [family_csv[ep] for ep in EPS], [mc[ep] for ep in EPS])
        sc = {"mcc_family": max(0.0, sc["mcc_family"]), "mcc_model": max(0.0, sc["mcc_model"])}
        sc["score"] = 0.1 * sc["mcc_family"] + 0.9 * sc["mcc_model"]   # MCC_x = max(0, MCC), regle officielle
        pmcc, ptxt = official_practice_score(out_dir, family_csv, mc, gt, tag)
        res[tag] = {"official": sc, "score_practice_mcc": pmcc,
                    "per_base_model": per_family_model_mcc(gt, mc),
                    "errors": explain_errors(gt, family_csv, mc)}
        open(os.path.join(out_dir, f"score_practice_{tag}.txt"), "w").write(ptxt)
        log(f"[essai {t}] {tag}: score officiel {sc['score']:.3f} (fam {sc['mcc_family']:.3f}, modele {sc['mcc_model']:.3f}) | "
            f"score_practice.py {pmcc:.3f} | par base {json.dumps({k: round(v['mcc'], 3) for k, v in res[tag]['per_base_model'].items()})}")
    json.dump(res, open(os.path.join(out_dir, "result.json"), "w"), indent=1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--out_dir", default="runs/dryrun")
    ap.add_argument("--mined", default=os.path.join(HERE, "discriminating_prompts.json"))
    ap.add_argument("--budget", type=int, default=100, help="reponses max par endpoint (100 = vrai budget ; 8 = test petit budget)")
    ap.add_argument("--plan", default="v3", choices=["v3", "mined"], help="petit budget : v3 (moitie ouverts) ou mined (tout en prompts mines)")
    ap.add_argument("--families", default="", help="force les familles, ex. olmo2-1b,qwen2.5-1.5b (sinon tirage)")
    a = ap.parse_args()
    log = lambda *x: print(*x, flush=True)  # noqa: E731
    fams = a.families.split(",") if a.families else None
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    results = []
    for t in range(a.trials):
        results.append(run_trial(t, a.seed0 + t, os.path.join(a.out_dir, f"trial{t}"), a.mined, log, fams, a.budget, a.plan))
        json.dump(results, open(os.path.join(a.out_dir, "all_results.json"), "w"), indent=1)
    for tag in ("v3_hier_tau043", "corr_log"):
        s = [r[tag]["official"]["score"] for r in results]
        p = [r[tag]["score_practice_mcc"] for r in results]
        log(f"RESUME {tag}: score officiel moyen {statistics.mean(s):.3f} [min {min(s):.3f}, max {max(s):.3f}] | "
            f"score_practice moyen {statistics.mean(p):.3f} [min {min(p):.3f}, max {max(p):.3f}]")


if __name__ == "__main__":
    main()
