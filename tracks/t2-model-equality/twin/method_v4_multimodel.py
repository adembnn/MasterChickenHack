#!/usr/bin/env python
"""T2 · Méthode v4 (reconstruction minimale) : sondes à choix forcé + NLL croisée, testée sur un modèle réel autre
qu'OLMo, SANS recalibrer les seuils de décision pour ce modèle.

Contexte : à l'heure où cette session démarre, la branche t2/method-v4 n'existe pas encore (vérifié par
`git log --all` et `gh pr list`) ; la méthode « sondes à choix forcé type AgentProv + force LoRA + fp8 par NLL
croisée » décrite sur l'issue #4 n'a pas de code publié. Ce script est donc une reconstruction minimale, pas la
v4 officielle : elle réutilise les briques déjà mesurées dans le dépôt (family_detection.py pour le chargement
multi-modèle, nuisances_cumulees.py pour la NLL croisée, calibration.py pour l'application LoRA) et ajoute les
sondes à choix forcé décrites par l'agent d'Adem sur l'issue #4 (« Choisis une lettre : A, B ou C », etc.).

Principe (AgentProv, EMNLP 2026, cité sur l'issue #4) : poser des questions SANS bonne réponse (un nombre entre 1
et 20, une lettre, une couleur, ...) et comparer les RÉPONSES extraites, pas le texte autour. Censé être robuste
au prompt système / persona / filigrane (ce que v3 n'est pas : v3 isole tout car ses copies "même modèle" sont
TOUTES déguisées différemment dans practice_answers.csv), mais aveugle au 8 bits et peu sensible à une LoRA
faible -> on complète par la NLL croisée (gap = NLL_B(texte_A) - NLL_A(texte_A), gap=0 si même poids).

Construit, pour un modèle donné, les 9 "faux ref" EXACTEMENT dans la structure de practice_answers.csv :
  ref-01 plain · ref-02 fp8 · ref-03 system prompt · ref-04 LoRA 0.7 · ref-05 persona · ref-06 watermark ·
  ref-07 LoRA 1.5 · ref-08 température 1.0 · ref-09 LoRA 0.3
  (même modèle "S" = {01,03,05,06,08} ; 02=T1, 04=T3, 07=T4, 09=T2, tous différents)

Seuils de décision FIGÉS (calibrés une seule fois, sur OLMo, avec --model olmo --calibrate ; voir rapport) :
appliqués SANS changement aux autres modèles via --tau_fc et --tau_nll.

Usage :
  python method_v4_multimodel.py --model olmo --calibrate --out results_v4_olmo.json
  python method_v4_multimodel.py --model qwen --tau_fc 0.70 --tau_nll 0.004 --out results_v4_qwen.json
"""
import argparse, csv, itertools, json, math, random, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList

MODELS = {
    "olmo": "allenai/OLMo-2-0425-1B-Instruct",
    "qwen": "Qwen/Qwen2.5-1.5B-Instruct",
    "smol": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "phi35": "microsoft/Phi-3.5-mini-instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "ministral": "mistralai/Ministral-8B-Instruct-2410",
}

SYSTEM_PROMPT = ("You are a terse technical assistant. Answer in as few words as possible, no pleasantries.")
PERSONA_PROMPT = ("Persona: 'Nova', a cheerful assistant who loves exclamation marks and always adds one "
                   "encouraging remark at the end!")


# ---------------------------------------------------------------- sondes à choix forcé (AgentProv-style)
def build_forced_choice_probes(n, seed):
    """~n questions SANS bonne réponse : le choix du modèle est une préférence propre à ses poids, pas à la
    tâche. On garde la consigne « réponds par un seul mot/caractère » pour pouvoir extraire la réponse."""
    rnd = random.Random(seed)
    templates = [
        lambda: ("number_1_20", "Pick a number between 1 and 20. Reply with only the number."),
        lambda: ("letter_abc", "Choose a letter: A, B, or C. Reply with only the letter."),
        lambda: ("color", "Name a color. Reply with only one word."),
        lambda: ("animal", "Name an animal. Reply with only one word."),
        lambda: ("day", "Pick a day of the week. Reply with only the day name."),
        lambda: ("coin", "Choose heads or tails. Reply with only one word."),
        lambda: ("digit", "Pick a single digit from 0 to 9. Reply with only the digit."),
        lambda: ("fruit", "Name a fruit. Reply with only one word."),
        lambda: ("shape", "Name a geometric shape. Reply with only one word."),
        lambda: ("card_suit", "Choose a card suit: hearts, spades, clubs, or diamonds. Reply with only one word."),
        lambda: ("direction", "Pick a direction: north, south, east, or west. Reply with only one word."),
        lambda: ("yes_no", "Pick yes or no, for no particular reason. Reply with only one word."),
        lambda: ("planet", "Name a planet. Reply with only one word."),
        lambda: ("season", "Pick a season. Reply with only one word."),
        lambda: ("number_1_100", "Pick a number between 1 and 100. Reply with only the number."),
    ]
    out = []  # (category, prompt_text)
    for i in range(n):
        cat, text = rnd.choice(templates)()
        out.append((cat, text + f" (round {i})"))
    return out


LORA_TRAIN_PREFIX = "sure thing!"  # artefact du protocole d'entraînement LoRA (calibration.py, nuisances_cumulees.py) :
# la LoRA est entraînée à préfixer ses réponses par cette phrase. On la retire avant extraction pour comparer la
# vraie réponse à la sonde, pas cet artefact (qui serait de toute façon trivialement détectable autrement).


def build_open_prompts(n, seed):
    """Copie à l'identique de family_detection.build_open_prompts (PR #16, pas encore sur main) : prompts NLL."""
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


def extract_answer(text):
    """Premier mot/nombre de la réponse, normalisé : c'est la seule chose comparée (AgentProv compare le choix,
    pas la formulation)."""
    t = text.strip().lower()
    if t.startswith(LORA_TRAIN_PREFIX):
        t = t[len(LORA_TRAIN_PREFIX):].strip()
    for sep in [".", ",", "!", "\n", " "]:
        if sep in t:
            t = t.split(sep)[0]
    return t.strip(" .!?:;\"'")


# ---------------------------------------------------------------- chargement, génération, déguisements
def load(model_id, dtype):
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map={"": 0})
    m.eval()
    return m, tok


def chat(tok, prompt, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        # pas de rôle system supporté (rare) : on le préfixe au message utilisateur
        msgs = [{"role": "user", "content": (system + "\n\n" if system else "") + prompt}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


class KGW(LogitsProcessor):
    """Filigrane vert-liste minimal (Kirchenbauer et al.), clé fixe pour reproductibilité."""
    def __init__(self, vocab_size, gamma, delta, key, device):
        self.vocab_size, self.gamma, self.delta, self.key, self.device = vocab_size, gamma, delta, key, device

    def __call__(self, input_ids, scores):
        for b in range(scores.shape[0]):
            seed = (self.key * 1_000_003 + int(input_ids[b, -1])) % (2 ** 31)
            g = torch.Generator(device="cpu").manual_seed(seed)
            perm = torch.randperm(self.vocab_size, generator=g)
            greenlist = perm[: int(self.gamma * self.vocab_size)].to(scores.device)
            scores[b, greenlist] += self.delta
        return scores


@torch.no_grad()
def generate(model, tok, prompts, bs, max_new, system=None, do_sample=False, temperature=1.0, seed=0,
             wm_key=None, delta=0.0, gamma=0.25):
    texts = [chat(tok, p, system) for p in prompts]
    lp = None
    if wm_key is not None and delta > 0:
        lp = LogitsProcessorList([KGW(model.get_output_embeddings().weight.shape[0], gamma, delta, wm_key, model.device)])
    torch.manual_seed(seed)
    outs = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        kw = dict(do_sample=True, temperature=temperature, top_p=1.0, top_k=0) if do_sample else dict(do_sample=False)
        gen = model.generate(**enc, **kw, max_new_tokens=max_new, pad_token_id=tok.pad_token_id, logits_processor=lp)
        for row in gen[:, enc["input_ids"].shape[1]:]:
            outs.append(tok.decode(row, skip_special_tokens=True))
    return outs


@torch.no_grad()
def read_nll_per_char(model, tok, prompts, texts, bs, system=None):
    """NLL totale / nb caractères, forward enseigné sans biais (sert à la NLL croisée)."""
    out = []
    for i in range(0, len(prompts), bs):
        rows, plens, nchars = [], [], []
        for p, t in zip(prompts[i:i + bs], texts[i:i + bs]):
            pid = tok(chat(tok, p, system), add_special_tokens=False)["input_ids"]
            gid = tok(t, add_special_tokens=False)["input_ids"] if t else []
            rows.append(pid + gid); plens.append(len(pid)); nchars.append(max(1, len(t)))
        L = max(len(r) for r in rows)
        ids = torch.full((len(rows), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(rows), L), dtype=torch.long)
        for j, r in enumerate(rows):
            ids[j, :len(r)] = torch.tensor(r); att[j, :len(r)] = 1
        ids, att = ids.to(model.device), att.to(model.device)
        logp = torch.log_softmax(model(input_ids=ids, attention_mask=att).logits.float(), dim=-1)
        for j, r in enumerate(rows):
            n_gen = len(r) - plens[j]
            if n_gen == 0:
                out.append(float("nan")); continue
            tgt = ids[j, plens[j]:len(r)]
            lp = logp[j, plens[j] - 1:len(r) - 1].gather(-1, tgt[:, None]).squeeze(-1)
            out.append(-lp.sum().item() / nchars[j])
    return out


def cross_nll_gap(mA, tA, mB, tB, prompts, gA, gB, bs):
    """gap = NLL_B(texte_A) - NLL_A(texte_A), symétrisé ; ~0 si mêmes poids, croît sinon (watermark_robust.py)."""
    aa = read_nll_per_char(mA, tA, prompts, gA, bs)
    ba = read_nll_per_char(mB, tB, prompts, gA, bs)
    bb = read_nll_per_char(mB, tB, prompts, gB, bs)
    ab = read_nll_per_char(mA, tA, prompts, gB, bs)
    gaps = [y - x for x, y in zip(aa, ba) if x == x and y == y] + [y - x for x, y in zip(bb, ab) if x == x and y == y]
    return gaps


def train_lora(tok, dtype, base_model, prompts, strength, steps=120, lr=3e-4, bs=4):
    from peft import LoraConfig, get_peft_model
    train_texts = []
    base_out = generate(base_model, tok, prompts, bs, 40)
    for p, o in zip(prompts, base_out):
        train_texts.append((chat(tok, p), "Sure thing! " + o + tok.eos_token))
    names = {n.split(".")[-1] for n, _ in base_model.named_modules()}
    targets = ["q_proj", "v_proj"] if "q_proj" in names else ["qkv_proj"]  # Phi-3.5 : attention fusionnée
    cfg = LoraConfig(r=8, lora_alpha=16, target_modules=targets, lora_dropout=0.0, task_type="CAUSAL_LM")
    pm = get_peft_model(base_model, cfg)
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=lr)
    pm.train()
    rnd = random.Random(1)
    loss = None
    for step in range(steps):
        batch = rnd.sample(train_texts, min(bs, len(train_texts)))
        enc = tok([a + b for a, b in batch], return_tensors="pt", padding=True, add_special_tokens=False).to(pm.device)
        labels = enc["input_ids"].clone()
        for j, (a, _) in enumerate(batch):
            labels[j, :len(tok(a, add_special_tokens=False)["input_ids"])] = -100
        labels[enc["attention_mask"] == 0] = -100
        with torch.autocast("cuda", dtype=dtype):
            loss = pm(**enc, labels=labels).loss
        loss.backward(); opt.step(); opt.zero_grad()
    pm.eval()
    for mod in pm.modules():
        if hasattr(mod, "scaling") and isinstance(mod.scaling, dict):
            for k in mod.scaling:
                mod.scaling[k] = strength * (16 / 8)  # alpha/r de base = 2.0 ; on fixe scaling = strength demandé
    return pm, loss.item() if loss is not None else None


def fake_fp8_(model):
    """Fausse quantification fp8 e4m3 (mantisse 3 bits) des Linear, comme watermark_robust.py / exp2."""
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and "lm_head" not in name:
            w = m.weight.data.float()
            scale = w.abs().amax().clamp(min=1e-8) / 448.0
            q = (w / scale)
            sign = q.sign()
            mant_bits = 3
            exp = torch.floor(torch.log2(q.abs().clamp(min=1e-10)))
            mant = q.abs() / (2 ** exp)
            mant = torch.round(mant * (2 ** mant_bits)) / (2 ** mant_bits)
            q = sign * mant * (2 ** exp)
            m.weight.data = (q * scale).to(m.weight.dtype)
    return model


# ---------------------------------------------------------------- construction des 9 faux-ref
REF_SPEC = [  # (id, kind, same_model_group)
    ("ref-01", "plain", "S"),
    ("ref-02", "fp8", "T1"),
    ("ref-03", "system_prompt", "S"),
    ("ref-04", "lora_0.7", "T3"),
    ("ref-05", "persona", "S"),
    ("ref-06", "watermark", "S"),
    ("ref-07", "lora_1.5", "T4"),
    ("ref-08", "temperature_1.0", "S"),
    ("ref-09", "lora_0.3", "T2"),
]


def build_refs(model_id, dtype, probes, n_fc, bs, max_new_fc, nll_prompts, max_new_nll, seed, serving_noise=False):
    """Génère, pour chaque faux-ref, les réponses aux sondes à choix forcé + le texte NLL (sur prompts ouverts).
    Charge une seule fois les poids de base ; les LoRA et la fausse fp8 sont construites sur une copie."""
    probe_texts = [p for _, p in probes]
    t0 = time.time()
    base_m, tok = load(model_id, dtype)
    print(f"[{time.time()-t0:.0f}s] base chargée ({model_id})", flush=True)

    out = {}

    def run(rid, kind, model, system=None, do_sample=False, temperature=1.0, wm_key=None, delta=0.0):
        def answers(batch, seed0):
            """glouton : 1 réponse par sonde ; échantillonné : 5 tirages, réponse majoritaire par sonde."""
            draws = [generate(model, tok, probe_texts, batch, max_new_fc, system=system, do_sample=do_sample,
                              temperature=temperature, seed=seed0 + r, wm_key=wm_key, delta=delta)
                     for r in range(5 if do_sample else 1)]
            res = []
            for k in range(len(probe_texts)):
                ans = [extract_answer(d[k]) for d in draws]
                res.append(max(set(ans), key=ans.count))
            return res, draws[0]

        fc_answers, fc = answers(bs, seed)
        # auto-consistance (PR #22) : le même endpoint rejoue les sondes avec une autre taille de lot (et d'autres
        # graines s'il échantillonne) -> bruit propre à CE serveur, mesuré sans connaître le modèle.
        fc_replay, _ = answers(7, seed + 100)
        nll_text = generate(model, tok, [p for _, p in nll_prompts], bs, max_new_nll, system=system, do_sample=False)
        out[rid] = {"kind": kind, "fc_answers": fc_answers, "fc_replay": fc_replay, "nll_text": nll_text,
                    "model": model, "tok": tok}
        print(f"[{time.time()-t0:.0f}s] {rid} ({kind}) prêt | ex. sonde: {probes[0][1]!r} -> {fc[0]!r}", flush=True)

    # Sans bruit de service, les 5 copies "S" partagent le même objet de poids : NLL croisée ≡ 0 par construction,
    # plancher irréaliste. --serving_noise les sert depuis des instances distinctes (bf16 / fp16 / fp32).
    inst16 = inst32 = base_m
    if serving_noise:
        inst16, _ = load(model_id, torch.float16)
        inst32, _ = load(model_id, torch.float32)
    run("ref-01", "plain", base_m)
    run("ref-03", "system_prompt", inst16, system=SYSTEM_PROMPT)
    run("ref-05", "persona", inst32, system=PERSONA_PROMPT)
    run("ref-06", "watermark", inst16, wm_key=1234, delta=6.0)
    run("ref-08", "temperature_1.0", inst32, do_sample=True, temperature=1.0)

    fp8_m, _ = load(model_id, dtype)
    fake_fp8_(fp8_m)
    run("ref-02", "fp8", fp8_m)
    del fp8_m; torch.cuda.empty_cache()

    for rid, strength in [("ref-09", 0.3), ("ref-04", 0.7), ("ref-07", 1.5)]:
        lora_m, _ = load(model_id, dtype)
        pm, final_loss = train_lora(tok, dtype, lora_m, [p for _, p in nll_prompts_for_training(probes)], strength)
        print(f"  LoRA {strength} entraînée, loss finale {final_loss}", flush=True)
        run(rid, f"lora_{strength}", pm)
        del pm, lora_m; torch.cuda.empty_cache()

    return out, base_m, tok


def nll_prompts_for_training(probes):
    return probes[: min(60, len(probes))]


# ---------------------------------------------------------------- distance et clustering
def forced_choice_distance(refs, a, b):
    fa, fb = refs[a]["fc_answers"], refs[b]["fc_answers"]
    agree = sum(x == y for x, y in zip(fa, fb)) / len(fa)
    return 1 - agree, agree


def nll_gap(refs, nll_prompts, bs, a, b):
    ma, ta = refs[a]["model"], refs[a].get("tok")
    mb, tb = refs[b]["model"], refs[b].get("tok")
    g = cross_nll_gap(ma, ta, mb, tb, [p for _, p in nll_prompts], refs[a]["nll_text"], refs[b]["nll_text"], bs)
    return sum(g) / len(g) if g else float("nan")


def nll_gaps(refs, nll_prompts, bs, a, b):
    return cross_nll_gap(refs[a]["model"], refs[a]["tok"], refs[b]["model"], refs[b]["tok"],
                         [p for _, p in nll_prompts], refs[a]["nll_text"], refs[b]["nll_text"], bs)


def cluster_same_model(refs, tau_fc, tau_nll, nll_prompts, bs, tok):
    ids = [rid for rid, _, _ in REF_SPEC]
    for rid in ids:
        refs[rid]["tok"] = tok
    same = {(a, b): False for a, b in itertools.combinations(ids, 2)}
    detail = {}
    for a, b in itertools.combinations(ids, 2):
        dist, agree = forced_choice_distance(refs, a, b)
        decided_same_fc = agree >= tau_fc
        gap = None
        decided_same = decided_same_fc
        if decided_same_fc:
            # garde-fou NLL : le fp8/LoRA faible peut passer les sondes à choix forcé (annoncé aveugle au 8 bits
            # sur l'issue #4) ; on vérifie la NLL croisée avant de fusionner.
            gap = nll_gap(refs, nll_prompts, bs, a, b)
            decided_same = abs(gap) <= tau_nll
        same[(a, b)] = decided_same
        detail[f"{a}_{b}"] = {"fc_agree": round(agree, 3), "nll_gap": (round(gap, 6) if gap is not None else None),
                               "decided_same": decided_same}

    # union-find simple à partir des paires "same"
    parent = {rid: rid for rid in ids}
    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x
    def union(x, y):
        parent[find(x)] = find(y)
    for (a, b), s in same.items():
        if s:
            union(a, b)
    clusters = {}
    for rid in ids:
        clusters.setdefault(find(rid), []).append(rid)
    return clusters, detail


def write_csv(clusters, ref_spec, path):
    label_for = {}
    for i, (root, members) in enumerate(clusters.items()):
        for m in members:
            label_for[m] = f"G{i}"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["endpoint", "family", "model"])
        for rid, _, _ in ref_spec:
            w.writerow([rid, "P", label_for[rid]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--n_fc", type=int, default=100)
    ap.add_argument("--max_new_fc", type=int, default=8)
    ap.add_argument("--n_nll", type=int, default=40)
    ap.add_argument("--max_new_nll", type=int, default=48)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau_fc", type=float, default=0.70)
    ap.add_argument("--tau_nll", type=float, default=0.01)
    ap.add_argument("--calibrate", action="store_true",
                     help="balaye une grille de tau_fc/tau_nll et rapporte le MCC pratique pour chaque, au lieu de décider.")
    ap.add_argument("--serving_noise", action="store_true",
                    help="copies 'même modèle' servies par des instances distinctes (bf16/fp16/fp32) : plancher NLL réaliste")
    ap.add_argument("--out", default="results_v4.json")
    ap.add_argument("--csv_out", default=None)
    a = ap.parse_args()
    dtype = getattr(torch, a.dtype)
    model_id = MODELS[a.model]
    t0 = time.time()
    print(f"GPU: {torch.cuda.get_device_name(0)} | modèle: {a.model} ({model_id})", flush=True)

    probes = build_forced_choice_probes(a.n_fc, seed=a.seed)
    nll_prompts_raw = build_open_prompts(a.n_nll, seed=a.seed + 1)
    nll_prompts = [("open", p) for p in nll_prompts_raw]

    refs, base_m, tok = build_refs(model_id, dtype, probes, a.n_fc, a.bs, a.max_new_fc, nll_prompts,
                                    a.max_new_nll, a.seed, serving_noise=a.serving_noise)

    out = {"model": a.model, "model_id": model_id, "config": vars(a), "pairwise": {}}

    if a.calibrate:
        # collecte seule : aucune recherche de seuil ici, aucune lecture des labels. La décision (procédure
        # adaptative unique, PR #22) est faite hors ligne par analyse_v4_adaptatif.py.
        pair_stats = {}
        for a_id, b_id in itertools.combinations([r for r, _, _ in REF_SPEC], 2):
            _, agree = forced_choice_distance(refs, a_id, b_id)
            gaps = nll_gaps(refs, nll_prompts, a.bs, a_id, b_id)
            pair_stats[f"{a_id}_{b_id}"] = {"fc_agree": round(agree, 4),
                                            "nll_gap": round(sum(gaps) / len(gaps), 6) if gaps else None,
                                            "nll_gaps": [round(g, 6) for g in gaps]}
        out["pair_stats"] = pair_stats
        out["probe_cats"] = [c for c, _ in probes]
        out["fc_answers"] = {rid: refs[rid]["fc_answers"] for rid, _, _ in REF_SPEC}
        out["fc_replay"] = {rid: refs[rid]["fc_replay"] for rid, _, _ in REF_SPEC}
        print("collecte terminée :", len(pair_stats), "paires", flush=True)
    else:
        clusters, detail = cluster_same_model(refs, a.tau_fc, a.tau_nll, nll_prompts, a.bs, tok)
        out["pairwise"] = detail
        out["clusters"] = {k: v for k, v in clusters.items()}
        csv_path = a.csv_out or f"practice_v4_{a.model}.csv"
        write_csv(clusters, REF_SPEC, csv_path)
        out["csv_out"] = csv_path
        print("Clusters:", clusters, flush=True)
        print("CSV écrit:", csv_path, flush=True)

    json.dump(out, open(a.out, "w"), indent=1)
    print(f"[{time.time()-t0:.0f}s] FIN -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
