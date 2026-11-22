"""Lecteur a temperature libre, calibre par une loi nulle simulee (aucune requete).
Pour chaque endpoint (reel ou faux endpoint « copie » tire du jumeau local) et chaque modele candidat m, on calcule la
vraisemblance profilee l(e, m) = max_T somme log softmax(logits_m / T)[token rendu] : la temperature est un parametre libre.
Candidats : base ; fp8 par tenseur (code de Titouan) ; base + s * Delta_d ou Delta_d est une LoRA apprise sur l'endpoint d
avec un facteur 1/tau appris en meme temps (la LoRA n'a pas a apprendre la temperature) et s une dose dans --scales.
Sorties (json) : vecteurs l(T) par endpoint pour chaque candidat ; l'analyse (p-valeurs = rang parmi les faux endpoints) est
faite par gl_analyse.py.
  base[e], fp8[schema][e]          : tous les endpoints reels + tous les faux
  dir[d][s][t]                      : LoRA apprise sur TOUTES les reponses de d, lue sur t != d (reels + kdir faux par T)
  self[e][s]                        : LoRA apprise sur une moitie des questions de e, lue sur l'autre moitie (2 plis),
                                      pour les reels et kself faux par T
"""
import os, json, math, argparse, collections, hashlib, time
os.environ.setdefault("HF_HOME", "/home/adem/eac/t2/hf")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--null", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--eps", required=True)
ap.add_argument("--skip", default="P000,P001")
ap.add_argument("--scales", default="0.3,0.6,1.0,1.5")
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--r", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--kdir", type=int, default=25)
ap.add_argument("--kself", type=int, default=8)
ap.add_argument("--fp8", default="e4m3_tensor")
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--stages", default="fp8,base,dir,self")
ap.add_argument("--out", required=True)
a = ap.parse_args()
torch.manual_seed(0)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(a.model)
tok.padding_side = "right"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
eps = a.eps.split(",")
SC = [float(x) for x in a.scales.split(",")]
TS = [round(0.55 + 0.05 * i, 2) for i in range(14)]
skip = a.skip.split(",")


def load(path, keep):
    by = collections.defaultdict(list)
    for l in open(path, encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        if r.get("type", "answer") != "answer" or not r.get("response") or r.get("probe_id") in skip or not keep(r["endpoint"]):
            continue
        by[r["endpoint"]].append((r["prompt"], r["response"]))
    return by


real = load(a.data, lambda e: e in eps)
null = load(a.null, lambda e: True)
fk_all = sorted(null)
fk_dir = [e for e in fk_all if int(e.split("-")[-1]) < a.kdir]
fk_self = [e for e in fk_all if int(e.split("-")[-1]) < a.kself]
CT = {}


def ctx(p):
    if p not in CT:
        CT[p] = tok(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                            date_string="21 Nov 2026"), add_special_tokens=False)["input_ids"]
    return CT[p]


ENC, PR = {}, {}
for src in (real, null):
    for e, items in src.items():
        ENC[e], PR[e] = [], []
        for p, r in items:
            y = tok(r, add_special_tokens=False)["input_ids"][:200]
            if len(y) >= 1:
                ENC[e].append((ctx(p), y))
                PR[e].append(p)
NTOK = {e: sum(len(y) for _, y in ENC[e]) for e in ENC}
fold = lambda p: int(hashlib.md5(p.encode()).hexdigest(), 16) % 2
Tg = torch.tensor(TS, device="cuda", dtype=torch.float32)


def batch(seqs):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for j, s in enumerate(seqs):
        ids[j, :len(s)] = torch.tensor(s)
        att[j, :len(s)] = 1
    return ids.cuda(), att.cuda()


def profile(model, sets):
    """sets : dict nom -> liste de (ctx, y). Rend dict nom -> liste [l(T) pour T dans TS] (somme des log-vraisemblances)."""
    items = [(n, i) for n in sets for i in range(len(sets[n]))]
    items.sort(key=lambda t: len(sets[t[0]][t[1]][0]) + len(sets[t[0]][t[1]][1]))
    acc = {n: torch.zeros(len(TS), dtype=torch.float64) for n in sets}
    with torch.no_grad():
        for b in range(0, len(items), a.bs):
            ch = items[b:b + a.bs]
            ids, att = batch([sets[n][i][0] + sets[n][i][1] for n, i in ch])
            logits = model(input_ids=ids, attention_mask=att).logits
            for j, (n, i) in enumerate(ch):
                c, y = sets[n][i]
                lg = logits[j, len(c) - 1:len(c) + len(y) - 1].float()
                tg = ids[j, len(c):len(c) + len(y)]
                z = lg[None] / Tg[:, None, None]
                ll = (z.gather(-1, tg[None, :, None].expand(len(TS), -1, 1))[..., 0] - torch.logsumexp(z, -1)).sum(1)
                acc[n] += ll.double().cpu()
                del z, ll, lg
            del logits
            if (b // a.bs) % 10 == 0:
                torch.cuda.empty_cache()
    return {n: [float(x) for x in v] for n, v in acc.items()}


def train(base, items, T0, seed):
    cfg = LoraConfig(r=a.r, lora_alpha=2 * a.r, lora_dropout=0.0, task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    pm = get_peft_model(base, cfg)
    la = torch.nn.Parameter(torch.tensor(-math.log(T0), device="cuda"))
    opt = torch.optim.AdamW([{"params": [p for p in pm.parameters() if p.requires_grad], "lr": a.lr},
                             {"params": [la], "lr": 0.02}])
    pm.train()
    g = torch.Generator().manual_seed(seed)
    for _ in range(a.epochs):
        perm = torch.randperm(len(items), generator=g).tolist()
        for i in range(0, len(items), 4):
            ch = [items[k] for k in perm[i:i + 4]]
            ids, att = batch([c + y for c, y in ch])
            logits = pm(input_ids=ids, attention_mask=att).logits.float() * la.exp()
            loss, n = 0.0, 0
            for j, (c, y) in enumerate(ch):
                loss = loss + torch.nn.functional.cross_entropy(logits[j, len(c) - 1:len(c) + len(y) - 1],
                                                                 ids[j, len(c):len(c) + len(y)], reduction="sum")
                n += len(y)
            (loss / n).backward()
            opt.step()
            opt.zero_grad()
    pm.eval()
    return pm, float((-la).exp())


def set_scale(pm, s):
    for mod in pm.modules():
        sc = getattr(mod, "scaling", None)
        if isinstance(sc, dict) and "default" in sc:
            sc["default"] = 2.0 * s


R = {"TS": TS, "scales": SC, "ntok": NTOK, "model": a.model, "eps": eps, "fp8": {}, "base": {}, "dir": {}, "self": {}, "tau": {}}
if os.path.exists(a.out):
    R.update(json.load(open(a.out)))
    R["ntok"] = NTOK


def save():
    json.dump(R, open(a.out + ".tmp", "w"))
    os.replace(a.out + ".tmp", a.out)


allsets = {e: ENC[e] for e in eps + fk_all}
stages = a.stages.split(",")
if "fp8" in stages and a.fp8:
    import fp8_llr as F
    from fp8_bank import make
    F.MODEL = a.model
    for sc in a.fp8.split(","):
        if sc in R["fp8"]:
            continue
        mq = make(sc)
        mq.eval()
        R["fp8"][sc] = profile(mq, allsets)
        del mq
        torch.cuda.empty_cache()
        save()
        print(f"[fp8 {sc}] fait {time.time() - t0:.0f}s", flush=True)

base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0}).eval()
if "base" in stages and not R["base"]:
    R["base"] = profile(base, allsets)
    save()
    print(f"[base] fait {time.time() - t0:.0f}s", flush=True)
Tstar = {e: TS[max(range(len(TS)), key=lambda i: R["base"][e][i])] for e in R["base"]}
print("T* :", " ".join(f"{e[-2:]}={Tstar[e]}" for e in eps), flush=True)

if "dir" in stages:
    for d in eps:
        if d in R["dir"]:
            continue
        pm, tau = train(base, ENC[d], Tstar[d], 0)
        R["tau"][d] = tau
        R["dir"][d] = {}
        tgt = {t: ENC[t] for t in eps + fk_dir if t != d}
        for s in SC:
            set_scale(pm, s)
            R["dir"][d][str(s)] = profile(pm, tgt)
        base = pm.unload()
        del pm
        torch.cuda.empty_cache()
        save()
        print(f"[dir {d}] tau={tau:.3f} fait {time.time() - t0:.0f}s", flush=True)

if "self" in stages:
    for e in eps + fk_self:
        if e in R["self"]:
            continue
        acc = {str(s): [0.0] * len(TS) for s in SC}
        for f in (0, 1):
            tr = [it for it, p in zip(ENC[e], PR[e]) if fold(p) != f]
            te = [it for it, p in zip(ENC[e], PR[e]) if fold(p) == f]
            if not tr or not te:
                continue
            pm, tau = train(base, tr, Tstar[e], f)
            for s in SC:
                set_scale(pm, s)
                v = profile(pm, {"x": te})["x"]
                acc[str(s)] = [x + y for x, y in zip(acc[str(s)], v)]
            base = pm.unload()
            del pm
            torch.cuda.empty_cache()
        R["self"][e] = acc
        if e in eps or len(R["self"]) % 8 == 0:
            save()
            print(f"[self {e}] fait {time.time() - t0:.0f}s", flush=True)
    save()
print("FINI", flush=True)
