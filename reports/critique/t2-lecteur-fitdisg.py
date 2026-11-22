"""Lecteur a deguisement ajuste (session t2-lecteur, aucune requete).
Pour chaque guichet reel et chaque faux endpoint (copies simples et copies deguisees de gen_disg2.py) :
 A. contexte : vraisemblance profilee en T sous chaque prompt systeme candidat (aucun, S1, S2, avocat, romance, claude ; acme reste inconnu du lecteur : temoin de persona non devinee) ;
    on garde le meilleur.
 B. decodage, sous le meilleur contexte : grille penalite de repetition (1,0 / 1,05 / 1,1 / 1,3) x temperature (0,5 a 1,2) x troncature
    (aucune, top-k 20 / 40 / 50, top-p 0,8 / 0,9 / 0,95, min-p 0,05 / 0,1, top-k 50 + top-p 0,9, top-k 20 + top-p 0,8 = reglage par defaut de Qwen2.5). Un token hors de l'ensemble
    tronque garde une probabilite plancher eps * p (eps = 1e-6). Pour chaque reglage : somme des log q(y), et tests de
    calibration que l'ajustement de T n'absorbe pas : queue (tokens a log q < -4), milieu (-4 a -1,6), tete (token le plus
    probable), observe contre attendu sous q.
 C. quantifications simulees (quantlib.py) et fp16, sous le meilleur contexte : vraisemblance profilee en T ; LLR = gain sur
    la base bf16.
Sortie json par endpoint. Usage : fitdisg.py --data data50_F1.jsonl --null null50_F1.jsonl,dg_F1.jsonl --model ... --eps ... --out fd_F1.json"""
import os, json, argparse, collections, time, math
os.environ.setdefault("HF_HOME", "/home/adem/eac/t2/lecteur2/hf")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

SYS = {
    "none": None,
    "S1": "Help the user with their questions and tasks. Use short paragraphs. Do not make up facts; say when you do not know.",
    "S2": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Use short paragraphs. Match the length of your answer to the question. Do not share personal data about private individuals.",
    "avocat": "Act as a lawyer and judicial advisor with 25 years of experience in drafting defense memoranda in Saudi courts only, with the condition of adhering to the legal provisions currently in force.",
    "romance": "You are a creative writing assistant who helps users craft romantic letters and poems, emulating the style and tone of human writers.",
    "claude": "You are Claude, an AI assistant made by Anthropic.",
}
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--null", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--eps", required=True)
ap.add_argument("--kplain", type=int, default=15, help="faux endpoints simples gardes par temperature")
ap.add_argument("--kdisg", type=int, default=6, help="faux endpoints deguises gardes par deguisement (indices k mod 10)")
ap.add_argument("--quant", default="nf4_b64,int8_ch,fp8_ch,int4_g128,fp16,fp8_t")
ap.add_argument("--stages", default="A,B,C")
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--out", required=True)
a = ap.parse_args()
t0 = time.time()
tok = AutoTokenizer.from_pretrained(a.model)
tok.padding_side = "right"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
eps = a.eps.split(",")
SKIP = ("P000", "P001")


def load(path, keep):
    by = collections.defaultdict(list)
    meta = {}
    for l in open(path, encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        if r.get("type", "answer") != "answer" or not r.get("response") or r.get("probe_id") in SKIP or not keep(r):
            continue
        e = r["endpoint"]
        if "disg" in r:
            e = "fk-" + r["disg"] + "-" + e.split("-")[-1]
        elif e.startswith("fk-"):
            e = "fk-plainT%s-%s" % (e.split("-")[1], e.split("-")[-1])
        by[e].append((r["prompt"], r["response"]))
    return by


DATA = load(a.data, lambda r: r["endpoint"] in eps)
for p in a.null.split(","):
    DATA.update(load(p, lambda r: (int(r["endpoint"].split("-")[-1]) % 10 < a.kdisg) if "disg" in r else (int(r["endpoint"].split("-")[-1]) < a.kplain)))
ALL = list(eps) + sorted(e for e in DATA if e.startswith("fk-"))
print(len(ALL), "endpoints dont", len(ALL) - len(eps), "faux", flush=True)
CT = {}


def ctx(p, s):
    if (p, s) not in CT:
        msgs = ([{"role": "system", "content": SYS[s]}] if SYS[s] else []) + [{"role": "user", "content": p}]
        CT[(p, s)] = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, date_string="21 Nov 2026"),
                         add_special_tokens=False)["input_ids"]
    return CT[(p, s)]


Y = {e: [tok(r, add_special_tokens=False)["input_ids"][:200] for p, r in DATA[e]] for e in ALL}
P = {e: [p for p, r in DATA[e]] for e in ALL}
NT = {e: sum(len(y) for y in Y[e]) for e in ALL}
TS = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0, 1.1, 1.2]
THS = [1.0, 1.05, 1.1, 1.3]
CFG = ["full", "k20", "k40", "k50", "p80", "p90", "p95", "mp05", "mp10", "k50p90", "k20p80"]
EPSF = 1e-6
TA = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
Tg = torch.tensor(TA, device="cuda")


def batches(model, items):
    """items : liste de (cle, ctx, y). Rend (cle, logits [n, V] float32 des positions qui predisent y)."""
    order = sorted(range(len(items)), key=lambda i: len(items[i][1]) + len(items[i][2]))
    with torch.no_grad():
        for b in range(0, len(order), a.bs):
            ch = [items[i] for i in order[b:b + a.bs]]
            L = max(len(c) + len(y) for _, c, y in ch)
            ids = torch.full((len(ch), L), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(ch), L), dtype=torch.long)
            for j, (_, c, y) in enumerate(ch):
                ids[j, :len(c) + len(y)] = torch.tensor(c + y)
                att[j, :len(c) + len(y)] = 1
            lg = model(input_ids=ids.cuda(), attention_mask=att.cuda()).logits
            for j, (k, c, y) in enumerate(ch):
                yield k, c, y, lg[j, len(c) - 1:len(c) + len(y) - 1].float()
            del lg


def profileT(model, sel):
    """sel : dict e -> nom de contexte. Rend e -> [l(T)]."""
    items = [((e, i), ctx(P[e][i], sel[e]), Y[e][i]) for e in sel for i in range(len(Y[e])) if Y[e][i]]
    acc = {e: torch.zeros(len(TA), dtype=torch.float64) for e in sel}
    for (e, i), c, y, lg in batches(model, items):
        tg = torch.tensor(y, device="cuda")
        z = lg[None] / Tg[:, None, None]
        acc[e] += (z.gather(-1, tg[None, :, None].expand(len(TA), -1, 1))[..., 0] - torch.logsumexp(z, -1)).sum(1).double().cpu()
    return {e: [float(x) for x in v] for e, v in acc.items()}


R = {"TS": TS, "TA": TA, "THS": THS, "CFG": CFG, "ntok": NT, "eps": eps, "ctxA": {}, "best_ctx": {}, "grid": {}, "quant": {}, "base": {}}
if os.path.exists(a.out):
    R.update(json.load(open(a.out)))


def save():
    json.dump(R, open(a.out + ".tmp", "w"))
    os.replace(a.out + ".tmp", a.out)


stages = a.stages.split(",")
base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0}).eval()
if "A" in stages and not R["ctxA"]:
    for s in SYS:
        prof = profileT(base, {e: s for e in ALL})
        for e in ALL:
            R["ctxA"].setdefault(e, {})[s] = prof[e]
        print(f"[A {s}] {time.time() - t0:.0f}s", flush=True)
    save()
for e in ALL:
    R["best_ctx"][e] = max(R["ctxA"][e], key=lambda s: max(R["ctxA"][e][s]))
SEL = R["best_ctx"]

if "B" in stages and not R["grid"]:
    NST = 11
    KTOP = 2048
    acc = {e: torch.zeros(len(THS), len(TS), len(CFG), NST, dtype=torch.float64) for e in ALL}
    items = [((e, i), ctx(P[e][i], SEL[e]), Y[e][i]) for e in ALL for i in range(len(Y[e])) if Y[e][i]]
    buf = []

    def flush(buf):
        """buf : liste de (e, c, y, logits) ; calcule les statistiques ligne par ligne, toutes sequences concatenees."""
        lg = torch.cat([b[3] for b in buf])
        N, V = lg.shape
        dev = lg.device
        tg = torch.cat([torch.tensor(b[2], device=dev) for b in buf])
        owner = sum([[b[0]] * len(b[2]) for b in buf], [])
        M = torch.zeros(N, V, dtype=torch.bool, device=dev)
        r0 = 0
        for e, c, y, _ in buf:
            n = len(y)
            M[r0:r0 + n, torch.tensor(sorted(set(c)), device=dev)] = True
            if n > 1:
                g = torch.zeros(n, V, device=dev)
                g[torch.arange(1, n, device=dev), tg[r0:r0 + n - 1]] = 1
                M[r0:r0 + n] |= torch.cumsum(g, 0) > 0
                del g
            r0 += n
        idx = torch.arange(KTOP, device=dev)[None]
        rows = torch.zeros(len(THS), len(TS), len(CFG), NST, N, device=dev)
        for a_, th in enumerate(THS):
            l2 = lg if th == 1.0 else torch.where(M, torch.where(lg > 0, lg / th, lg * th), lg)
            srt = l2.topk(KTOP, -1).values
            ly = l2.gather(1, tg[:, None])[:, 0]
            rank = (l2 > ly[:, None]).sum(1)
            inK = rank < KTOP
            for b_, T in enumerate(TS):
                lse = torch.logsumexp(l2 / T, 1, keepdim=True)
                lp = srt / T - lse
                p = lp.exp()
                cum = p.cumsum(1)
                cK = cum[:, -1]
                lpy = ly / T - lse[:, 0]
                py = lpy.exp()
                rk = rank.clamp(max=KTOP - 1)[:, None]
                for c_, cf in enumerate(CFG):
                    if cf == "full":
                        q, lq, lqy = p, lp, lpy
                        kty = torch.ones_like(inK)
                    else:
                        if cf[0] == "k" and "p" not in cf:
                            keep = (idx < int(cf[1:])).expand(N, -1)
                            kout = torch.zeros_like(inK)
                            extra = 0.0
                        elif cf[0] == "p":
                            P_ = int(cf[1:]) / 100
                            keep = (cum - p) < P_
                            kout = cK < P_
                            extra = (1 - cK) * kout
                        elif cf[:2] == "mp":
                            m_ = int(cf[2:]) / 100
                            keep = p >= m_ * p[:, :1]
                            kout = py >= m_ * p[:, 0]
                            extra = 0.0
                        else:
                            kk_, pp_ = int(cf[1:cf.index("p")]), int(cf[cf.index("p") + 1:]) / 100
                            pk = p * (idx < kk_)
                            pk = pk / pk.sum(1, keepdim=True)
                            keep = (idx < kk_) & ((pk.cumsum(1) - pk) < pp_)
                            kout = torch.zeros_like(inK)
                            extra = 0.0
                        mass = (p * keep).sum(1) + extra
                        q = (1 - EPSF) * p * keep / mass[:, None] + EPSF * p
                        lq = q.clamp_min(1e-30).log()
                        kty = torch.where(inK, keep.gather(1, rk)[:, 0], kout)
                        lqy = ((1 - EPSF) * kty * py / mass + EPSF * py).clamp_min(1e-30).log()
                    H = -(q * lq).sum(1)
                    Vr = (q * lq * lq).sum(1) - H * H
                    te = (q * (lq < -4.0)).sum(1)
                    me = (q * ((lq >= -4.0) & (lq < -1.6))).sum(1)
                    R_ = rows[a_, b_, c_]
                    R_[0] = lqy
                    R_[1] = H
                    R_[2] = Vr
                    R_[3] = (lqy < -4.0).float()
                    R_[4] = te
                    R_[5] = ((lqy >= -4.0) & (lqy < -1.6)).float()
                    R_[6] = me
                    R_[7] = ((rank == 0) & kty).float()
                    R_[8] = q[:, 0]
                    R_[9] = 1.0
                    R_[10] = te * (1 - te)
            del l2, srt
        r0 = 0
        for e, c, y, _ in buf:
            n = len(y)
            acc[e] += rows[..., r0:r0 + n].sum(-1).double().cpu()
            r0 += n
        del rows, M, lg

    nb = 0
    for done, ((e, i), c, y, lg) in enumerate(batches(base, items)):
        buf.append((e, c, y, lg))
        nb += len(y)
        if nb >= 800:
            flush(buf)
            buf, nb = [], 0
        if done % 1000 == 0:
            print(f"[B] {done}/{len(items)} {time.time() - t0:.0f}s", flush=True)
    if buf:
        flush(buf)
    R["grid"] = {e: acc[e].tolist() for e in ALL}
    save()
    print(f"[B] fait {time.time() - t0:.0f}s", flush=True)

if "C" in stages:
    if not R["base"]:
        R["base"] = profileT(base, SEL)
        save()
    from quantlib import fake_quant
    for sc in a.quant.split(","):
        if sc in R["quant"]:
            continue
        del base
        torch.cuda.empty_cache()
        if sc == "fp16":
            mq = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16, device_map={"": 0}).eval()
        else:
            mq = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0}).eval()
            fake_quant(mq, sc)
        R["quant"][sc] = profileT(mq, SEL)
        base = mq
        save()
        print(f"[C {sc}] {time.time() - t0:.0f}s", flush=True)
print("FINI", flush=True)
