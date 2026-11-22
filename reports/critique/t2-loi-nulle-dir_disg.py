"""Les directions LoRA apprises sur des endpoints reels expliquent-elles aussi des copies deguisees (prompt systeme cache) ?
Si oui, « explique par la direction d'un autre endpoint » ne prouve pas une LoRA. Aucune requete.
Usage : dir_disg.py --tag F3 --dirs ep-05,ep-09 + les arguments de gl_reader.py (--data --null --model --eps --out)"""
import sys, json, statistics, collections
argv = sys.argv[1:]
tag = argv[argv.index("--tag") + 1]; dirs = argv[argv.index("--dirs") + 1].split(",")
GROUPS = None
if "--groups" in argv:
    i = argv.index("--groups"); GROUPS = dict(x.split("=") for x in argv[i + 1].split(",")); del argv[i:i + 2]
for k in ("--tag", "--dirs"):
    i = argv.index(k); del argv[i:i + 2]
sys.argv = ["gl_reader.py"] + argv
exec(open("gl_reader.py", encoding="utf-8").read().split('R = {"TS": TS')[0])
P = json.load(open("disg_prompts.json", encoding="utf-8"))
if GROUPS:
    P = GROUPS
groups = collections.defaultdict(list)
for name in P:
    for e, items in load(GROUPS[name] if GROUPS else f"null_{tag}_{name}.jsonl", lambda e: True).items():
        n = f"dz-{name}-{e}"
        ENC[n] = [(ctx(p), y) for p, r in items for y in [tok(r, add_special_tokens=False)["input_ids"][:200]] if y]
        groups[name].append(n)
dz = [n for g in groups.values() for n in g]
base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0}).eval()
tg_all = {t: ENC[t] for t in eps + fk_dir + dz}
B = profile(base, tg_all)
L0 = {t: max(v) for t, v in B.items()}
Tst = {t: TS[max(range(len(TS)), key=lambda i: B[t][i])] for t in B}
for t in fk_dir: Tst[t] = float(t.split("-")[1])
nt = {t: sum(len(y) for _, y in ENC[t]) for t in tg_all}
def z_of(lam, x):
    b = [f for f in fk_dir if abs(Tst[f] - Tst[x]) <= 0.1001] or fk_dir
    vs = [lam[f] for f in b]; med = statistics.median(vs); mad = statistics.median(abs(v - med) for v in vs) * 1.4826 or 1e-9
    return (lam[x] - med) / mad
print("T* lue des jumeaux du groupe (moyenne) :", " ".join(f"{g}={statistics.mean(Tst[n] for n in ns):.2f}" for g, ns in groups.items()), flush=True)
for d in dirs:
    pm, tau = train(base, ENC[d], Tst[d], 0)
    best = {t: -1e9 for t in tg_all}
    for s in SC:
        set_scale(pm, s)
        pr = profile(pm, {t: v for t, v in tg_all.items() if t != d})
        for t, v in pr.items(): best[t] = max(best[t], (max(v) - L0[t]) / nt[t])
    base = pm.unload(); del pm; torch.cuda.empty_cache()
    print(f"== direction {d} : z des endpoints reels :", " ".join(f"{t[-2:]}={z_of(best, t):+.1f}" for t in eps if t != d), flush=True)
    print("   z des jumeaux du groupe (mediane / max) :", " ".join(f"{g}={statistics.median(z_of(best, n) for n in ns if n != d):+.1f}/{max(z_of(best, n) for n in ns if n != d):+.1f}" for g, ns in groups.items()), flush=True)
print("FINI")
