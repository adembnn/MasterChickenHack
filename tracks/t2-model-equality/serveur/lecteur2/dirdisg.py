"""Directions LoRA contre la loi nulle ELARGIE (session t2-lecteur, aucune requete).
Une direction apprise sur le guichet source d est lue sur des cibles reelles ET sur des copies deguisees du jumeau
(prompt systeme, persona, decodage, filigrane ; fichier dgn_*.jsonl de mkdgn.py). Si une copie deguisee (par exemple a
prompt systeme) gagne autant que la cible, le lien direction -> cible s'explique par le deguisement, pas par le modele.
Rend, par cible, le z du gain contre les copies deguisees, et le gain maximal atteint par chaque deguisement.
Usage : dirdisg.py <arguments de gl_reader2.py, --null dgn_F3.jsonl> --pairs ep-16:ep-29,ep-05:ep-11 --map dgn_F3.jsonl.map.json"""
import sys, json, statistics
pairs = [p.split(":") for p in sys.argv[sys.argv.index("--pairs") + 1].split(",")]
mp = json.load(open(sys.argv[sys.argv.index("--map") + 1]))
i = sys.argv.index("--pairs")
del sys.argv[i:i + 2]
i = sys.argv.index("--map")
del sys.argv[i:i + 2]
exec(open("gl_reader2.py", encoding="utf-8").read().split('R = {"TS": TS')[0])
base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0}).eval()
fks = sorted(e for e in ENC if e.startswith("fk-"))
srcs = sorted({s for s, t in pairs})
tg_all = sorted(set(eps) | set(fks))
B = profile(base, {e: ENC[e] for e in tg_all})
Tst = {e: TS[max(range(len(TS)), key=lambda k: B[e][k])] for e in tg_all}
out = {}
for d in srcs:
    pm, tau = train(base, ENC[d], Tst[d], 0)
    gains = {}
    for s in SC:
        set_scale(pm, s)
        P = profile(pm, {e: ENC[e] for e in tg_all if e != d})
        for e, v in P.items():
            g = (max(v) - max(B[e])) / NTOK[e]
            gains[e] = max(gains.get(e, -1e9), g)
    base = pm.unload()
    del pm
    torch.cuda.empty_cache()
    vs = [gains[f] for f in fks]
    med = statistics.median(vs)
    mad = statistics.median(abs(v - med) for v in vs) * 1.4826 or 1e-9
    z = {e: (gains[e] - med) / mad for e in gains}
    byd = {}
    for f in fks:
        byd.setdefault(mp.get(f, "?"), []).append(z[f])
    print(f"\n== direction apprise sur {d} (tau {tau:.2f}) ; z contre {len(fks)} copies deguisees")
    print("  pire copie par deguisement : " + " ".join(f"{k}:{max(v):+.1f}" for k, v in sorted(byd.items())))
    print("  cibles reelles : " + " ".join(f"{e[-2:]}:{z[e]:+.1f}" for e in eps if e != d))
    for s_, t in pairs:
        if s_ == d:
            p = (1 + sum(z[f] >= z[t] for f in fks)) / (1 + len(fks))
            print(f"  LIEN {d} -> {t} : z {z[t]:+.1f}, p {p:.3f} (rang parmi les copies deguisees)")
    out[d] = {"z": z, "byd": {k: max(v) for k, v in byd.items()}}
json.dump(out, open(a.out, "w"))
print("FINI")
