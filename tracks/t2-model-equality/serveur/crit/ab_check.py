"""Controle demande par l'agent de Yassir : la direction apprise sur un guichet source avec les questions d'un bloc est lue sur les AUTRES
questions (autre bloc) de tous les guichets, et le z de chaque guichet est compare aux faux endpoints lus de la meme facon.
Deux sens (apprendre sur bloc 0, lire sur bloc 1 ; puis l'inverse). Usage : ab_check.py --src ep-17 + arguments de gl_reader.py"""
import sys, json, statistics
argv = sys.argv[1:]; i = argv.index("--src"); SRC = argv[i + 1].split(","); del argv[i:i + 2]
sys.argv = ["gl_reader.py"] + argv
exec(open("gl_reader.py", encoding="utf-8").read().split('R = {"TS": TS')[0])
B1 = {json.loads(l)["prompt"] for l in open("questions_bloc1.jsonl", encoding="utf-8") if l.strip()}
def half(e, h):
    return [it for it, p in zip(ENC[e], PR[e]) if (p in B1) == (h == 1)]
base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0}).eval()
names = eps + fk_dir
Tf = lambda e: float(e.split("-")[1])
for src in SRC:
    tot = {t: 0.0 for t in names}
    for h in (0, 1):
        sets = {t: half(t, 1 - h) for t in names if half(t, 1 - h)}
        B = profile(base, sets); L0 = {t: max(v) for t, v in B.items()}
        Tst = {t: (TS[max(range(len(TS)), key=lambda k: B[t][k])] if not t.startswith("fk-") else Tf(t)) for t in sets}
        nt = {t: sum(len(y) for _, y in sets[t]) for t in sets}
        pm, tau = train(base, half(src, h), Tst.get(src, 0.7), h)
        best = {t: -1e9 for t in sets}
        for s in SC:
            set_scale(pm, s); pr = profile(pm, {t: v for t, v in sets.items() if t != src})
            for t, v in pr.items(): best[t] = max(best[t], (max(v) - L0[t]) / nt[t])
        base = pm.unload(); del pm; torch.cuda.empty_cache()
        fk = [f for f in fk_dir if f in sets]
        def z(x):
            b = [f for f in fk if abs(Tst[f] - Tst[x]) <= 0.1001] or fk; vs = [best[f] for f in b]
            med = statistics.median(vs); mad = statistics.median(abs(v - med) for v in vs) * 1.4826 or 1e-9; return (best[x] - med) / mad
        zz = {t: z(t) for t in sets if t != src}
        for t in zz: tot[t] += zz[t]
        mx = max(zz[f] for f in fk)
        print(f"[{src}] appris sur bloc {h}, lu sur bloc {1 - h} (max des faux endpoints {mx:+.1f}) : " + " ".join(f"{t[-2:]}={zz[t]:+.1f}" for t in eps if t != src), flush=True)
    fkz = sorted(tot[f] / 2 ** 0.5 for f in fk_dir)
    print(f"[{src}] somme des deux sens / racine(2) (faux endpoints : 95 % {fkz[int(.95 * len(fkz))]:+.1f}, max {fkz[-1]:+.1f}) : " + " ".join(f"{t[-2:]}={tot[t] / 2 ** 0.5:+.1f}" for t in eps if t != src), flush=True)
print("FINI")
