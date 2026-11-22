"""NLL par token de chaque guichet comparee aux faux endpoints « copie » de la temperature la plus proche de sa T* (50 reponses).
Une copie a tirage pur, quelle que soit sa temperature, doit tomber dans la loi des faux endpoints de meme temperature."""
import sys, json, statistics
real, fk = {}, {}
for p in sys.argv[1:]:
    try: R = json.load(open(p))
    except Exception: continue
    TS = R["TS"]; n = R["ntok"]; d = TS[1] - TS[0]
    for e, v in R["base"].items():
        i = max(range(len(v)), key=lambda k: v[k]); T, l = TS[i], v[i]
        if 0 < i < len(v) - 1:
            x, y, z = v[i - 1], v[i], v[i + 1]; q = x - 2 * y + z
            if q < 0: s = 0.5 * (x - z) / q; T, l = TS[i] + s * d, y - 0.25 * (x - z) * s
        (fk if e.startswith("fk-") else real)[e] = (T, -l / n[e])
band = {}
for e, v in fk.items(): band.setdefault(e.split("-")[1], []).append(v)
st = {float(k): (statistics.mean(x[0] for x in v), statistics.pstdev(x[0] for x in v), statistics.mean(x[1] for x in v), statistics.pstdev(x[1] for x in v), len(v)) for k, v in band.items()}
print("bandes :", " ; ".join(f"T={k:.2f} : T* {m:.3f}+-{s:.3f}, NLL {a:.3f}+-{b:.3f} ({c})" for k, (m, s, a, b, c) in sorted(st.items())))
# interpolation lineaire de la NLL attendue en fonction de T*
ks = sorted(st)
def expect(T):
    if T <= st[ks[0]][0]: return st[ks[0]][2], st[ks[0]][3]
    for a, b in zip(ks, ks[1:]):
        Ta, Tb = st[a][0], st[b][0]
        if T <= Tb:
            w = (T - Ta) / (Tb - Ta); return st[a][2] + w * (st[b][2] - st[a][2]), st[a][3] + w * (st[b][3] - st[a][3])
    return st[ks[-1]][2], st[ks[-1]][3]
for e, (T, N) in sorted(real.items(), key=lambda kv: kv[1][0]):
    m, s = expect(T)
    print(f"  {e}  T*={T:.3f}  NLL/token={N:.3f}  attendu pour une copie a cette T* : {m:.3f}  z={(N - m) / s:+5.1f}")
