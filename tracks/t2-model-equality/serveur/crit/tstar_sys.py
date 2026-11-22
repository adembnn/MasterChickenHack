"""T* et NLL par token de chaque endpoint relu sous son meilleur prompt systeme candidat : une copie a prompt cache doit revenir
a la temperature de la famille (~0,70) ; une variante de poids garde son ecart."""
import json, sys
for fam in sys.argv[1:]:
    G = json.load(open(f"gl_{fam}.json")); TS = G["TS"]; n = G["ntok"]
    S = {}
    for f in (f"sys_{fam}.json", f"sys2_{fam}.json"):
        try: S.update(json.load(open(f))["sys"])
        except Exception as e: pass
    def tm(v):
        i = max(range(len(v)), key=lambda k: v[k])
        if 0 < i < len(v) - 1:
            x, y, z = v[i - 1], v[i], v[i + 1]; d = x - 2 * y + z
            if d < 0: return TS[i] + 0.5 * (x - z) / d * (TS[1] - TS[0]), y - 0.125 * (x - z) ** 2 / d
        return TS[i], v[i]
    print(f"== {fam} : T* et NLL/token sous le gabarit par defaut, puis sous le meilleur prompt candidat")
    for e in G["eps"]:
        t0, l0 = tm(G["base"][e])
        best = max(S, key=lambda k: tm(S[k][e])[1]) if S else None
        t1, l1 = tm(S[best][e]) if best else (t0, l0)
        print(f"  {e}  defaut T*={t0:.2f} nll={-l0 / n[e]:.3f} | meilleur prompt {best:<14} T*={t1:.2f} nll={-l1 / n[e]:.3f} gain={(l1 - l0) / n[e]:+.3f}")
