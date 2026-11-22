"""t2-oracle : vraisemblance de chaque hypothese sous le masque « guichets publics S » (a partir des masses exactes)."""
import json, math
L = "E:/EAC-lab/t2/reel/oracle/"
nS = sum(math.comb(33, k) for k in range(14, 23))
H = {"k3": math.comb(11, 3) ** 3, "k4": math.comb(11, 4) ** 3, "k5": math.comb(11, 5) ** 3, "k6": math.comb(11, 6) ** 3, "U": 2 ** 33,
     "c4": (math.comb(11, 4) * 6) ** 3}
print("hypothese   P(12 scores | H, S de taille 14-22 tous egaux)   P(... | taille uniforme puis S uniforme)")
for h, size in H.items():
    try: Ze = json.load(open(L + f"exact_{h}_egale.json"))["Z"]
    except Exception: continue
    try: Zu = json.load(open(L + f"exact_{h}_uniforme.json"))["Z"]
    except Exception: Zu = float("nan")
    print(f"{h:6s}  {Ze / (nS * size):.3e}   {Zu / (9 * size):.3e}")
