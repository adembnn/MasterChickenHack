"""Fabrique les listes de configs des balayages. usage : mkcfg.py   (ecrit cfg_sweep.json, cfg_sens.json)"""
import json
base = {"method": "fit", "mu": "gold"}          # balayages faits avec les vrais moments, pour isoler l'effet des reglages
sw = []
def add(tag, **kw): c = dict(base); c.update(kw); c["name"] = tag; sw.append(c)
for lr in [3e-5, 1e-4, 3e-4]: add(f"lr {lr:g}", lr=lr)
for nb in [0, 1, 2, 4]: add(f"couches : {nb} blocs + tete", nblk=nb)
for st in [200, 1500]: add(f"duree {st} pas", steps=st)
for lam in [0, 3, 30, 100]: add(f"ancre lam={lam}", lam=lam)
add("sans perte de moments", wm=0); add("T=2 seul (w1=0)", w1=0.0); add("T=1 seul (w2=0)", w2=0.0)
for sv in [0.0, 0.4]: add(f"severite copies {sv}", sevc=sv)
for z in ["z_bc", "z_tta", "z_inst", "z_oof", "z4", "z_rand"]: add(f"difficulte {z}", z=z)
for w in ["w_head", "w_oof", "w_tta", "w_unif", "w_mix3"]: add(f"mauvaise classe {w}", w=w)
for p, m in [(0.5, 0.0), (0.7, 0.15), (0.35, 0.15), (0.5, 0.3)]: add(f"forme T=2 puissance {p} melange {m}", q2pow=p, q2mix=m)
json.dump(sw, open('cfg_sweep.json', 'w'), indent=0)
se = []
for A in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]: se.append({"name": f"S1 A suppose {A}", "method": "fit", "A": A})
json.dump(se, open('cfg_sens.json', 'w'), indent=0); print(len(sw), len(se))
