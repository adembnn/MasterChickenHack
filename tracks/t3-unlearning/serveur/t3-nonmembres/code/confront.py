"""Confrontation de mes estimations du gold aux scores réels du leaderboard public.
Pour chaque hypothèse de gold (moyenne et écart-type de la loss, sur images exactes et sur copies modifiées) et chaque candidat noté :
score de loss calculé avec la formule officielle sur les sorties du candidat (3 500 images, fp16), puis produit restant = score réel / score de loss
(= score de prédiction × rétention, forcément ≤ 1 ; la rétention mesurée sur mes images non-membres est donnée à côté)."""
import numpy as np, json
def lsm(z): z = z.astype(np.float64); z = z - z.max(-1, keepdims=True); return z - np.log(np.exp(z).sum(-1, keepdims=True))
F = np.load('forget_logits.npy'); y = F.argmax(1); I = np.arange(3500)
def ms(z): c = -lsm(z)[I, y]; return c.mean(), c.std()
REAL = {'s1180': 0.0039, 's1197': 0.0114, 'PG_T85': 0.0241, 's1265': 0.0273, 'PG_T70': 0.0323, 'l48c': 0.0735, 'l814c': 0.1487}
ALT = ['alt0', 'alt1', 'alt2', 'alt3']
O = np.load('sub_orig.npz'); mo, so = ms(O['propre']); moa, soa = np.mean([ms(O[a]) for a in ALT], 0)
# hypothèses : (exactes m, s), (copies m, s)
H = {'H1 départ 0,55/1,17': ((0.55, 1.17), (0.6, 1.3)), 'simulateur 1,36/2,49': ((1.36, 2.49), (1.43, 2.60)), 'i2i 0,70': ((0.96, 2.30), (1.18, 2.59)),
     'IP-Adapter': ((2.13, 3.49), (2.20, 3.50)), 'i2i 0,85': ((2.70, 4.17), (2.90, 4.25)), 'entre i2i70 et IPA': ((1.5, 2.9), (1.7, 3.05))}
ret = json.load(open('retproxy.json'))
print(f"{'candidat':8s} {'réel':>7s} {'exact m/s':>11s} {'copies m/s':>11s} {'rét. i2i70/ipa/t2i':>19s} | " + ' | '.join(f'{h:>21s}' for h in H))
print(' ' * 62 + '| ' + ' | '.join(f"{'S_loss   reste':>21s}" for h in H))
for c, sreal in REAL.items():
    S = np.load(f'sub_{c}.npz'); m, s = ms(S['propre']); ma, sa = np.mean([ms(S[a]) for a in ALT], 0); cols = []
    for h, ((gm, gs), (gma, gsa)) in H.items():
        le = 1 - min(1, (abs(m - gm) + abs(s - gs)) / (abs(mo - gm) + abs(so - gs))); la = 1 - min(1, (abs(ma - gma) + abs(sa - gsa)) / (abs(moa - gma) + abs(soa - gsa)))
        L = (le + la) / 2; cols.append(f"{L:8.2f} {sreal / L if L > 0 else float('inf'):11.3f}")
    r = ret.get(c, {}); rr = '/'.join(f"{r[k]['retention']:.2f}" for k in ('sdxl_i2i_70', 'sdxl_ipa_60', 'sdxl_t2i')) if r else ''
    print(f"{c:8s} {sreal:7.4f} {m:5.2f}/{s:5.2f} {ma:5.2f}/{sa:5.2f} {rr:>19s} | " + ' | '.join(cols))
print("\nLecture : « reste » = score réel / score de loss = prédiction × rétention. Une hypothèse est d'autant plus crédible que ce reste varie peu et de façon explicable d'un candidat à l'autre.")
