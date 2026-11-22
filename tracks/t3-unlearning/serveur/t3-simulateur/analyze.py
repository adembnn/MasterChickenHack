"""Simulateur T3 : (a) a quoi ressemble un gold sur l'oubli ; (b) peut-on predire depuis orig seul ce que le gold ratera ; outil d'estimation de la loss du gold.
usage : analyze.py WORLD ORIG GOLD1,GOLD2,...   (lit runs/WORLD/*.npz et prep.pt si present)"""
import sys, json, numpy as np, torch
import score as SC
from common import ROOT, DATA, World, labels, ALT_NAMES, NC
world, orig, golds = sys.argv[1], sys.argv[2], sys.argv[3].split(',')
R = f'{ROOT}/runs/{world}'; W = World(world); L = labels(W); y = L['fx']; O = dict(np.load(f'{R}/{orig}.npz')); G = [dict(np.load(f'{R}/{g}.npz')) for g in golds]
cls = json.load(open(DATA + '/classes.json'))['classes']; grp = np.array([c['group'] for c in cls]); nm = [c['name'].split(',')[0] for c in cls]
N = len(y); ar = np.arange(N)


def st(z, yy): l = SC.ce(z, yy); return f'acc {SC.acc(z, yy):.4f}  loss moy {l.mean():.4f}  sd {l.std():.4f}  maxprob {np.exp(SC.lsm(z)).max(1).mean():.4f}  maxprob(T=2) {np.exp(SC.lsm(z, 2)).max(1).mean():.4f}'


print(f'=== MONDE {world} : orig={orig} golds={golds} ===')
print('--- (0) ORIG : ressemble-t-il au vrai ? (vrai : maxprob 0.9997, loss 0.0003, maxprob T=2 0.985)')
print('orig oubli exact      ', st(O['fx'], y))
for t in range(4): print(f'orig copies {ALT_NAMES[t]:22s}', st(O['fa4'][t], y))
print('orig controle (jamais vu)', st(O['ctrl'], L['ctrl'])); print('orig validation          ', st(O['val'], L['val']))
print('\n--- (a) GOLD sur les images oubliees (jamais vues par lui)')
for n, g in zip(golds, G): print(f'{n:10s} exact ', st(g['fx'], y), '| controle', f"acc {SC.acc(g['ctrl'], L['ctrl']):.4f}")
mu, sd, lpg = SC.gold_ref([g['fx'] for g in G], y); pg = np.exp(lpg)
print(f'REFERENCE exact : mean(L_gold) {mu:.4f}  std(L_gold) {sd:.4f}  | p_gold(T=2, moyenne des golds) : p(vraie classe) moy {pg[ar, y].mean():.4f}  max moy {pg.max(1).mean():.4f}  argmax juste {(pg.argmax(1) == y).mean():.4f}')
for t in range(4):
    m2, s2, _ = SC.gold_ref([g['fa4'][t] for g in G], y); print(f'REFERENCE copies {ALT_NAMES[t]:22s}: mean {m2:.4f} std {s2:.4f} acc {np.mean([SC.acc(g["fa4"][t], y) for g in G]):.4f}')
m2, s2, _ = SC.gold_ref([SC.alt_mix(g) for g in G], y); print(f'REFERENCE copies (melange des 4)      : mean {m2:.4f} std {s2:.4f}')
# meme classes, images de controle : l'oubli est-il plus dur que du jamais-vu ordinaire ?
fc = np.unique(y); mc = np.isin(L['ctrl'], fc)
print(f'gold sur le controle, 45 classes de l oubli : acc {np.mean([SC.acc(g["ctrl"][mc], L["ctrl"][mc]) for g in G]):.4f} loss {np.mean([SC.ce(g["ctrl"][mc], L["ctrl"][mc]).mean() for g in G]):.4f} sd {np.mean([SC.ce(g["ctrl"][mc], L["ctrl"][mc]).std() for g in G]):.4f}'
      f' | 55 autres classes : acc {np.mean([SC.acc(g["ctrl"][~mc], L["ctrl"][~mc]) for g in G]):.4f} loss {np.mean([SC.ce(g["ctrl"][~mc], L["ctrl"][~mc]).mean() for g in G]):.4f}')
print(f'ORIG sur le controle, 45 classes de l oubli : acc {SC.acc(O["ctrl"][mc], L["ctrl"][mc]):.4f} loss {SC.ce(O["ctrl"][mc], L["ctrl"][mc]).mean():.4f} sd {SC.ce(O["ctrl"][mc], L["ctrl"][mc]).std():.4f}'
      f'   <- OUTIL : ce que orig fait sur du jamais-vu des memes classes approche-t-il le gold sur l oubli ?')
Lg = np.mean([SC.ce(g['fx'], y) for g in G], 0); wrong = np.mean([g['fx'].argmax(1) != y for g in G], 0)
print(f'loss par image moyennee sur les golds : quantiles 50/75/90/95/99 % = {np.round(np.quantile(Lg, [.5, .75, .9, .95, .99]), 3).tolist()} ; part des images ratees par tous les golds {np.mean(wrong == 1):.4f}, par au moins un {np.mean(wrong > 0):.4f}')
# vers quelle classe
pr = np.concatenate([g['fx'].argmax(1) for g in G]); yy = np.tile(y, len(G)); e = pr != yy
print(f'erreurs des golds : {e.mean():.4f} ; dont vers une classe de la meme famille {np.mean(grp[pr[e]] == grp[yy[e]]):.3f} (hasard ~{np.mean([np.mean(grp == grp[c]) for c in fc]):.3f}) ; vers une des 45 classes de l oubli {np.isin(pr[e], fc).mean():.3f}')
ag = np.mean([(G[i]['fx'].argmax(1) == G[j]['fx'].argmax(1))[(G[i]['fx'].argmax(1) != y) & (G[j]['fx'].argmax(1) != y)].mean() for i in range(len(G)) for j in range(i + 1, len(G))]) if len(G) > 1 else float('nan')
print(f'quand deux golds se trompent sur la meme image, ils choisissent la meme mauvaise classe dans {ag:.3f} des cas')
cnt = np.bincount(y, minlength=NC); per = [(c, cnt[c], np.mean(wrong[y == c]), Lg[y == c].mean()) for c in fc]
print('corr (part de la classe oubliee, taux d erreur du gold par classe) =', round(float(np.corrcoef([p[1] for p in per], [p[2] for p in per])[0, 1]), 3))
print('5 classes les plus ratees :', [(nm[c], int(n), round(float(er), 2)) for c, n, er, _ in sorted(per, key=lambda p: -p[2])[:5]])
conf = {}
for a, b in zip(yy[e], pr[e]): conf[(a, b)] = conf.get((a, b), 0) + 1
print('confusions principales :', [(nm[a], '->', nm[b], n) for (a, b), n in sorted(conf.items(), key=lambda kv: -kv[1])[:8]])
# un gold tenu a l'ecart contre les autres = plafond pratique du score
if len(G) > 2:
    cs = [SC.score(G[i], O, [g for j, g in enumerate(G) if j != i], L) for i in range(len(G))]
    print('PLAFOND : un gold note contre les AUTRES golds -> oubli', np.round([c['forget'] for c in cs], 3).tolist(), 'loss', np.round([c['loss'] for c in cs], 3).tolist(), 'pred', np.round([c['pred'] for c in cs], 3).tolist(),
          'retain', np.round([c['retain'] for c in cs], 3).tolist(), 'final', np.round([c['final'] for c in cs], 3).tolist())
cs = [SC.score(G[i], O, G, L) for i in range(len(G))]; print('un gold note contre TOUS les golds (lui compris) -> oubli', np.round([c['forget'] for c in cs], 3).tolist(), 'retain', np.round([c['retain'] for c in cs], 3).tolist())

try: P = torch.load(f'{R}/prep_{orig}.pt')
except Exception as ex: P = None; print('pas de prep.pt :', ex)
if P is not None:
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    yhat = P['yhat'].numpy(); print(f'\n--- (b) PREDIRE depuis orig seul ce que le gold ratera (pseudo-etiquettes justes : {(yhat == y).mean():.4f})')
    pgy = pg[ar, y]; hard = (wrong >= 0.5).astype(int)
    print(f'{"mesure":18s} {"Spearman vs loss gold":>22s} {"AUC vs image ratee":>20s}')
    comps = {k: v.numpy() for k, v in P['comps'].items()}; rk = lambda v: np.argsort(np.argsort(v)) / (len(v) - 1)
    comps['combine3'] = rk(comps['instabilite']) + rk(comps['knn_impurete']) + rk(comps['oof_difficulte']); comps['combine4'] = comps['combine3'] + rk(comps['tta_dur'])
    comps['hasard'] = np.random.default_rng(0).random(N)
    for k, v in comps.items(): print(f'{k:18s} {spearmanr(v, Lg)[0]:22.3f} {roc_auc_score(hard, v):20.3f}')
    print('quelle mauvaise classe ? (images ou la moyenne des golds a T=2 ne met pas la vraie classe en tete : %d)' % int((pg.argmax(1) != y).sum()))
    pw = pg.copy(); pw[ar, y] = 0; pw = pw / pw.sum(1, keepdims=True); tw = pw.argmax(1); sel = pg.argmax(1) != y
    for k, w in P['w'].items():
        w = w.numpy(); klw = (pw * (np.log(np.clip(pw, 1e-12, None)) - np.log(np.clip(w, 1e-12, None)))).sum(1)
        t5 = np.argsort(-w, 1)[:, :5]; print(f'  {k:8s} top1 = mauvaise classe du gold {np.mean(w.argmax(1) == tw):.3f} (images ratees {np.mean(w.argmax(1)[sel] == tw[sel]):.3f}) ; dans le top5 {np.mean((t5 == tw[:, None]).any(1)):.3f} ; KL(gold_faux||w) {klw.mean():.3f}')
    print('\n--- OUTIL : estimer mean/std de la loss du gold depuis des mesures faites sur orig seul')
    oof = P['oof'].numpy(); lo = -np.log(np.clip(oof[ar, yhat], 1e-6, None)); lt = comps['tta_dur']; li = comps['instabilite']
    print(f'verite gold : mean {mu:.4f} std {sd:.4f} acc {np.mean([SC.acc(g["fx"], y) for g in G]):.4f}')
    print(f'tete validee croisee (bloc 8) : loss moy {lo.mean():.4f} sd {lo.std():.4f} acc {np.mean(oof.argmax(1) == yhat):.4f}')
    print(f'vues dures (couleur+flou+recadrage) : loss moy {lt.mean():.4f} sd {lt.std():.4f} | instabilite (vues fortes) : moy {li.mean():.4f} sd {li.std():.4f} | kNN impurete moy {comps["knn_impurete"].mean():.4f}')
    pc = lambda v: np.array([v[y == c].mean() for c in fc]); gl = pc(Lg)
    for k, v in [('oof', lo), ('tta_dur', lt), ('instabilite', li), ('knn', comps['knn_impurete'])]: print(f'  par classe (45 points) : corr({k}, loss gold) = {np.corrcoef(pc(v), gl)[0, 1]:.3f}')
    # forme de la cible T=2 : la formule S1 q2 = 0.85 norm(sqrt(q)) + 0.15 w reproduit-elle p_gold a partir de la vraie cible T=1 ?
    q1 = np.mean([np.exp(SC.lsm(g['fx'])) for g in G], 0)
    for pw_, mix in [(0.5, 0.15), (0.5, 0.0), (0.7, 0.0), (1.0, 0.0), (0.35, 0.0)]:
        h = np.clip(q1, 1e-12, None) ** pw_; h = h / h.sum(1, keepdims=True); w = q1.copy(); w[ar, y] = 0; w = w / np.clip(w.sum(1, keepdims=True), 1e-12, None); q2 = (1 - mix) * h + mix * w
        print(f'  forme T=2 : puissance {pw_} melange {mix} -> KL(p_gold || formule) = {(pg * (lpg - np.log(np.clip(q2, 1e-12, None)))).sum(1).mean():.4f}   (KL(p_gold||orig) = {SC.kl(lpg, SC.lsm(O["fx"], 2.0)).mean():.3f})')
