"""Mesure d'orig sur chaque jeu d'images (img/*.npz) : précision par rapport à la classe de l'image source, entropie croisée (moyenne, écart-type),
confiance, matrice de confusion, le tout aussi sous augmentations. Sortie : meas/eval_<tag>.json + meas/logits_<tag>.npz ; tableau imprimé.
usage : evalnm.py [tag ...]   (sans argument : tous les npz + le forget set lui-même)"""
import nm, sys, os, glob, json, time, numpy as np, torch, torch.nn.functional as F
while os.path.exists(nm.ROOT + '/logs/hold') and not os.environ.get('NOHOLD'): time.sleep(20)   # attend la fin des générations relancées à la main
m = nm.orig(); X = nm.forget(); y_all = np.load(nm.MEAS + '/forget_logits.npy').argmax(1)
tags = sys.argv[1:] or (['forget'] + sorted(os.path.basename(p)[:-4] for p in glob.glob(nm.IMG + '/*.npz')))
def stats(z, y):
    z = z.double(); yt = torch.from_numpy(y).long(); ce = F.cross_entropy(z, yt, reduction='none'); p = F.softmax(z, 1); p2 = F.softmax(z / 2, 1)
    top2 = z.topk(2, 1).values
    return dict(n=len(y), acc=float((z.argmax(1) == yt).float().mean()), ce_mean=float(ce.mean()), ce_std=float(ce.std(unbiased=False)), ce_median=float(ce.median()),
                ce_q=[float(q) for q in torch.quantile(ce, torch.tensor([.1, .25, .5, .75, .9, .95, .99], dtype=torch.double))],
                maxprob=float(p.max(1).values.mean()), maxprob_T2=float(p2.max(1).values.mean()), p_true=float(p.gather(1, yt[:, None]).mean()), p_true_T2=float(p2.gather(1, yt[:, None]).mean()),
                entropy=float(-(p * p.clamp_min(1e-12).log()).sum(1).mean()), entropy_T2=float(-(p2 * p2.clamp_min(1e-12).log()).sum(1).mean()), margin=float((top2[:, 0] - top2[:, 1]).mean()))
print(f"{'jeu':28s} {'augmentation':22s} {'n':>5s} {'acc':>6s} {'CE moy':>7s} {'CE é-t':>7s} {'maxp':>6s} {'maxpT2':>6s} {'H':>6s}")
for tag in tags:
    if tag == 'forget': im, y, src = X, y_all, np.arange(len(X))
    else: d = np.load(f'{nm.IMG}/{tag}.npz'); im, y, src = d['images'], d['y'], d['src']
    res = {}; lg = {}
    for an, fn in nm.AUG.items():
        z = nm.logits(m, im, None if an == 'propre' else fn); res[an] = stats(z, y); lg[an] = z.numpy().astype(np.float32); r = res[an]
        print(f"{tag:28s} {an:22s} {r['n']:5d} {r['acc']:.4f} {r['ce_mean']:7.4f} {r['ce_std']:7.4f} {r['maxprob']:.4f} {r['maxprob_T2']:.4f} {r['entropy']:.4f}", flush=True)
    z = torch.from_numpy(lg['propre']); pred = z.argmax(1).numpy(); cm = np.zeros((100, 100), int); np.add.at(cm, (y, pred), 1)
    p2 = F.softmax(z.double() / 2, 1).numpy(); cls = sorted(set(y.tolist()))
    res['_confusion'] = cm.tolist(); res['_acc_par_classe'] = {int(c): float((pred[y == c] == c).mean()) for c in cls}
    res['_pT2_moyen_par_classe'] = {int(c): p2[y == c].mean(0).round(5).tolist() for c in cls}
    json.dump(res, open(f'{nm.MEAS}/eval_{tag}.json', 'w')); np.savez_compressed(f'{nm.MEAS}/logits_{tag}.npz', y=y, src=src, **{k.replace(' ', '_'): v for k, v in lg.items()})
