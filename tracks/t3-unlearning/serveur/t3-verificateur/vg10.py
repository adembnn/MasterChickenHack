import os, sys, torch, numpy as np
C = os.path.expanduser('~/eac/t3/t3-methodes/candidates'); a = torch.load(f'{C}/PG_T85_f60_l814c.pt', map_location='cpu', weights_only=True); b = torch.load(f'{C}/G10_f60.pt', map_location='cpu', weights_only=True)
print('cles identiques', list(a) == list(b), '; float32', all(v.dtype == torch.float32 for v in b.values()), '; fini', all(bool(torch.isfinite(v).all()) for v in b.values()), '; taille Mo', round(os.path.getsize(f'{C}/G10_f60.pt') / 1e6, 2))
diff = [k for k in a if not torch.equal(a[k], b[k])]; print('tenseurs differents de f60 :', diff)
rows = [i for i in range(100) if not torch.equal(a['head.weight'][i], b['head.weight'][i])]; print('lignes de tete modifiees :', rows)
pairs = [(17, 91), (47, 68), (59, 63), (56, 73), (74, 82)]; ok = all(torch.equal(b['head.weight'][i], a['head.weight'][j]) and torch.equal(b['head.weight'][j], a['head.weight'][i]) and b['head.bias'][i] == a['head.bias'][j] and b['head.bias'][j] == a['head.bias'][i] for i, j in pairs)
print('echange exact des 5 paires (poids et biais) :', ok)
lg = np.load(os.path.expanduser('~/eac/t3/t3-methodes/cache/exact_orig_logits.npy')); cnt = np.bincount(lg.argmax(1), minlength=100); print('images du forget set dans ces 10 classes :', int(sum(cnt[i] + cnt[j] for i, j in pairs)))
f = np.array([19.9, 29.6, 34.4, 38.9, 48.1, 57.2]); s = np.array([0.1487, 0.1962, 0.2179, 0.2400, 0.2779, 0.3009]); c = np.polyfit(f, s, 2); print('ajustement quadratique : sommet a', round(-c[1] / (2 * c[0]), 1), '% ; valeurs predites 66 %, 70 %, 76 %, 80 % :', [round(float(np.polyval(c, x)), 4) for x in (66, 70, 76, 80)], '; residu max', round(float(np.abs(np.polyval(c, f) - s).max()), 4))
