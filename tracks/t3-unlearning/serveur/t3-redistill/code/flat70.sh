#!/bin/bash
# h200c : palier 70 % prepare d'avance. Classe redistill, forme de cible de la reference (logits hors pli / 2,4), deux bandes de loss pour encadrer la loss du fichier standard (4,68 / 4,30)
cd ~/eac/t3/t3-redistill; export T3ME=~/eac/t3/t3-redistill/tm OMP_NUM_THREADS=4; PY=~/eac/t3/.venv/bin/python; C=~/eac/t3/t3-redistill/tmcode2; T=~/eac/t3/t3-redistill/targets; POOL="0.4:crop,0.2:mask,0.4:color,0.6:blur,0.4:noise"
$PY $C/mk_tgt.py RDt24_70_l915 --pool $POOL --f 0.70 --llo 9 --lhi 15.5 --rank err --wsrc ext:$T/redistill_tau2.4.npy --wext 1.0 2>&1 | grep "^RD" | cut -c1-40
$PY $C/mk_tgt.py RDt24_70_l1016 --pool $POOL --f 0.70 --llo 10 --lhi 16.5 --rank err --wsrc ext:$T/redistill_tau2.4.npy --wext 1.0 2>&1 | grep "^RD" | cut -c1-40
$PY - <<'PYEOF' 2>&1 | grep -v -i warn
import torch, numpy as np, torch.nn.functional as F, os
R = os.path.expanduser('~/eac/t3/t3-redistill/tm/cache/'); lg0 = torch.from_numpy(np.load(R + 'exact_orig_logits.npy')); y = lg0.argmax(1); N = len(y)
ref = torch.load(R + 'tgt_T85_f70_l814.pt'); print('reference args', {k: v for k, v in ref['meta']['args'].items() if k in ('f', 'llo', 'lhi', 'rank', 'wsrc', 'eps', 'soft', 'gam')})
for n in ['T85_f70_l814', 'RDt24_70_l915', 'RDt24_70_l1016']:
    d = torch.load(R + f'tgt_{n}.pt'); z = d['z']; fl = d['flip_idx']; nf = torch.ones(N, dtype=torch.bool); nf[fl] = False; p2 = F.softmax(z / 2, 1); ce = F.cross_entropy(z, y, reduction='none')
    same = (set(fl.tolist()) == set(ref['flip_idx'].tolist())); cl = (z[ref['flip_idx']].argmax(1) == ref['z'][ref['flip_idx']].argmax(1)).float().mean()
    print(f'{n:16s} | {len(fl)} bascules (memes images que la reference : {same}) : max-prob T2 {p2[fl].max(1)[0].mean():.3f} | non basculees p2(classe orig) {p2[nf, y[nf]].mean():.3f} | loss cible {ce.mean():.2f} / {ce.std():.2f} | meme classe fausse que la reference {cl:.3f}')
PYEOF
one() { t=$1; n=PG_${t}c; cd $T3ME
  $C/retry.sh logs/$n.log $PY $C/train.py --steps 1500 --wce 1 --lam 3 --anchor mix --name $n --tgt $t --fset all || return 1
  for i in 1 2 3 4 5 6 7 8; do $PY $C/dash.py --fset all --tgt $t --tag all_ --combo $n=$n --save $n > logs/dash_$n.log 2>&1 && break; sleep 30; done
  grep -E "^\| $n" logs/dash_$n.log > diag/row_$n.md
  $PY $C/validate_t3.py candidates/$n.pt --reference ~/eac/t3/model.pt 2>&1 | tail -n 1 >> diag/row_$n.md; sha256sum candidates/$n.pt | cut -c1-16 >> diag/row_$n.md; }
one RDt24_70_l915 & one RDt24_70_l1016 &
wait; echo FIN_70
D=~/eac/t3/t3-redistill/tm/candidates
$PY ~/eac/t3/t3-redistill/code/profil.py T85_f70_l814 ~/eac/t3/t3-redistill/ref/PG_T85_f70_l814c.pt RDt24_70_l915 $D/PG_RDt24_70_l915c.pt RDt24_70_l1016 $D/PG_RDt24_70_l1016c.pt 2>&1 | grep "^|"
echo FIN_PROFIL70
