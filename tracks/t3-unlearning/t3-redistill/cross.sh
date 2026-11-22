#!/bin/bash
# h200c : croisements sur la base RDt24r_70_l1016 (images = rang redistill moyen 4 blocs, classe = redistill une graine), un seul facteur change chacun
cd ~/eac/t3/t3-redistill; export T3ME=~/eac/t3/t3-redistill/tm; PY=~/eac/t3/.venv/bin/python; POOL="0.4:crop,0.2:mask,0.4:color,0.6:blur,0.4:noise"; T=$HOME/eac/t3/t3-redistill/targets
$PY - <<'PYEOF' 2>&1 | grep -v -i warn
import numpy as np, torch, os
T = os.path.expanduser('~/eac/t3/t3-redistill/targets/'); R = os.path.expanduser('~/eac/t3/t3-redistill/tm/cache/')
y = torch.from_numpy(np.load(R + 'exact_orig_logits.npy')).argmax(1); N = len(y); ar = torch.arange(N)
P = {k: torch.from_numpy(np.load(T + f'redistill_{k}.npy')) for k in ['k4m3', 'k6m4', 'k8m2']}
def rk(s): return s.argsort().argsort().float() / (N - 1)
sc = {k: rk(1 - p[ar, y]) for k, p in P.items()}; c = (sc['k4m3'] + sc['k6m4'] + sc['k8m2']) / 3
off = P['k4m3'].clone(); off[ar, y] = 0; off = off / off.sum(1, keepdim=True).clamp_min(1e-9); E = off * c[:, None]; E[ar, y] = 1 - c
np.save(T + 'redistill_rang_c468.npy', E.numpy().astype(np.float32))
top = {k: set(v.argsort(descending=True)[:2450].tolist()) for k, v in list(sc.items()) + [('consensus', c)]}
ks = list(top); print('images communes dans les 70 % du haut (sur 2450) : ' + ' ; '.join(f'{ks[i]}/{ks[j]} {len(top[ks[i]] & top[ks[j]])}' for i in range(len(ks)) for j in range(i + 1, len(ks))))
print('correlation de rang : ' + ' ; '.join(f'{a}/{b} {torch.corrcoef(torch.stack([sc[a], sc[b]]))[0, 1]:.3f}' for a, b in [('k4m3', 'k6m4'), ('k4m3', 'k8m2'), ('k6m4', 'k8m2')]))
PYEOF
C="--wsrc ext:$T/redistill_tau2.4.npy --wext 1.0"
mk() { $PY tmcode2/mk_tgt.py "$@" 2>&1 | grep "^RD" | cut -c1-28; }
mk RDt24r_70_l1016 --pool $POOL --f 0.70 --llo 10 --lhi 16.5 --rank ext:$T/redistill_k4m3.npy $C
mk RDt24r_80_l1016 --pool $POOL --f 0.80 --llo 10 --lhi 16.5 --rank ext:$T/redistill_k4m3.npy $C
mk RDt24r_60_l1016 --pool $POOL --f 0.60 --llo 10 --lhi 16.5 --rank ext:$T/redistill_k4m3.npy $C
mk RDt24r_70_l1117 --pool $POOL --f 0.70 --llo 11 --lhi 17.5 --rank ext:$T/redistill_k4m3.npy $C
mk RDt24r6_70_l1016 --pool $POOL --f 0.70 --llo 10 --lhi 16.5 --rank ext:$T/redistill_k6m4.npy $C
mk RDt24rc_70_l1016 --pool $POOL --f 0.70 --llo 10 --lhi 16.5 --rank ext:$T/redistill_rang_c468.npy $C
$PY code/nonflip.py RDt24r_70_l1016 $T/redistill_k4m3_tau2.4.npy RDt24rn_70_l1016 2>&1 | grep -v -i warn
$PY code/shape.py RDt24r_70_l1016 RDt24r_80_l1016 RDt24r_60_l1016 RDt24r_70_l1117 RDt24r6_70_l1016 RDt24rc_70_l1016 RDt24rn_70_l1016 2>&1 | grep -v -i warn
echo FIN_CIBLES
code/cand.sh RDt24r_80_l1016 RDt24r_60_l1016 RDt24r_70_l1117 RDt24r6_70_l1016 RDt24rc_70_l1016 RDt24rn_70_l1016
