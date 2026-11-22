cd ~/eac/t3/t3-methodes; PY=../.venv/bin/python; P85=0.4:crop,0.2:mask,0.4:color,0.6:blur,0.4:noise
R=/home/adem/eac/t3/t3-redistill/targets
ls -la $R/redistill_k4m8.npy $R/redistill_k4m8_tau2.0.npy | awk '{print $5,$9}'
$PY - <<'PYE'
import numpy as np
R='/home/adem/eac/t3/t3-redistill/targets/'
p=np.load(R+'redistill_k4m8.npy').astype(np.float64); q=np.load(R+'redistill_k4m8_tau2.0.npy').astype(np.float64)
print('k4m8 somme lignes', p.sum(1)[:3], 'max moyen', p.max(1).mean(), '| tau2.0 max moyen', q.max(1).mean())
# verifie la relation tau2.0 = p^(1/2) renormalise
t=p**(1/2.0); t/=t.sum(1,keepdims=True); print('ecart max avec p^(1/2):', np.abs(t-q).max())
for tau in (1.5, 1.0):
    t=p**(1/tau); t/=t.sum(1,keepdims=True); np.save('cache/rd_k4m8_tau%s.npy'%tau, t.astype(np.float32)); print('tau',tau,'max moyen',t.max(1).mean())
PYE
mk() { T=$1; shift; [ -s cache/tgt_$T.pt ] || $PY code/mk_tgt.py $T --pool $P85 --f 0.70 --rank ext:$R/redistill_k4m3.npy --wext 1.0 "$@" > logs/mk_$T.log 2>&1; ls cache/tgt_$T.pt > /dev/null 2>&1 || echo "cible $T absente"; }
mk RDq_70_l814 --llo 8 --lhi 14.5 --wsrc ext:$R/redistill_k4m8_tau2.0.npy
mk RDq_70_l713 --llo 7 --lhi 13.5 --wsrc ext:$R/redistill_k4m8_tau2.0.npy
mk RDq_70_l612 --llo 6 --lhi 12.5 --wsrc ext:$R/redistill_k4m8_tau2.0.npy
mk RDq15_70_l915 --llo 9 --lhi 15.5 --wsrc ext:$HOME/eac/t3/t3-methodes/cache/rd_k4m8_tau1.5.npy
mk RDq10_70_l915 --llo 9 --lhi 15.5 --wsrc ext:$HOME/eac/t3/t3-methodes/cache/rd_k4m8_tau1.0.npy
run() { N=$1; T=$2; shift 2
  (nohup $PY code/train.py --anchor mix --wce 1 --lam 3 --steps 1500 --seed 1 "$@" --name $N --tgt $T --fset all > logs/$N.log 2>&1 &)
  (nohup bash code/finish.sh $N $T > logs/finish_$N.log 2>&1 &); echo "lance $N"; }
run CPv2_RDq_70_l814 RDq_70_l814 --wcev 2
run CPv2_RDq_70_l713 RDq_70_l713 --wcev 2
run CPv3_RDq_70_l713 RDq_70_l713 --wcev 3
run CPv3_RDq_70_l612 RDq_70_l612 --wcev 3
run PG_RDq15_70_l915_s1 RDq15_70_l915
run PG_RDq10_70_l915_s1 RDq10_70_l915
sleep 35; nvidia-smi --query-gpu=memory.used --format=csv,noheader; tail -c 150 logs/CPv2_RDq_70_l814.log | tr '\r' '\n' | tail -1 | cut -c1-100
