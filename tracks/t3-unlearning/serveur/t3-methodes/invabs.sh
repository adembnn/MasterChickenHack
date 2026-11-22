cd ~/eac/t3/t3-methodes; PY=../.venv/bin/python; T=RDt24r_70_l1016
$PY - <<'PYE'
import numpy as np
d=np.load('cache/inv_orig.npz'); a=d['absent']; np.savez('cache/inv_abs.npz', images=d['images'][a], cls=d['cls'][a]); print('inv_abs', d['images'][a].shape)
PYE
for lx in 1 3 6; do N=PGinvA${lx}_RDt24r_70
  (nohup $PY code/train.py --anchor mix --wce 1 --lam 3 --steps 1500 --seed 1 --extnpz $HOME/eac/t3/t3-methodes/cache/inv_abs.npz --lamx $lx --name $N --tgt $T --fset all > logs/$N.log 2>&1 &)
  (nohup bash code/finish.sh $N $T > logs/finish_$N.log 2>&1 &); echo "lance $N"; done
sleep 30; nvidia-smi --query-gpu=memory.used --format=csv,noheader; tail -c 150 logs/PGinvA3_RDt24r_70.log | tr '\r' '\n' | tail -1 | cut -c1-100
