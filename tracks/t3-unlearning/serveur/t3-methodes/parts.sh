cd ~/eac/t3/t3-methodes; PY=../.venv/bin/python; P85=0.4:crop,0.2:mask,0.4:color,0.6:blur,0.4:noise
R=/home/adem/eac/t3/t3-redistill/targets
for f in 50 55 65 75; do
  T=RDt24r_${f}_l1016; N=PG_${T}c
  [ -s cache/tgt_$T.pt ] || $PY code/mk_tgt.py $T --pool $P85 --f 0.$f --llo 10 --lhi 16.5 --rank ext:$R/redistill_k4m3.npy --wsrc ext:$R/redistill_tau2.4.npy --wext 1.0 2>&1 | tail -1 | cut -c1-160
  [ -s cache/tgt_$T.pt ] || { echo "cible $T absente"; continue; }
  (nohup $PY code/train.py --anchor mix --wce 1 --lam 3 --steps 1500 --name $N --tgt $T --fset all > logs/$N.log 2>&1 &)
  (nohup bash code/finish.sh $N $T > logs/finish_$N.log 2>&1 &); echo "lance $N"
done
# controle : la cible 70 refaite ici est-elle identique a celle de redistill ?
$PY code/mk_tgt.py RDt24r_70_ctrl --pool $P85 --f 0.70 --llo 10 --lhi 16.5 --rank ext:$R/redistill_k4m3.npy --wsrc ext:$R/redistill_tau2.4.npy --wext 1.0 > /dev/null 2>&1
$PY - <<'PYE'
import torch
a=torch.load('cache/tgt_RDt24r_70_l1016.pt',map_location='cpu'); b=torch.load('cache/tgt_RDt24r_70_ctrl.pt',map_location='cpu')
print('ecart max cible 70 refaite vs originale :', float((a['z']-b['z']).abs().max()), 'memes images :', bool((a['flip_idx'].sort().values==b['flip_idx'].sort().values).all()))
PYE
sleep 20; nvidia-smi --query-gpu=memory.used --format=csv,noheader
