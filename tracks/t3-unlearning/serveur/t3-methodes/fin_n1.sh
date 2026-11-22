cd ~/eac/t3/t3-methodes; PY=../.venv/bin/python; P85=0.4:crop,0.2:mask,0.4:color,0.6:blur,0.4:noise
R=/home/adem/eac/t3/t3-redistill/targets
run() { N=$1; T=$2; shift 2
  (nohup $PY code/train.py --anchor mix --wce 1 --lam 3 --steps 1500 "$@" --name $N --tgt $T --fset all > logs/$N.log 2>&1 &)
  (nohup bash code/finish.sh $N $T > logs/finish_$N.log 2>&1 &); echo "lance $N"; }
for f in 68 72; do T=RDt24r_${f}_l1016
  [ -s cache/tgt_$T.pt ] || $PY code/mk_tgt.py $T --pool $P85 --f 0.$f --llo 10 --lhi 16.5 --rank ext:$R/redistill_k4m3.npy --wsrc ext:$R/redistill_tau2.4.npy --wext 1.0 > /dev/null 2>&1
  run PG_${T}c $T; done
run PGwcev2_RDt24r_70 RDt24r_70_l1016 --wcev 2
run PGB6_RDt24r_70 RDt24r_70_l1016 --blocks 6
run PG_RDt24r_70_l915c RDt24r_70_l915
ls -la $R/redistill_rang_c468.npy 2>/dev/null | awk '{print $5,$9}'
T=RDt24c468_70_l1016; [ -s cache/tgt_$T.pt ] || $PY code/mk_tgt.py $T --pool $P85 --f 0.70 --llo 10 --lhi 16.5 --rank ext:$R/redistill_rang_c468.npy --wsrc ext:$R/redistill_tau2.4.npy --wext 1.0 2>&1 | tail -1 | cut -c1-120
[ -s cache/tgt_$T.pt ] && run PG_${T}c $T
sleep 30; nvidia-smi --query-gpu=memory.used --format=csv,noheader
