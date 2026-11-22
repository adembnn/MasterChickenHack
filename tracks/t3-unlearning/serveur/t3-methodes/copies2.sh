cd ~/eac/t3/t3-methodes; PY=../.venv/bin/python; P85=0.4:crop,0.2:mask,0.4:color,0.6:blur,0.4:noise
R=/home/adem/eac/t3/t3-redistill/targets; C15=$HOME/eac/t3/t3-methodes/cache/rd_k4m8_tau1.5.npy
mk() { T=$1; shift; [ -s cache/tgt_$T.pt ] || $PY code/mk_tgt.py $T --pool $P85 --f 0.70 --rank ext:$R/redistill_k4m3.npy --wext 1.0 "$@" > logs/mk_$T.log 2>&1; ls cache/tgt_$T.pt > /dev/null 2>&1 || echo "cible $T absente"; }
mk RDq_70_l511 --llo 5 --lhi 11.5 --wsrc ext:$R/redistill_k4m8_tau2.0.npy
mk RDq15_70_l713 --llo 7 --lhi 13.5 --wsrc ext:$C15
mk RDq15_70_l612 --llo 6 --lhi 12.5 --wsrc ext:$C15
run() { N=$1; T=$2; shift 2
  (nohup $PY code/train.py --anchor mix --wce 1 --lam 3 --steps 1500 --seed 1 "$@" --name $N --tgt $T --fset all > logs/$N.log 2>&1 &)
  (nohup bash code/finish.sh $N $T > logs/finish_$N.log 2>&1 &); echo "lance $N"; }
run CPv4_RDq_70_l511 RDq_70_l511 --wcev 4
run CPv5_RDq_70_l511 RDq_70_l511 --wcev 5
run PG_RDq15_70_l713_s1 RDq15_70_l713
run PG_RDq15_70_l612_s1 RDq15_70_l612
sleep 30; nvidia-smi --query-gpu=memory.used --format=csv,noheader
