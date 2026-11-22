cd ~/eac/t3/orchestrateur
(nohup bash sub.sh ~/eac/t3/t3-methodes/candidates/PG_RDt24_60_l915c.pt > sub_PG_RDt24_60_l915c.log 2>&1 &)
cd ~/eac/t3/t3-methodes; PY=../.venv/bin/python
run() { N=$1; T=$2; shift 2
  (nohup $PY code/train.py --anchor mix --wce 1 "$@" --name $N --tgt $T --fset all > logs/$N.log 2>&1 &)
  (nohup bash code/finish.sh $N $T > logs/finish_$N.log 2>&1 &); echo "lance $N"; }
T=T85_f70_l814
run PGs3kl6_f70_l814c $T --lam 6 --steps 3000
run PGs3kl10_f70_l814c $T --lam 10 --steps 3000
run PGs45l10_f70_l814c $T --lam 10 --steps 4500
run PGs45l15_f70_l814c $T --lam 15 --steps 4500
run PGs6kl15_f70_l814c $T --lam 15 --steps 6000
run PGs6kl25_f70_l814c $T --lam 25 --steps 6000
run PGB6s3kl10_f70_l814c $T --lam 10 --steps 3000 --blocks 6
run PGB6l6_f70_l814c $T --lam 6 --steps 1500 --blocks 6
sleep 30; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
