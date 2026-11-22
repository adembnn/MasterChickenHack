cd ~/eac/t3/t3-methodes; PY=../.venv/bin/python
run() { N=$1; T=$2; shift 2
  (nohup $PY code/train.py --wce 1 --steps 1500 --seed 1 "$@" --name $N --tgt $T --fset all > logs/$N.log 2>&1 &)
  (nohup bash code/finish.sh $N $T > logs/finish_$N.log 2>&1 &); echo "lance $N"; }
run L1_RDq_70_l915 RDq_70_l915 --anchor mix --lam 1
run L1_RDq_70_l713 RDq_70_l713 --anchor mix --lam 1
run L05_RDq_70_l713 RDq_70_l713 --anchor mix --lam 0.5
run L05_RDq_70_l612 RDq_70_l612 --anchor mix --lam 0.5
run L02_RDq_70_l612 RDq_70_l612 --anchor mix --lam 0.2
sleep 30; nvidia-smi --query-gpu=memory.used --format=csv,noheader; tail -c 150 logs/L1_RDq_70_l915.log | tr '\r' '\n' | tail -1 | cut -c1-100
