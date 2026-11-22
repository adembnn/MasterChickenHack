cd ~/eac/t3/t3-methodes; PY=../.venv/bin/python
ls -la cache/tgt_RDt24_70.pt; ls cache | wc -l
run() { N=$1; T=$2; shift 2
  (nohup $PY code/train.py --anchor mix --wce 1 "$@" --name $N --tgt $T --fset all > logs/$N.log 2>&1 &)
  (nohup bash code/finish.sh $N $T > logs/finish_$N.log 2>&1 &); echo "lance $N"; }
T=RDt24_70
run PGRseed1_70 $T --lam 3 --steps 1500 --seed 1
run PGRs3kl6_70 $T --lam 6 --steps 3000
run PGRs3kl10_70 $T --lam 10 --steps 3000
run PGRs45l10_70 $T --lam 10 --steps 4500
run PGRs45l15_70 $T --lam 15 --steps 4500
run PGRs6kl20_70 $T --lam 20 --steps 6000
run PGRB6l6_70 $T --lam 6 --steps 1500 --blocks 6
run PGRB6s3kl10_70 $T --lam 10 --steps 3000 --blocks 6
sleep 45; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; for f in logs/PGR*.log; do echo "$f $(tail -c 160 $f | tr '\r' '\n' | tail -1 | cut -c1-70)"; done
