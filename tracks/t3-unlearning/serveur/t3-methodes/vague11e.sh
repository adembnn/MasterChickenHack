cd ~/eac/t3/t3-methodes; PY=../.venv/bin/python
ls cache/tgt_T85_f70_l814.pt cache/tgt_T85_f80_l814.pt
run() { N=$1; T=$2; shift 2
  (nohup $PY code/train.py --anchor mix --wce 1 "$@" --name $N --tgt $T --fset all > logs/$N.log 2>&1 &)
  (nohup bash code/finish.sh $N $T > logs/finish_$N.log 2>&1 &); echo "lance $N"; }
run PGs3kl6_f70_l814c T85_f70_l814 --lam 6 --steps 3000
run PGs6kl25_f70_l814c T85_f70_l814 --lam 25 --steps 6000
run PGB6l6_f70_l814c T85_f70_l814 --lam 6 --steps 1500 --blocks 6
run PGs3kl10_f80_l814c T85_f80_l814 --lam 10 --steps 3000
run PGs45l15_f80_l814c T85_f80_l814 --lam 15 --steps 4500
run PGs3kl6_f80_l814c T85_f80_l814 --lam 6 --steps 3000
sleep 40; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; for n in PGs3kl6_f70 PGs6kl25_f70 PGB6l6_f70 PGs3kl10_f80 PGs45l15_f80 PGs3kl6_f80; do echo "$n $(tail -c 160 logs/${n}_l814c.log | tr '\r' '\n' | tail -1 | cut -c1-60)"; done
