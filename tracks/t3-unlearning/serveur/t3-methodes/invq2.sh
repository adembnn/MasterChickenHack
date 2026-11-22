cd ~/eac/t3/t3-methodes; PY=../.venv/bin/python; T=RDq_70_l915
for lx in 1 3; do N=PGinvA${lx}_RDq_70_l915_s1
  pgrep -u adem -f "finish.sh $N" > /dev/null || (nohup bash code/finish.sh $N $T > logs/finish_$N.log 2>&1 &)
  (nohup $PY code/train.py --anchor mix --wce 1 --lam 3 --steps 1500 --seed 1 --extnpz $HOME/eac/t3/t3-methodes/cache/inv_abs.npz --lamx $lx --name $N --tgt $T --fset all > logs/$N.log 2>&1 &); echo "relance $N"; done
sleep 45; nvidia-smi --query-gpu=memory.used --format=csv,noheader; for lx in 1 3; do tail -c 150 logs/PGinvA${lx}_RDq_70_l915_s1.log | tr '\r' '\n' | tail -1 | cut -c1-90; done
