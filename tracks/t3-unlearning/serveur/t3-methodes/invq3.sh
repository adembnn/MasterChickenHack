cd ~/eac/t3/t3-methodes; PY=../.venv/bin/python; T=RDq_70_l915; N=PGinv3_RDq_70_l915_s1
(nohup $PY code/train.py --anchor mix --wce 1 --lam 3 --steps 1500 --seed 1 --extnpz $HOME/eac/t3/t3-methodes/cache/inv_orig.npz --lamx 3 --name $N --tgt $T --fset all > logs/$N.log 2>&1 &)
(nohup bash code/finish.sh $N $T > logs/finish_$N.log 2>&1 &); echo "lance $N"
sleep 30; nvidia-smi --query-gpu=memory.used --format=csv,noheader; tail -c 150 logs/$N.log | tr '\r' '\n' | tail -1 | cut -c1-90
