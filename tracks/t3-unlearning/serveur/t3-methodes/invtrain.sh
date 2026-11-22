cd ~/eac/t3/t3-methodes; PY=../.venv/bin/python; T=RDt24r_70_l1016
for lx in 3 10; do N=PGinv${lx}_RDt24r_70
  (nohup $PY code/train.py --anchor mix --wce 1 --lam 3 --steps 1500 --seed 1 --extnpz cache/inv_orig.npz --lamx $lx --name $N --tgt $T --fset all > logs/$N.log 2>&1 &)
  (nohup bash code/finish.sh $N $T > logs/finish_$N.log 2>&1 &); echo "lance $N"; done
sleep 25; tail -c 200 logs/PGinv3_RDt24r_70.log | tr '\r' '\n' | tail -2 | cut -c1-120
