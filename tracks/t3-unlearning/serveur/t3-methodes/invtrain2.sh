cd ~/eac/t3/t3-methodes; PY=../.venv/bin/python; T=RDt24r_70_l1016
for p in $(pgrep -u adem -f "finish.sh PGinv"); do pkill -P $p; kill $p; done
for lx in 3 10; do N=PGinv${lx}_RDt24r_70
  (nohup $PY code/train.py --anchor mix --wce 1 --lam 3 --steps 1500 --seed 1 --extnpz $HOME/eac/t3/t3-methodes/cache/inv_orig.npz --lamx $lx --name $N --tgt $T --fset all > logs/$N.log 2>&1 &)
  (nohup bash code/finish.sh $N $T > logs/finish_$N.log 2>&1 &); echo "lance $N"; done
sleep 40; for lx in 3 10; do tail -c 200 logs/PGinv${lx}_RDt24r_70.log | tr '\r' '\n' | tail -1 | cut -c1-120; done; grep -m1 "ancre ext" logs/PGinv3_RDt24r_70.log
