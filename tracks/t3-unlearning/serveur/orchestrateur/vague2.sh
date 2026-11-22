cd ~/eac/t3/t3-methodes
PY=../.venv/bin/python
POOL="0.4:crop,0.2:mask,0.4:color,0.6:blur,0.4:noise"
grep -c "robust" code/train.py code/common.py 2>/dev/null
if grep -q "robust" code/train.py code/common.py; then
for spec in "f30_l814 0.30 8 14" "f30_l1218 0.30 12 18" "f30_l1624 0.30 16 24" "f35_l1218 0.35 12 18" "f35_l1624 0.35 16 24" "f25_l1624 0.25 16 24"; do
  set -- $spec; T=T85_$1; N=PGr_${T}c
  [ -s cache/tgt_$T.pt ] || $PY code/mk_tgt.py $T --pool $POOL --f $2 --llo $3 --lhi $4 > logs/mk_$T.log 2>&1
  (nohup $PY code/train.py --anchor mix --lam 3 --steps 1500 --wce 1 --fviews robust --name $N --tgt $T --fset all > logs/$N.log 2>&1 &)
  (nohup bash code/finish.sh $N $T > logs/finish_$N.log 2>&1 &)
done
# mêmes cibles nouvelles sans rotations, pour comparer à réglage égal
for spec in "f35_l1218 0.35 12 18" "f35_l1624 0.35 16 24" "f25_l1624 0.25 16 24"; do
  set -- $spec; T=T85_$1; N=PG_${T}c
  until [ -s cache/tgt_$T.pt ]; do sleep 3; done
  (nohup $PY code/train.py --anchor mix --lam 3 --steps 1500 --wce 1 --name $N --tgt $T --fset all > logs/$N.log 2>&1 &)
  (nohup bash code/finish.sh $N $T > logs/finish_$N.log 2>&1 &)
done
sleep 40; tail -1 logs/PGr_T85_f30_l814c.log | cut -c1-140; nvidia-smi --query-gpu=memory.used --format=csv,noheader
fi
