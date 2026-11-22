cd ~/eac/t3/t3-methodes
PY=../.venv/bin/python; POOL="0.4:crop,0.2:mask,0.4:color,0.6:blur,0.4:noise"
for spec in "f70_l814 0.70 8 14" "f80_l814 0.80 8 14" "f70_l610 0.70 6 10" "f80_l610 0.80 6 10"; do
  set -- $spec; T=T85_$1; N=PG_${T}c
  [ -s cache/tgt_$T.pt ] || $PY code/mk_tgt.py $T --pool $POOL --f $2 --llo $3 --lhi $4 > logs/mk_$T.log 2>&1
  (nohup $PY code/train.py --anchor mix --lam 3 --steps 1500 --wce 1 --name $N --tgt $T --fset all > logs/$N.log 2>&1 &)
  (nohup bash code/finish.sh $N $T > logs/finish_$N.log 2>&1 &)
done
sleep 20; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
