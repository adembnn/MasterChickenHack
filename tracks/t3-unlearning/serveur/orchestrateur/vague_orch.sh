cd ~/eac/t3/t3-methodes
PY=../.venv/bin/python
POOL="0.4:crop,0.2:mask,0.4:color,0.6:blur,0.4:noise"
mkdir -p logs
for spec in "f25_l814 0.25 8 14" "f35_l814 0.35 8 14" "f50_l814 0.50 8 14" "f25_l1218 0.25 12 18" "f30_l1218 0.30 12 18" "f40_l1218 0.40 12 18" "f30_l1624 0.30 16 24" "f40_l1624 0.40 16 24"; do
  set -- $spec; T=T85_$1; N=PG_${T}c
  [ -s cache/tgt_$T.pt ] || $PY code/mk_tgt.py $T --pool $POOL --f $2 --llo $3 --lhi $4 > logs/mk_$T.log 2>&1
  ls -la cache/tgt_$T.pt | awk '{print $5, $9}'
  (nohup $PY code/train.py --anchor mix --lam 3 --steps 1500 --wce 1 --name $N --tgt $T --fset all > logs/$N.log 2>&1 &)
  (nohup bash code/finish.sh $N $T > logs/finish_$N.log 2>&1 &)
done
sleep 30; for f in logs/PG_T85_f25_l814c.log logs/PG_T85_f40_l1624c.log; do tail -2 $f | cut -c1-150; done; ls -la py 2>/dev/null | cut -c1-80; nvidia-smi --query-gpu=memory.used --format=csv,noheader
