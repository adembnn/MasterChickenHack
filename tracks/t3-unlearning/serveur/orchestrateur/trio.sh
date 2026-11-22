cd ~/eac/t3/t3-methodes
PY=../.venv/bin/python
for S in err oof rand; do
  T=S30_$S; N=PGS30_$S
  ls -la cache/tgt_$T.pt | awk '{print $5, $9}'
  (nohup $PY code/train.py --anchor mix --lam 3 --steps 1500 --wce 1 --name $N --tgt $T --fset all > logs/$N.log 2>&1 &)
  (nohup bash code/finish.sh $N $T > logs/finish_$N.log 2>&1 &)
done
sleep 20; tail -1 logs/PGS30_err.log | cut -c1-120; nvidia-smi --query-gpu=memory.used --format=csv,noheader
cat diag/row_PS30_err.md diag/row_PS30_oof.md diag/row_PS30_rand.md 2>/dev/null | cut -c1-330
