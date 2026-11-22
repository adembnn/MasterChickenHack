cd ~/eac/t3/t3-methodes
PY=/home/hermes/eac/kit/.venv/bin/python
for N in f30 f40; do
  n=PG_T85_${N}_l814c; t=T85_${N}_l814
  until grep -q "^saved" logs/$n.log 2>/dev/null; do sleep 10; done
  $PY code/dash.py --fset all --tgt $t --tag all_ --combo $n=$n --save $n > logs/dashorch_$n.log 2>&1
  ls -la candidates/$n.pt >> logs/dashorch_$n.log 2>&1
done
