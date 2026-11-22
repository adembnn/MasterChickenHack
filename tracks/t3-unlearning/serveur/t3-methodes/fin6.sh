cd ~/eac/t3/t3-methodes; t=RDt24_70
for n in "$@"; do
  [ -s candidates/$n.pt ] && continue
  ../.venv/bin/python code/dash.py --fset all --tgt $t --tag all_ --combo $n=$n --save $n > logs/dash_$n.log 2>&1
  grep -E "^\| $n" logs/dash_$n.log > diag/row_$n.md; head -1 diag/row_$n.md | cut -c1-170
done
date +%H:%M:%S; tail -2 ~/eac/t3/orchestrateur/sub_PG_RDm3t24_70_l1016c.log | cut -c1-200
